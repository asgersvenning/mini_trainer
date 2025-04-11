
import json
import os
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize
from tqdm import tqdm as TQDM
from tqdm.contrib.concurrent import thread_map

from .classifier import Classifier, get_model
from .utils import (convert2bf16, convert2fp16, convert2fp32, get_image_data,
                    write_metadata)


def prepare_split(paths : List[str], desc="Preprocessing images for split...", resize_size : Union[int, Tuple[int, int]]=256, device=torch.device("cpu"), dtype=torch.float16):
    match dtype:
        case torch.float16:
            converter = convert2fp16
        case torch.float32:
            converter = convert2fp32
        case torch.bfloat16:
            converter = convert2bf16
        case _:
            raise ValueError("Only fp16 supported for now.")
    shape = resize_size if not isinstance(resize_size, int) and len(resize_size) == 2 else (resize_size, resize_size)
    tensor = torch.zeros((len(paths), 3, *shape), device=device, dtype=dtype)
    def write_one_image(args):
        idx, path = args
        try:
            tensor[idx] = resize(converter(decode_image(path, ImageReadMode.RGB)), shape).to(device)
        except Exception as e:
            e.add_note(f'Path: {path}')
            raise e
    # num_workers = os.cpu_count() - 1
    # num_workers -= num_workers % 2
    # thread_map(write_one_image, enumerate(paths), tqdm_class=TQDM, total=len(paths), desc=desc, max_workers=num_workers)
    [write_one_image(v) for v in TQDM(enumerate(paths), total=len(paths), desc=desc)]
    return tensor

def default_training_augmentation():
    """
    Returns a training augmentation pipeline for normalized tensors.
    Assumes the input tensor is already normalized (e.g., in the range [0, 1] or standardized).

    Returns:
        transforms.Compose: A composition of augmentations.
    """
    return tt.Compose([
        tt.RandomHorizontalFlip(p=0.5),
        tt.RandomVerticalFlip(p=0.5),
        tt.RandomRotation(degrees=15),
        tt.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # tt.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # # Convert back to tensor (in case some augmentations convert to PIL Image)
        # tt.ToTensor(),
    ])

def get_dataset_dataloader(
        train_image_data : Dict, 
        val_image_data : Dict, 
        class2idx : Dict[str, int], 
        resize_size : Union[int, Tuple[int, int]],
        batch_size : int=16, 
        device=torch.device("cpu"), 
        dtype=torch.float32
    ):
    if not isinstance(resize_size, int):
        if not (isinstance(resize_size, tuple) and len(resize_size) == 2 and all(map(lambda x : isinstance(x, int), resize_size))):
            raise TypeError(f'Invalid resize size passed, foun {resize_size}, but expected an integer or a tuple of two integers')
    print(f"Building datasets with image size {resize_size}")
    train_tensor = prepare_split(train_image_data["path"], "Preprocessing training images...", resize_size, device, dtype)
    val_tensor = prepare_split(val_image_data["path"], "Preprocessing validation images...", resize_size, device, dtype)

    train_labels = torch.tensor([class2idx[str(cls)] for cls in train_image_data["class"]]).long().to(device)
    val_labels = torch.tensor([class2idx[str(cls)] for cls in val_image_data["class"]]).long().to(device)

    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=0, 
        pin_memory=False
    )

    return train_dataset, val_dataset, train_loader, val_loader

def easy_get_dataset_dataloader(data_path, class_path):
    if not os.path.exists(class_path):
        raise FileNotFoundError(f'Species index file ({class_path}) for not found.')
    with open(class_path, "rb") as f:
        class2idx = json.load(f)
    return get_dataset_dataloader(*get_image_data(data_path), class2idx)

def base_model_builder(
        model : str,
        weights : Optional[str],
        fine_tune : bool,
        device : torch.device,
        dtype : torch.dtype,
        num_classes : int,
        **kwargs : Any
    ) -> Tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if len(kwargs) != 0:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(f"my_fun() got unexpected keyword argument(s): {unexpected}")
    model, head_name, model_preprocess = get_model(model)
    model : nn.Module
    if not isinstance(model, nn.Module):
        raise TypeError(f"Unknown model type `{type(model)}`, expected `{nn.Module}`")
    num_embeddings = getattr(model, head_name)[1].in_features
    if weights is not None:
        model = Classifier.load(model, weights, device=device, dtype=torch.float32)
    else:
        setattr(model, head_name, Classifier(num_embeddings, num_classes))
        model.to(device, torch.float32)
    if fine_tune:
        for name, param in model.named_parameters():
            if param.requires_grad and not head_name in name:
                param.requires_grad_(False)
    return model, model_preprocess

def base_dataloader_builder(
        data_index : str,
        input_dir : str,
        classes : List[str],
        class2idx : Dict[str, int],
        preprocess : Callable[[torch.Tensor], torch.Tensor],
        batch_size : int,
        device : torch.device,
        dtype = torch.dtype,
        resize_size : Optional[int]=None,
        train_proportion : float=0.9,
        idx2class : Optional[Dict[int, str]]=None
    ):
    # Prepare datasets/dataloaders
    if data_index is None:
        with NamedTemporaryFile() as tmpfile:
            write_metadata(input_dir, classes, tmpfile.name, train_proportion=train_proportion)
            train_image_data, val_image_data = get_image_data(tmpfile.name)
    else:
        train_image_data, val_image_data = get_image_data(data_index)

    train_dataset, val_dataset, train_loader, val_loader = get_dataset_dataloader(
        train_image_data, 
        val_image_data, 
        class2idx, 
        preprocess.resize_size if hasattr(preprocess, "resize_size") else resize_size, 
        batch_size,
        device, 
        dtype
    )

    return train_loader, val_loader

def base_lr_schedule_builder(
        optimizer : torch.optim.Optimizer,
        learning_rate : float,
        epochs : int,
        warmup_epochs : int,
        steps_per_epoch : int,
        min_factor : float=1 / 10**6,
        start_factor : float=1 / 10**2
    ):
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(epochs - warmup_epochs) * steps_per_epoch, 
        eta_min=learning_rate * min_factor
    )
    if warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=start_factor, 
            total_iters=warmup_epochs * steps_per_epoch
        )
    
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
        milestones=[warmup_epochs * steps_per_epoch]
    ) if warmup_epochs > 0 else main_lr_scheduler
    return lr_scheduler