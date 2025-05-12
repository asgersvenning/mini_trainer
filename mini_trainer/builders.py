
import json
import os
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize
from tqdm import tqdm as TQDM

from .classifier import Classifier
from .utils import (convert2bf16, convert2fp16, convert2fp32,
                    cosine_schedule_with_warmup, get_image_data,
                    write_metadata)


# Base builder utilities
def parse_class_index(path : Optional[str]=None, dir : Optional[str]=None):
    """
    Accepts a path to 
    """
    if not os.path.exists(path):
        if dir is None or not os.path.isdir(dir):
            raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not \'{dir}\'.')
        cls2idx = {cls : i for i, cls in enumerate(sorted([f for f in map(os.path.basename, os.listdir(dir)) if os.path.isdir(os.path.join(dir, f))]))}
        if path is not None:
            with open(path, "w") as f:
                json.dump(cls2idx, f)
    else:
        with open(path, "rb") as f:
            cls2idx = json.load(f)
    cls = list(cls2idx.keys())
    idx2cls = {v : k for k, v in cls2idx.items()}
    ncls = len(idx2cls)
    return {"num_classes" : ncls}, {"classes" : cls, "class2idx" : cls2idx, "idx2class" : idx2cls}

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
            raise TypeError(f'Invalid resize size passed, found {resize_size}, but expected an integer or a tuple of two integers')
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

class BaseBuilder:
    """
    The base builder used in the `mini_trainer` training pipeline.

    Subclass this builder and override the relevant functions to alter the `mini_trainer` training pipeline.

    Methods:
        **`spec_model_dataloader`** : Parses or builds the model specification from the class index or input directory.
        **`build_model`**: Builds the model.
        **`build_dataloader`**: Builds the training and validation dataloaders.
        **`build_augmentation`**: Builds the augmentation method(s).
        **`build_optimizer`**: Builds the model optimizer method (e.g. SGD/ADAM).
        **`build_criterion`**: Builds the optimization criterion (i.e. loss function).
        **`build_lr_scheduler`**: Builds the learning rate scheduler (shape only, magnitude defined by optimizer).
    """
    def __init__(self):
        pass

    @staticmethod
    def spec_model_dataloader(path : Optional[str]=None, dir : Optional[str]=None, *args, **kwargs):
        """
        Returns:
            (extra_model_kwargs, extra_dataloader_kwargs) (`Tuple[Dict[str, Any], Dict[str, Any]]`): Extra keyword arguments for the model and dataloader building functions.
        """
        if args or kwargs:
            raise ValueError(f'`BaseBuilder.spec_model_dataloader` expects only `path` and `dir`, but got {len(args)} additional positional arguments ({args}) and {len(kwargs)} additional keyword arguments ({list(kwargs.keys())}).')
        return parse_class_index(path=path, dir=dir)

    @staticmethod
    def build_model(
            fine_tune : bool=False,
            cls : Type[Classifier]=Classifier,
            **kwargs : Any
        ) -> Tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
        """
        The mandatory keyword arguments depend on the class of `cls`, but are likely to be ["model_name", "weights", "device", "dtype" and "num_classes"] or a superset containing these.

        Returns:
            (model, model_preprocess) (`Tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]`): The loaded model and an appropriate preprocessing function (e.g. RGB[0,1] normalizer).
        """
        model, model_preprocess = cls.build(**kwargs)
        if fine_tune:
            for name, param in model.named_parameters():
                if param.requires_grad and not model._architecture_output_name in name:
                    param.requires_grad_(False)
        return model, model_preprocess

    @staticmethod
    def build_dataloader(
            input_dir : str,
            classes : List[str],
            class2idx : Dict[str, int],
            preprocess : Callable[[torch.Tensor], torch.Tensor],
            batch_size : int,
            device : torch.device,
            dtype = torch.dtype,
            data_index : Optional[str]=None,
            resize_size : Optional[int]=None,
            train_proportion : float=0.9,
            idx2class : Optional[Dict[int, str]]=None
        ):
        """
        Returns:
            (train_loader, validation_loader) (`Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]`): The training and validation dataloaders.
        """
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

    @staticmethod
    def build_augmentation():
        """
        Returns a training augmentation pipeline for normalized tensors.
        Assumes the input tensor is already normalized (e.g., in the range [0, 1] or standardized).

        Returns:
            transforms (`transforms.Compose`): A composition of augmentations.
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

    @staticmethod
    def build_optimizer(params, *args, optimizer_cls : Type[torch.optim.Optimizer]=torch.optim.AdamW, **kwargs):
        """
        Returns:
            optimizer (`torch.optim.Optimizer`): The model optimizer (e.g. `torch.optim.SGD` for standard gradient descent).
        """
        return optimizer_cls(params=params, *args, **kwargs)
    
    @staticmethod
    def build_criterion(*args, criterion_cls : Type[nn.modules.loss._Loss]=nn.CrossEntropyLoss, **kwargs):
        """
        Returns:
            loss_fn (`nn.modules.loss._Loss`): The loss function for optimization (e.g. `torch.nn.CrossEntropyLoss` for classification).
        """
        return criterion_cls(*args, **kwargs)

    @staticmethod
    def build_lr_scheduler(
            optimizer : torch.optim.Optimizer,
            epochs : int,
            warmup_epochs : int,
            steps_per_epoch : int,
            min_factor : float = 1e-6,
            start_factor : float = 1e-4
        ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Only the *shape* of the LR curve is defined here; the *magnitude* should be set in the optimizer. I suggest using `torch.optim.lr_scheduler.LambdaLR`.

        Returns:
            lr_scheduler (`torch.optim.lr_scheduler.LRScheduler`): The learning rate scheduler (shape only).
        """
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule_with_warmup(epochs * steps_per_epoch, warmup_epochs * steps_per_epoch, start_factor, min_factor))