import os
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchvision.io import ImageReadMode

from mini_trainer.classifier import Classifier
from mini_trainer.utils import cosine_schedule_with_warmup, memory_proportion
from mini_trainer.utils.data import (get_image_data, parse_class_index,
                                     prepare_split, write_metadata)
from mini_trainer.utils.io import LazyDataset, make_read_and_resize_fn
from mini_trainer.utils.logging import MultiLogger


def get_dataset_dataloader(
        train_image_data : dict, 
        val_image_data : dict, 
        resize_size : Union[int, tuple[int, int]],
        batch_size : int=16, 
        num_workers : Optional[int]=None,
        device=torch.device("cpu"), 
        dtype=torch.float32
    ):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    else:
        if not (isinstance(resize_size, tuple) and len(resize_size) == 2 and all(map(lambda x : isinstance(x, int), resize_size))):
            raise TypeError(f'Invalid resize size passed, found {resize_size}, but expected an integer or a tuple of two integers')
        
    print(f"Building datasets with image size {resize_size}")
    dataset_is_small = memory_proportion((len(train_image_data["path"]) + len(val_image_data["path"]), *resize_size), device, dtype) < 0.25
    if dataset_is_small:
        train_tensor = prepare_split(train_image_data["path"], "Preprocessing training images...", resize_size, device, dtype)
        val_tensor = prepare_split(val_image_data["path"], "Preprocessing validation images...", resize_size, device, dtype)

        train_labels = torch.tensor([
            cls[0] if isinstance(cls, (list, np.ndarray)) else cls 
            for cls in train_image_data["class"] 
        ]).long().to(device)
        val_labels = torch.tensor([
            cls[0] if isinstance(cls, (list, np.ndarray)) else cls
            for cls in val_image_data["class"]
        ]).long().to(device)

        train_dataset = TensorDataset(train_tensor, train_labels)
        val_dataset = TensorDataset(val_tensor, val_labels)
        
        # When the entire dataset is preloaded there is no need to use multiprocessing for dataloading
        num_workers = 0
    else:
        reader = make_read_and_resize_fn(ImageReadMode.RGB, resize_size, torch.device("cpu"), dtype)
        def proc_path_label(path_label : tuple[str, Union[int, list[int], np.ndarray]]):
            path, label = path_label
            if isinstance(label, (list, np.ndarray)):
                label = label[0]
            return reader(path), torch.tensor(label, dtype=torch.long)

        train_dataset = LazyDataset(proc_path_label, [(path, cls) for path, cls in zip(train_image_data["path"], train_image_data["class"])])
        val_dataset   = LazyDataset(proc_path_label, [(path, cls) for path, cls in zip(  val_image_data["path"],   val_image_data["class"])])

        if num_workers is None:
            num_workers = os.cpu_count() - 1
            num_workers -= num_workers % 2

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=not dataset_is_small,
        pin_memory_device="" if dataset_is_small else str(device)
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=num_workers, 
        pin_memory=not dataset_is_small,
        pin_memory_device="" if dataset_is_small else str(device)
    )

    return train_dataset, val_dataset, train_loader, val_loader

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
        **`build_logger`**: Builds the training diagnostics logger(s).
    """
    def __init__(self):
        pass

    @staticmethod
    def spec_model_dataloader(path : Optional[str]=None, dir : Optional[str]=None, *args, **kwargs):
        """
        Returns:
            (extra_model_kwargs, extra_dataloader_kwargs) (`tuple[dict[str, Any], dict[str, Any]]`): Extra keyword arguments for the model and dataloader building functions.
        """
        if args or kwargs:
            raise ValueError(f'`BaseBuilder.spec_model_dataloader` expects only `path` and `dir`, but got {len(args)} additional positional arguments ({args}) and {len(kwargs)} additional keyword arguments ({list(kwargs.keys())}).')
        return parse_class_index(path=path, dir=dir)

    @staticmethod
    def build_model(
            fine_tune : bool=False,
            cls : Type[Classifier]=Classifier,
            **kwargs : Any
        ) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
        """
        The mandatory keyword arguments depend on the class of `cls`, but are likely to be ["model_name", "weights", "device", "dtype" and "num_classes"] or a superset containing these.

        Returns:
            (model, model_preprocess) (`tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]`): The loaded model and an appropriate preprocessing function (e.g. RGB[0,1] normalizer).
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
            classes : list[str],
            cls2idx : dict[str, int],
            preprocess : Callable[[torch.Tensor], torch.Tensor],
            batch_size : int,
            device : torch.device,
            dtype = torch.dtype,
            data_index : Optional[str]=None,
            resize_size : Optional[int]=None,
            train_proportion : float=0.9,
            idx2cls : Optional[dict[int, str]]=None,
            num_workers : Optional[int]=None):
        """
        Returns:
            (train_loader, validation_loader) (`tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]`): The training and validation dataloaders.
        """
        # Prepare datasets/dataloaders
        if data_index is None:
            with NamedTemporaryFile() as tmpfile:
                write_metadata(input_dir, classes, cls2idx, tmpfile.name, train_proportion=train_proportion)
                train_image_data, val_image_data = get_image_data(tmpfile.name)
        else:
            train_image_data, val_image_data = get_image_data(data_index)

        train_dataset, val_dataset, train_loader, val_loader = get_dataset_dataloader(
            train_image_data=train_image_data, 
            val_image_data=val_image_data,
            resize_size=getattr(preprocess, "resize_size", resize_size),  
            batch_size=batch_size,
            num_workers=num_workers,
            device=device, 
            dtype=dtype
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
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, 
            lr_lambda=cosine_schedule_with_warmup(epochs * steps_per_epoch, warmup_epochs * steps_per_epoch, start_factor, min_factor)
        )
    
    @staticmethod
    def build_logger(**kwargs):
        return MultiLogger(**kwargs)