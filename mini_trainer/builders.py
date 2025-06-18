import os
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.io import ImageReadMode

from mini_trainer.classifier import Classifier, last_layer_weights
from mini_trainer.utils import cosine_schedule_with_warmup, memory_proportion
from mini_trainer.utils.augmentation import SaltAndPepper
from mini_trainer.utils.data import (get_image_data, parse_class_index,
                                     write_metadata)
from mini_trainer.utils.io import LazyDataset, make_read_and_resize_fn
from mini_trainer.utils.logging import MultiLogger
from mini_trainer.utils.loss import class_weight_distribution_regularization


def get_dataset_dataloader(
        train_image_data : dict, 
        val_image_data : dict, 
        resize_size : Union[int, tuple[int, int]],
        batch_size : int=16, 
        num_workers : Optional[int]=None,
        subsample : Optional[int]=None,
        device=torch.device("cpu"), 
        dtype=torch.float32
    ):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    else:
        if not (isinstance(resize_size, tuple) and len(resize_size) == 2 and all(map(lambda x : isinstance(x, int), resize_size))):
            raise TypeError(f'Invalid resize size passed, found {resize_size}, but expected an integer or a tuple of two integers')
        
    print(f"Building datasets with image size {resize_size}")
    if subsample is None or subsample <= 1:
        pass
    else:
        train_image_data = {k : v[::subsample] for k, v in train_image_data.items()}
        val_image_data = {k : v[::subsample] for k, v in val_image_data.items()}
    
    dataset_shape = (len(train_image_data["path"]) + len(val_image_data["path"]), *resize_size)
    dataset_fits_in_cuda = False # memory_proportion(dataset_shape, device, dtype) < 0.25
    dataset_fits_in_cpu = memory_proportion(dataset_shape, "cpu", dtype) < 0.5
    dataset_fits_on_disk = memory_proportion(dataset_shape, "disk", dtype) < 0.5
    
    reader = make_read_and_resize_fn(resize_size, torch.device("cpu"), torch.uint8)
    def proc_path_label(path_label : tuple[str, Union[int, list[int], np.ndarray]]):
        path, label = path_label
        if isinstance(label, (list, np.ndarray)):
            label = label[0]
        return reader(path), torch.tensor(label, dtype=torch.long)

    train_dataset = LazyDataset(
        func=proc_path_label, 
        items=[(path, cls) for path, cls in zip(train_image_data["path"], train_image_data["class"])],
        cache=device if dataset_fits_in_cuda else "ram" if dataset_fits_in_cpu else "disk" if dataset_fits_on_disk else None
    )
    val_dataset = LazyDataset(
        func=proc_path_label, 
        items=[(path, cls) for path, cls in zip(  val_image_data["path"],   val_image_data["class"])],
        cache=device if dataset_fits_in_cuda else "ram" if dataset_fits_in_cpu else "disk" if dataset_fits_on_disk else None
    )

    if num_workers is None:
        num_workers = os.cpu_count() - 4
        # num_workers = max(int(num_workers * 3 / 4), num_workers - 4)
        num_workers -= num_workers % 2
        # num_workers = max(0, num_workers)
    if dataset_fits_in_cuda or dataset_fits_in_cpu:
        # When the entire dataset is preloaded there is no need to use multiprocessing for dataloading
        num_workers = 0

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    pin_memory = not (dataset_fits_in_cuda or dataset_fits_in_cpu)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True, # Ensures compatibility with batch normalization
        pin_memory=pin_memory,
        pin_memory_device=str(device) if pin_memory else "",
        persistent_workers=num_workers > 0 # pin_memory
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        num_workers=num_workers,
        pin_memory=False,
        pin_memory_device="",
        persistent_workers=num_workers > 0 # False
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
    def parameter_groups(
        model: nn.Module,
        head_lr: float,
        backbone_lr: float,
        head_weight_decay: float,
        backbone_weight_decay: float,
    ) -> list[dict[str, Any]]:
        """
        Groups all model parameters into 'head' and 'backbone'.
        The 'head' is identified by the attribute name stored in `model._backbone_output_name`.
        All other parameters are considered 'backbone'.
        This method does not filter by requires_grad; it groups all parameters.
        """
        if not hasattr(model, '_backbone_output_name'):
            raise AttributeError("Model does not have `_backbone_output_name` attribute to identify the head.")
        
        head_attr_name = getattr(model, '_backbone_output_name')
        if not isinstance(head_attr_name, str):
            raise TypeError(f"`model._backbone_output_name` must be a string, got {type(head_attr_name)}.")
        
        if not hasattr(model, head_attr_name):
            raise AttributeError(f"Model does not have an attribute named '{head_attr_name}' as specified by `_backbone_output_name`.")

        head_module = getattr(model, head_attr_name)
        if not isinstance(head_module, nn.Module):
            raise TypeError(f"The attribute '{head_attr_name}' (value of `_backbone_output_name`) must be an nn.Module.")

        head_module_param_ids = set(id(p) for p in head_module.parameters())

        actual_head_params = []
        actual_backbone_params = []

        for p in model.parameters(): # Iterate through all parameters the model exposes
            if id(p) in head_module_param_ids:
                actual_head_params.append(p)
            else:
                actual_backbone_params.append(p)
        
        param_groups = []
        if actual_head_params:
            param_groups.append({
                "params": actual_head_params, "lr": head_lr, "name": "head",
                "weight_decay": head_weight_decay
            })
        elif head_module_param_ids: # Head module exists and has params, but they aren't in model.parameters()? Unlikely.
             raise RuntimeError(f"Head module '{head_attr_name}' seems to have parameters, but none were found directly in `model.parameters()` for the head group.")


        if actual_backbone_params:
            param_groups.append({
                "params": actual_backbone_params, "lr": backbone_lr, "name": "backbone",
                "weight_decay": backbone_weight_decay
            })
        
        if not param_groups and list(model.parameters()):
            raise RuntimeError(f"Model has parameters, but no distinct head/backbone groups formed.")


        return param_groups

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
            subsample : int | None=None,
            idx2cls : Optional[dict[int, str]]=None,
            num_workers : Optional[int]=None
        ):
        """
        Returns:
            (train_label_cls, train_loader, validation_loader) (`tuple[np.ndarray, torch.utils.data.DataLoader, torch.utils.data.DataLoader]`): The training and validation dataloaders.
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
            subsample=subsample,
            device=device, 
            dtype=dtype
        )

        return train_image_data["class"], train_loader, val_loader

    @staticmethod
    def build_augmentation(dtype : torch.dtype):
        """
        Returns a training augmentation pipeline for normalized tensors.
        Assumes the input tensor is already normalized (e.g., in the range [0, 1] or standardized).

        Returns:
            transforms (`transforms.Compose`): A composition of augmentations.
        """
        return tt.Compose([
            tt.AugMix(severity=3),
            SaltAndPepper(proportion=(0.001, 0.05), probability=0.75),
            tt.RandomHorizontalFlip(),
            tt.RandomVerticalFlip(),
            tt.RandomRotation(15),
            tt.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            # # tt.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
            # tt.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # # Convert back to tensor (in case some augmentations convert to PIL Image)
            # tt.ToTensor()
        ])
    
    @classmethod
    def build_optimizer(
        cls,
        model: nn.Module,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float=1e-3,
        weight_decay: float=0.0001,
        backbone_lr: Optional[float]=None,
        backbone_weight_decay: Optional[float]=None,
        **optimizer_kwargs # Other optimizer_cls arguments (e.g., betas, eps for AdamW)
    ) -> torch.optim.Optimizer:
        """
        Builds an optimizer with separate parameter groups for head and backbone.
        All parameters of the model are assigned to groups.
        Requires `model` to have `_backbone_output_name` attribute.
        """
        head_lr = lr
        backbone_lr = backbone_lr or head_lr / 3
        head_weight_decay = weight_decay
        backbone_weight_decay = backbone_weight_decay or head_weight_decay

        param_groups = cls.parameter_groups(
            model, 
            head_lr=head_lr, backbone_lr=backbone_lr,
            head_weight_decay=head_weight_decay, backbone_weight_decay=backbone_weight_decay
        )

        if not param_groups and not list(model.parameters()): # Model has no parameters
            final_params_for_optimizer = [] # Optimizer on empty list
        elif not param_groups and list(model.parameters()): # Should be caught by group_parameters warning.
             raise ValueError("Model has parameters, but group_parameters returned no groups. This indicates an issue.")
        else:
            final_params_for_optimizer = param_groups
        
        # Default LR for the optimizer itself (used if a group has no LR or if not using groups)
        if 'lr' not in optimizer_kwargs:
            optimizer_kwargs['lr'] = head_lr # A sensible default

        # Clip gradients during backprop (prevent gradient explosion)
        backbone_params = [pg["params"] for pg in param_groups if pg["name"] == "backbone"][0]
        for p in backbone_params:
            if p.requires_grad:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))

        return optimizer_cls(params=final_params_for_optimizer, **optimizer_kwargs)
    
    @staticmethod
    def build_criterion(
            *args, 
            weighted : bool=False,
            labels : Optional[np.ndarray]=None, 
            num_classes : Optional[int]=None, 
            criterion_cls : Type[nn.modules.loss._Loss]=nn.CrossEntropyLoss, 
            device : Optional[torch.types.Device]=None,
            dtype : Optional[torch.dtype]=None,
            **kwargs
        ):
        """
        Returns:
            loss_fn (`nn.modules.loss._Loss`): The loss function for optimization (e.g. `torch.nn.CrossEntropyLoss` for classification).
        """
        if not weighted or labels is None or num_classes is None:
            return criterion_cls(*args, **kwargs)
        counts = torch.ones((num_classes, ))
        for cls_idx in labels:
            counts[cls_idx] += 1
        weights = 1 / (counts.mean() + counts)
        weights /= torch.mean(weights)
        return criterion_cls(*args, weight=weights.to(device, dtype), **kwargs)

    @staticmethod
    def build_regularizer(strength : float=1e-5, *args, **kwargs):
        strength = float(strength)
        if strength == 0:
            return lambda _: 0.
        return lambda model: strength * class_weight_distribution_regularization(last_layer_weights(model))

    @staticmethod
    def build_lr_scheduler(
            optimizer : torch.optim.Optimizer,
            epochs : int,
            warmup_epochs : float,
            steps_per_epoch : int,
            min_factor : float = 1e-6,
            start_factor : float = 1e-4
        ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Only the *shape* of the LR curve is defined here; the *magnitude* should be set in the optimizer. I suggest using `torch.optim.lr_scheduler.LambdaLR`.

        Returns:
            lr_scheduler (`torch.optim.lr_scheduler.LRScheduler`): The learning rate scheduler (shape only).
        """
        warmup_steps = round(warmup_epochs * steps_per_epoch)
        warmup_proportion = warmup_epochs / epochs
        head_schedule = cosine_schedule_with_warmup(epochs * steps_per_epoch, warmup_steps, start_factor, min_factor)
        
        backbone_warmup_steps = round(warmup_steps * (1 - warmup_proportion))
        _backbone_schedule = cosine_schedule_with_warmup(epochs * steps_per_epoch - warmup_steps, backbone_warmup_steps, start_factor, min_factor)
        def backbone_schedule(step : int):
            if step < warmup_steps:
                return 0
            return _backbone_schedule(step - warmup_steps)
        
        lr_lambdas = []
        for grp in optimizer.param_groups:
            name = grp["name"]
            match name:
                case "head":
                    lr_lambdas.append(head_schedule)
                case "backbone":
                    lr_lambdas.append(backbone_schedule)
                case _:
                    raise KeyError(f'Unknown parameter group "{name}" expected one of "head"/"backbone".')

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, 
            lr_lambda=lr_lambdas
        )
    
    @staticmethod
    def build_logger(**kwargs):
        return MultiLogger(**kwargs)
    

class AutoEmbedder(nn.Module):
    def __init__(self, original_model : Classifier):
        super().__init__()
        self.original_model = original_model
        self.embeddings = None
        self._hook_handle = None
        
        # Attempt to find the final nn.Linear layer
        last_linear_layer = None
        for module in self.original_model.modules():
            if isinstance(module, nn.Linear):
                last_linear_layer = module
        
        if last_linear_layer is None:
            if isinstance(self.original_model, nn.Linear): # Model itself is Linear
                last_linear_layer = self.original_model
            else: # Check last child
                children = list(self.original_model.children())
                if children and isinstance(children[-1], nn.Linear):
                    last_linear_layer = children[-1]
                else:
                    raise ValueError("Could not automatically find a final nn.Linear layer to hook.")
        
        self._hook_handle = last_linear_layer.register_forward_hook(self._capture_embeddings_hook)

    def _capture_embeddings_hook(self, module, input_args, output):
        # input_args[0] is the input tensor to the hooked nn.Linear layer
        if input_args:
            self.embeddings = input_args[0].detach() # .clone() if modification is a concern

    def forward(self, x):
        self.embeddings = None # Reset
        predictions = self.original_model(x)
        return predictions, self.embeddings

    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        self.remove_hook()