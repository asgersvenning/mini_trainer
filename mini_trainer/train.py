import os
import random
from argparse import ArgumentParser
from typing import (Any, Callable, Concatenate, Dict, List, Optional, Tuple,
                    Type, Union)

import numpy as np
import torch
import torchvision
from torch import nn

from .builders import (base_dataloader_builder, base_lr_schedule_builder,
                       base_model_builder, default_training_augmentation)
from .trainer import train
from .utils import (average_checkpoints, debug_augmentation, parse_class_index,
                    save_on_master)


def main(
    input : str,
    output : str = ".",
    model : str = "efficientnet_v2_s",
    checkpoint : Optional[List[str]]=None,
    weights : Optional[str]=None,
    data_index : Optional[str]=None,
    class_index : Optional[str]=None,
    fine_tune : bool=False,
    learning_rate : float=0.0001,
    epochs: int=15,
    batch_size : int=16,
    warmup_epochs : float=2.0,
    name: Optional[str]=None,
    device : str="cuda:0",
    dtype : str="float16",
    seed : Optional[int]=None,
    spec_model_dataloader : Callable[
        Concatenate[
            str,
            str,
            ...
        ],
        Tuple[
            Dict[str, Any],
            Dict[str, Any]
        ]
    ]=parse_class_index,
    spec_model_dataloader_kwargs : Dict[str, Any]={},
    model_builder : Callable[
        Concatenate[
            str, 
            int, 
            Optional[str], 
            bool, 
            torch.dtype, 
            torch.device, 
            ...
        ], 
        Tuple[
            torch.nn.Module, 
            Callable[[torch.Tensor], torch.Tensor]
        ]
    ]=base_model_builder,
    model_builder_kwargs : Dict[str, Any]={},
    dataloader_builder : Callable[
        Concatenate[
            str, 
            str, 
            List[str], 
            Dict[str, int], 
            Callable[[torch.Tensor], torch.Tensor], 
            int, 
            torch.device, 
            torch.dtype,
            ...
        ],
        Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            Union[
                torchvision.transforms.Compose,
                Callable[[torch.Tensor], torch.Tensor]
            ]
        ]
    ]=base_dataloader_builder,
    dataloader_builder_kwargs : Dict[str, Any]={
        "resize_size" : 256, 
        "train_proportion" : 0.9
    },
    augmentation_builder : Callable[
        Concatenate[...],
        torchvision.transforms.Compose
    ]=default_training_augmentation,
    augmentation_builder_kwargs : Dict[str, Any]={},
    criterion_builder : Union[
        Callable[
            Concatenate[...],
            nn.modules.loss._Loss
        ],
        Type[nn.modules.loss._Loss]
    ] = nn.CrossEntropyLoss,
    criterion_kwargs : Dict[str, Any]={"label_smoothing" : 0.1},
    optimizer_builder : Union[
        Callable[
            Concatenate[...],
            torch.optim.Optimizer
        ],
        Type[torch.optim.Optimizer]
    ]=torch.optim.AdamW,
    optimizer_kwargs : Dict[str, Any]={"weight_decay" : 1e-4},
    lr_schedule_builder : Callable[
        Concatenate[
            torch.optim.Optimizer,
            float,
            int,
            int,
            int,
            ...
        ],
        torch.optim.lr_scheduler.LRScheduler
    ]=base_lr_schedule_builder,
    lr_schedule_builder_kwargs : Dict[str, Any]={
        "min_factor" : 1 / 10**6, 
        "start_factor" : 1 / 10**2
    }
) -> None:
    """
    Train a classifier.

    Args:
        model (str, optional):
            Name of the model type from the torchvision model zoo.
            See: https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights.
            Default is 'efficientnet_v2_s'. Not case-sensitive.

        input (str):
            Path to a directory containing a subdirectory for each class, where
            each subdirectory's name corresponds to the class name. (required)

        output (str, optional):
            Root directory for all created files and directories.
            Default is current working directory '.'.

        checkpoint (Optional[List[str]], optional):
            Path to one or more checkpoint files for restarting training.
            If multiple files are supplied, training is restarted from an 'average'
            of checkpoint states. Default is None.

        weights (Optional[str], optional):
            Model weights used to initialize model before training.
            Default is None.

        data_index (Optional[str], optional):
            Path to a JSON file containing three arrays with keys 'path', 'split',
            and 'class', representing a structured dataset.
            Default is None.

        class_index (str, optional):
            Path to a JSON file containing the mapping from class names to indices.
            If the file does not exist, it will be created based on subdirectories
            found under `output` if it is set. Default is 'class_index.json'.

        fine_tune (bool, optional):
            If True, update only the classifier weights during training.
            Default is False.

        learning_rate (bool, optional):
            Base learning rate for training. 
            Influences the magnitude, but not shape of the learning rate curve. 
            Default is 0.0001.

        epochs (int, optional):
            Number of training epochs. Default is 15.

        batch_size (int, optional):
            Number of images per mini-batch for training and validation.
            Default is 16.

        warmup_epochs (float, optional):
            Number of warmup epochs at the start of training.
            Default is 2.0.

        name (Optional[str], optional):
            Name of the output model. If not provided, a descriptive name
            will be inferred from other arguments. Default is None.

        device (str, optional):
            Device used for training (e.g., 'cuda:0', 'cpu').
            Default is 'cuda:0'.

        dtype (str, optional):
            PyTorch data type for images during training and validation
            (e.g., 'float16'). The model parameters are always stored in float32,
            and training is done with autocasting. Default is 'float16'.

        seed (Optional[int], optional):
            Initial seed for Python's random number generator to ensure reproducibility,
            especially for train/validation splits. Default is None.
        
        **kwargs: Additional arguments to be documented. 
            All additional arguments are not available from the commandline, but exist to enable usage of 
            the `train.py` and `predict.py` functionality with custom models, loss functions, data loaders etc.

    Returns:
        None
    """
        # Prepare state
    if name is None:
        name = f'{model}_{"fine_tune" if fine_tune else "full"}_e{epochs}'
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.seed(seed)
    
    input_dir = os.path.abspath(input)
    output_dir = os.path.abspath(output)

    device : torch.device = torch.device(device)
    dtype : torch.dtype = getattr(torch, dtype)

    # Load additional information for model and dataloader instantiation
    # e.g. number of classes, class-to-index dictionary
    extra_model_kwargs, extra_dataloader_kwargs = spec_model_dataloader(
        path=class_index if class_index is not None else os.path.join(output_dir, "class_index.json"), 
        dir=input_dir,
        **spec_model_dataloader_kwargs
    )

    # Prepare model
    nn_model, model_preprocess = model_builder(
        model=model, 
        weights=weights, 
        fine_tune=fine_tune, 
        device=device, 
        dtype=dtype,
        **{**extra_model_kwargs, **model_builder_kwargs}
    ) 
    if not isinstance(nn_model, torch.nn.Module):
        raise TypeError(
            'Expected `model_builder` to return a tuple, where the first element'
            f'is an object inheriting from `torch.nn.Module`, but got `{type(nn_model)}`.'
        )
    
    # Prepare dataloader
    train_loader, val_loader = dataloader_builder(
        data_index=data_index,
        input_dir=input_dir,
        preprocess=model_preprocess,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        **{**extra_dataloader_kwargs, **dataloader_builder_kwargs}
    )
    if not isinstance(train_loader, torch.utils.data.DataLoader):
        raise TypeError(
            'Expected `dataloader_builder` to return an objects'
            f'inheriting from `torch.utils.data.DataLoader`, but got `{type(train_loader)}.'
        )
    if not isinstance(val_loader, torch.utils.data.DataLoader):
        raise TypeError(
            'Expected `dataloader_builder` to return an objects'
            f'inheriting from `torch.utils.data.DataLoader`, but got `{type(val_loader)}.'
        )

    augmentation = augmentation_builder(**augmentation_builder_kwargs)
    if not isinstance(augmentation, torchvision.transforms.Compose):
        raise TypeError(
            'Expected `augmentation_builder` to return an objects'
            f'inheriting from `torchvision.transforms.Compose`, but got `{type(augmentation)}.'
        )
    debug_augmentation(
        augmentation=augmentation,
        dataset=train_loader.dataset,
        output_dir=output_dir,
        strict=True
    )

    # Define training "hyperparameters"
    start_epoch = 0
    if warmup_epochs > epochs:
        raise ValueError(
            f'Number of warmup epochs ({warmup_epochs}) must be'
            f'lower than number of total epochs ({epochs}).'
        )

    # parameters = set_weight_decay(nn_model, 1e-3)
    optimizer_kwargs["lr"] = learning_rate 
    optimizer = optimizer_builder(nn_model.parameters(recurse=True), **optimizer_kwargs)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            'Expected `optimizer_builder` to return an object'
            f'inheriting from `torch.optim.Optimizer`, but got `{type(optimizer)}.'
        )
    criterion = criterion_builder(**criterion_kwargs)
    if not isinstance(criterion, torch.nn.modules.loss._Loss):
        raise TypeError(
            'Expected `criterion_builder` to return an object'
            f'inheriting from `torch.nn.modules.loss._Loss`, but got `{type(criterion)}.'
        )

    lr_scheduler = lr_schedule_builder(
        optimizer=optimizer,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        steps_per_epoch=len(train_loader),
        **lr_schedule_builder_kwargs
    )
    if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LRScheduler):
        raise TypeError(
            'Expected `lr_schedule_builder` to return an object'
            f'inheriting from `torch.optim.lr_scheduler.LRScheduler`, but got `{type(lr_scheduler)}.'
        )

    if checkpoint is not None:
        checkpoint_files = checkpoint
        if isinstance(checkpoint_files, list) and len(checkpoint_files) == 1:
            checkpoint_files = checkpoint_files[0]
        if isinstance(checkpoint_files, str):
            checkpoint_data = torch.load(checkpoint_files, device)
        else:
            checkpoint_data = average_checkpoints(checkpoint_files)
        nn_model.load_state_dict(checkpoint_data["model"])
        optimizer.load_state_dict(checkpoint_data["optimizer"])
        lr_scheduler.load_state_dict(checkpoint_data["lr_scheduler"])
        start_epoch = checkpoint_data["epoch"]
        if not isinstance(start_epoch, int):
            if isinstance(start_epoch, (torch.Tensor, np.ndarray)) and len(start_epoch) == 1:
                start_epoch = start_epoch.tolist()[0]
            if isinstance(start_epoch, float) and np.isclose(start_epoch % 1, 0):
                start_epoch = int(start_epoch)
            else:
                raise TypeError(f"Invalid 'start_epoch' value in {checkpoint}, found `{start_epoch}` but expected an `int`.")

    # Run training
    train(
        nn_model, 
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler,
        epochs,
        start_epoch,
        model_preprocess,
        augmentation,
        device=device,
        dtype=dtype,
        output_dir=output_dir,
        weight_store_rate=5
    )

    # Save result model
    nn_model.eval()
    save_on_master(nn_model.state_dict(), os.path.join(output_dir, f"{name}.pt"))

def cli():
    parser = ArgumentParser(
        prog = "train",
        description = "Train a classifier"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="path to a directory containing a subdirectory for each class, where the name of each subdirectory should correspond to the name of the class."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=".", required=False,
        help="Root directory for all created files and directories. Default is current working directory ('.')."
    )  
    parser.add_argument(
        "-m", "--model", type=str, default="efficientnet_v2_s", required=False,
        help="name of the model type from the torchvision model zoo (https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights). Not case-sensitive."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, nargs="+", required=False,
        help="Path to [a] checkpoint file(s) for restarting training. If multiple files are supplied training is restarted from an 'average' of checkpoint states."
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=False,
        help="Model weights used to initialize model before training."
    )
    parser.add_argument(
        "-D", "--data_index", type=str, required=False,
        help="JSON file containing three arrays with keys 'path', 'split' and 'class'. The arrays should all have equal lengths and can be considered \"columns\" in a table. The 'split' column should contain values 'train', 'validation' or other, and the 'class' column' should contain the the class *names* (not indices) for each file/path."
    )
    parser.add_argument(
        "-C", "--class_index", type=str, required=False,
        help="path to a JSON file containing the class name to index mapping. If it doesn't exist, one will be created based on the directories found under `output` if it is set."
    )
    parser.add_argument(
        "--fine-tune", action="store_true", required=False,
        help="Update only the classifier weights"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, required=False,
        help="Number of training epochs (default=15)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, required=False,
        help="Number of images used in each mini-batch for training/validation (default=16)."
    )
    parser.add_argument(
        "--warmup_epochs", type=float, default=2, required=False,
        help="Number of warmup epochs (default=2)."
    )
    parser.add_argument(
        "-n", "--name", type=str, required=False,
        help="name of the output model. If not provided, a helpful name will be inferred from the other arguments."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", required=False,
        help='Device used for training (default="cuda:0").'
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", required=False,
        help="PyTorch data type used for storing images for training/validation (default=float16). The model is always stored in float32, and training is done with autocasting."
    )
    parser.add_argument(
        "--seed", type=int, required=False,
        help="Set the initial seed for the RNG in the core Python library `random`. This is particularly important for reproducible train/validation splits."
    )
    main(**vars(parser.parse_args()))

if __name__ == "__main__":  
    cli()

