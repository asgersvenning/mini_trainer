import os
import random
from argparse import ArgumentParser
from typing import Any, Optional, Type

import numpy as np
import torch
import torchvision

from mini_trainer import Formatter
from mini_trainer.builders import BaseBuilder
from mini_trainer.trainer import train
from mini_trainer.utils import (average_checkpoints, increment_name_dir,
                                save_on_master)
from mini_trainer.utils.plot import debug_augmentation


def main(
    input : str,
    output : str = ".",
    checkpoint : Optional[list[str]]=None,
    class_index : Optional[str]=None,
    epochs : int=15,
    name: Optional[str]=None,
    device : str="cuda:0",
    dtype : str="float16",
    seed : Optional[int]=None,
    builder : Type[BaseBuilder]=BaseBuilder,
    spec_model_dataloader_kwargs : dict[str, Any]={},
    model_builder_kwargs : dict[str, Any]={
        "model_name" : "efficientnet_v2_s",
        "weights" : None,
        "fine_tune" : False
    },
    dataloader_builder_kwargs : dict[str, Any]={
        "bacth_size" : 16,
        "resize_size" : 256, 
        "train_proportion" : 0.9
    },
    augmentation_builder_kwargs : dict[str, Any]={},
    optimizer_builder_kwargs : dict[str, Any]={
        "lr" : 0.001,
        "weight_decay" : 1e-4
    },
    criterion_builder_kwargs : dict[str, Any]={"label_smoothing" : 0.1},
    lr_schedule_builder_kwargs : dict[str, Any]={
        "warmup_epochs" : 2.0
    },
    logger_builder_kwargs : dict[str, Any]={"verbose" : False}
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

        checkpoint (Optional[list[str]], optional):
            Path to one or more checkpoint files for restarting training.
            If multiple files are supplied, training is restarted from an 'average'
            of checkpoint states. Default is None.

        data_index (Optional[str], optional):
            Path to a JSON file containing three arrays with keys 'path', 'split',
            and 'class', representing a structured dataset.
            Default is None.

        class_index (str, optional):
            Path to a JSON file containing the mapping from class names to indices.
            If the file does not exist, it will be created based on subdirectories
            found under `output` if it is set. Default is 'class_index.json'.

        epochs (int, optional):
            Number of training epochs. Default is 15.

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

        builder (Type[BaseBuilder], optional):
            An object inheriting from `mini_trainer.builders.BaseBuilder`. This object 
            is responsible for instantiating the model, dataloader, augmentation, optimizer, 
            criterion (loss function) and learning rate scheduler.
        
        **kwargs: 
            Additional arguments are passed to the various builder methods. 
            See `mini_trainer.builders.BaseBuilder` for details.
    
    Returns:
        None
    """
    # Prepare state    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.seed(seed)
    
    input_dir = os.path.abspath(input)
    output_dir = os.path.abspath(output)

    name = increment_name_dir(name, output_dir)

    device : torch.device = torch.device(device)
    dtype : torch.dtype = getattr(torch, dtype)

    # Load additional information for model and dataloader instantiation
    # e.g. number of classes, class-to-index dictionary
    extra_model_kwargs, extra_dataloader_kwargs = builder.spec_model_dataloader(
        path=class_index if class_index is not None else os.path.join(output_dir, "class_index.json"), 
        dir=input_dir,
        **spec_model_dataloader_kwargs
    )

    # Prepare model
    nn_model, model_preprocess = builder.build_model(
        device=device, 
        dtype=torch.float32, # Loading the model with a lower precision leads to instable training, instead we use `torch.autocast` to facilitate mixed precision training
        **{**extra_model_kwargs, **model_builder_kwargs}
    ) 
    if not isinstance(nn_model, torch.nn.Module):
        raise TypeError(
            'Expected `model_builder` to return a tuple, where the first element'
            f'is an object inheriting from `torch.nn.Module`, but got `{type(nn_model)}`.'
        )
    
    # Prepare dataloader
    train_labels, train_loader, val_loader = builder.build_dataloader(
        input_dir=input_dir,
        preprocess=model_preprocess,
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

    augmentation = builder.build_augmentation(**augmentation_builder_kwargs)
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

    # Setup optimizer, criterion (loss function) and learning rate scheduler
    start_epoch = 0

    optimizer = builder.build_optimizer(params=nn_model.parameters(recurse=True), **optimizer_builder_kwargs)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            'Expected `optimizer_builder` to return an object'
            f'inheriting from `torch.optim.Optimizer`, but got `{type(optimizer)}.'
        )
    criterion = builder.build_criterion(
        labels=train_labels, 
        num_classes=extra_model_kwargs["num_classes"], 
        device=device,
        dtype=dtype,
        **criterion_builder_kwargs
    )
    if not isinstance(criterion, torch.nn.modules.loss._Loss):
        raise TypeError(
            'Expected `criterion_builder` to return an object'
            f'inheriting from `torch.nn.modules.loss._Loss`, but got `{type(criterion)}.'
        )

    lr_scheduler = builder.build_lr_scheduler(
        optimizer=optimizer,
        epochs=epochs,
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

    # Instantiate logger
    logger = builder.build_logger(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        output=output_dir,
        name=name+"_log",
        **logger_builder_kwargs
    )

    # Run training
    train(
        model=nn_model, 
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        epochs=epochs,
        start_epoch=start_epoch,
        preprocess=model_preprocess,
        augmentation=augmentation,
        device=device,
        dtype=dtype,
        output_dir=output_dir,
        weight_store_rate=5
    )

    # Store final logs (logs are stored continously, but this flushes the logs to disk)
    logger.save()

    # Save result model
    nn_model.eval()
    save_on_master(nn_model.state_dict(), os.path.join(output_dir, f"{name}.pt"))

def cli(description="Train a classifier", **kwargs):
    parser = ArgumentParser(
        prog="train",
        description=description,
        formatter_class=Formatter
    )

    if kwargs:
        for argname, args in kwargs.items():
            parser.add_argument(f'--{argname}', **args)

    input_args = parser.add_argument_group("Input [mandatory]")
    input_args.add_argument(
        "-i", "--input", type=str, required=True,
        help=
        "Path to a directory containing a subdirectory for each class,\n" 
        "where the name of each subdirectory should correspond to the name of the class."
    )
    out_args = parser.add_argument_group("Output [optional]")
    out_args.add_argument(
        "-o", "--output", type=str, default=".", required=False,
        help=
        "Root directory for all created files and directories.\n"
        "Default is current working directory ('.')."
    )
    out_args.add_argument(
        "-n", "--name", type=str, required=False,
        help=
        "Name of the output model.\n"
        "If not provided, a helpful name will be inferred from the other arguments."
    )
    out_args.add_argument(
        "-t", "--tensorboard", action="store_true", required=False,
        help="Enable tensorboard logging."
    )
    mod_args = parser.add_argument_group("Model [optional]")
    mod_args.add_argument(
        "-m", "--model", type=str, default="efficientnet_v2_s", required=False,
        help=
        "Name of the model type from the torchvision model zoo (not case-sensitive):\n"
        "https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights)"
    )
    mod_args.add_argument(
        "-c", "--checkpoint", type=str, nargs="+", required=False,
        help= 
        "Path to [a] checkpoint file(s) for restarting training.\n"
        "If multiple files are supplied training is restarted from an 'average' of checkpoint states."
    )
    mod_args.add_argument(
        "-w", "--weights", type=str, required=False,
        help="Model weights used to initialize model before training."
    )
    mod_args.add_argument(
        "-C", "--class_index", type=str, required=False,
        help=
        "path to a JSON file containing the class name to index mapping.\n"
        "If it doesn't exist, one will be created based on the directories found under `output` if it is set."
    )
    train_args = parser.add_argument_group("Training [optional]")
    train_args.add_argument(
        "-D", "--data_index", type=str, required=False,
        help=
        "JSON file containing three arrays with keys 'path', 'split' and 'class'.\n"
        "The arrays should all have equal lengths and can be considered \"columns\" in a table.\n"
        "The 'split' column should contain values 'train', 'validation' or other,\n"
        "and the 'class' column' should contain the the class *names* (not indices) for each file/path."
    )
    train_args.add_argument(
        "-e", "--epochs", type=int, default=15, required=False,
        help="Number of training epochs (default=15)."
    )
    train_args.add_argument(
        "--lr", "--learning_rate", default=0.001, required=False,
        help="Initial learning rate after warmup (default=0.001)."
    )
    train_args.add_argument(
        "--batch_size", type=int, default=16, required=False,
        help="Number of images used in each mini-batch for training/validation (default=16)."
    )
    train_args.add_argument(
        "--warmup_epochs", type=float, default=2.0, required=False,
        help="Number of warmup epochs (default=2.0)."
    )
    train_args.add_argument(
        "--label_smoothing", type=float, default=0.1, required=False,
        help="Label smoothing applied to training (default=0.1)."
    )
    train_args.add_argument(
        "--class_weighted", action="store_true", required=False,
        help="Add class-weights to cross entropy loss (or other criterion) proportional to the inverse log-counts."
    )
    train_args.add_argument(
        "--fine-tune", action="store_true", required=False,
        help="OBS: This should probably not be used. Update only the classifier weights."
    )
    cfg_args = parser.add_argument_group("Config [optional]")
    cfg_args.add_argument(
        "--subsample", type=int, default=None, required=False,
        help="Subsample the data for training and eval (useful for testing). Default is None (no subsampling)."
    )
    cfg_args.add_argument(
        "--device", type=str, default="cuda:0", required=False,
        help='Device used for training (default="cuda:0").'
    )
    cfg_args.add_argument(
        "--dtype", type=str, default="float16", required=False,
        help=
        "PyTorch data type used for storing images for training/validation (default=float16).\n" 
        "The model is always stored in float32, and training is done with autocasting."
    )
    cfg_args.add_argument(
        "--seed", type=int, required=False,
        help=
        "Set the initial seed for the RNG in the core Python library `random`.\n"
        "This is particularly important for reproducible train/validation splits."
    )
    args = vars(parser.parse_args())
    # Distribute builder arguments to the relevant functions
    args["model_builder_kwargs"] = {
        "model_type" : args.pop("model"),
        "weights" : args.pop("weights"),
        "fine_tune" : args.pop("fine_tune")
    }
    args["dataloader_builder_kwargs"] = {
        "data_index" : args.pop("data_index"),
        "batch_size" : args.pop("batch_size"),
        "resize_size" : 256, 
        "train_proportion" : 0.9,
        "subsample" : args.pop("subsample")
    }
    args["optimizer_builder_kwargs"] = {
        "lr" : args.pop("lr"),
        "weight_decay" : 1e-4
    }
    args["criterion_builder_kwargs"] = {
        "label_smoothing" : args.pop("label_smoothing"),
        "weighted" : args.pop("class_weighted")
    }
    args["lr_schedule_builder_kwargs"] = {
        "warmup_epochs" : args.pop("warmup_epochs"),
        "min_factor" : 1 / 10**6, 
        "start_factor" : 1 / 10**2
    }
    # Set reasonable default name for unspecified CLI training runs
    if args["name"] is None:
        args["name"] = \
        f'{args["model_builder_kwargs"]["model_type"]}_' \
        f'{"fine_tune" if args["model_builder_kwargs"]["fine_tune"] else "full"}_' \
        f'e{args["epochs"]}'
    if args.pop("tensorboard"):
        from mini_trainer.utils.tensorboard import TensorboardLogger
        from mini_trainer.utils.logging import MetricLoggerWrapper
        from torch.utils.tensorboard.writer import SummaryWriter
        
        run_name = increment_name_dir(args["name"], tensorboard_dir := os.path.join(args["output"], "tensorboard"))
        tensorboard_writer = SummaryWriter(os.path.join(tensorboard_dir, run_name), flush_secs=30)
        
        args["logger_builder_kwargs"] = {
            "verbose" : True,
            "logger_cls" : [MetricLoggerWrapper, TensorboardLogger],
            "logger_cls_extra_kwargs" : [{}, {"writer" : tensorboard_writer}]
        }
    
    # Call the Python training API
    return args

if __name__ == "__main__":  
    main(**cli())

