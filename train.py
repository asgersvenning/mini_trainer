import json
import os
import random
from argparse import ArgumentParser
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms as tt
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize
from tqdm import tqdm as TQDM

from trainer import train
from utils import (average_checkpoints, get_image_data, parse_class_index,
                   save_on_master, set_weight_decay, write_metadata)

_UNSUPPORTED_MODELS = [
    'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 
    'fcos_resnet50_fpn', 
    'keypointrcnn_resnet50_fpn', 
    'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 
    'mvit_v1_b', 'mvit_v2_s', 
    'raft_large', 'raft_small', 
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 
    'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 
    'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 
    'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32'
]

convert2fp16 = tt.ConvertImageDtype(torch.float16)
convert2bf16 = tt.ConvertImageDtype(torch.bfloat16)
convert2fp32 = tt.ConvertImageDtype(torch.float32)
convert2uint8 = tt.ConvertImageDtype(torch.uint8)

def preprocess(item, transform, func):
    if isinstance(item, str):
        path = str(item)
        if not os.path.exists(path):
            raise FileNotFoundError("Unable to find image: " + path)
        image = decode_image(path, ImageReadMode.RGB)
    elif isinstance(item, torch.Tensor):
        image = item
    else:
        raise TypeError(f"'item' must be of type `str` or `torch.Tensor`, not {type(item).__qualname__}")
    return transform(func(image))

def get_model(backbone_model: Union[str, torch.nn.Module], model_args: dict = {},
              classifier_name: Union[str, List[str]] = ["classifier", "fc"]):
    default_transform = None
    if isinstance(backbone_model, str):
        if backbone_model in _UNSUPPORTED_MODELS:
            raise ValueError(f"The model {backbone_model} is not supported.")
        default_weights = torchvision.models.get_model_weights(backbone_model).DEFAULT
        default_transform = default_weights.transforms(antialias=True)
        backbone_model = torchvision.models.get_model(backbone_model, weights=default_weights, **model_args)
    if not isinstance(backbone_model, nn.Module):
        raise ValueError("backbone_model must be a string or a torch.nn.Module")
    backbone_classifier_name = None
    if isinstance(classifier_name, str):
        classifier_name = [classifier_name]
    for name in classifier_name:
        if hasattr(backbone_model, name):
            backbone_classifier_name = name
            break
    if backbone_classifier_name is None:
        raise AttributeError(f"No classifier found with names {classifier_name}")

    return backbone_model, backbone_classifier_name, partial(preprocess, transform=default_transform, func=convert2fp16)

class Classifier(nn.Module):
    def __init__(self, in_features : int, out_features : int, hidden : bool=False):
        super().__init__()
        # Create a BatchNormalization Layer
        self.batch_norm = nn.BatchNorm1d(in_features)

        # Create one hidden layer
        self.hidden = hidden and nn.Linear(in_features, in_features)

        # Create a standard linear layer.
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # Set the bias to -1 and freeze it.
        with torch.no_grad():
            self.linear.bias.fill_(-1)
        self.linear.bias.requires_grad_(False)

    def forward(self, x):
        if self.hidden:
            x = nn.functional.leaky_relu(self.hidden(x), True)
        x = self.batch_norm(x)
        return self.linear(x)
    
    @staticmethod
    def load(model_type : str, path : str, device=torch.device("cpu"), dtype=torch.float32):
        # Parse model architecture
        architecture, head_name, _ = get_model(model_type)

        # Read weight file
        weights = torch.load(path, device, weights_only=True)
        num_classes, num_embeddings = weights[f"{head_name}.linear.weight"].shape
        
        # Load weights into model architecture
        setattr(architecture, head_name, Classifier(num_embeddings, num_classes))
        architecture.load_state_dict(weights)
        architecture.to(device, dtype)
        
        return architecture

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
    for i, p in enumerate(TQDM(paths, desc=desc)):
        try:
            tensor[i] = resize(converter(decode_image(p, ImageReadMode.RGB)), shape).to(device)
        except Exception as e:
            e.add_note(f'Path: {p}')
            raise e
    return tensor

def get_training_augmentation():
    """
    Returns a training augmentation pipeline for normalized tensors.
    Assumes the input tensor is already normalized (e.g., in the range [0, 1] or standardized).

    Returns:
        transforms.Compose: A composition of augmentations.
    """
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=15),
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # # Convert back to tensor (in case some augmentations convert to PIL Image)
        # torchvision.transforms.ToTensor(),
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

def main(
    input: str,
    output: str = ".",
    model: str = "efficientnet_v2_s",
    checkpoint: Optional[List[str]] = None,
    weights: Optional[str] = None,
    data_index: Optional[str] = None,
    class_index: str = "class_index.json",
    fine_tune: bool = False,
    epochs: int = 15,
    batch_size: int = 16,
    warmup_epochs: float = 2.0,
    name: Optional[str] = None,
    device: str = "cuda:0",
    dtype: str = "float16",
    seed: Optional[int] = None
) -> None:
    """
    Train a simple classifier.

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
            found under `input`. Default is 'class_index.json'.

        fine_tune (bool, optional):
            If True, update only the classifier weights during training.
            Default is False.

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

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    classes, class2idx, idx2class, num_classes = parse_class_index(class_index, input_dir)

    # Prepare model
    model, head_name, model_preprocess = get_model(model)
    model : torch.nn.Module
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Unknown model type `{type(model)}`, expected `{torch.nn.Module}`")
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

    # Prepare datasets/dataloaders
    if data_index is None:
        with NamedTemporaryFile() as tmpfile:
            write_metadata(input_dir, classes, tmpfile.name, train_proportion=0.9)
            train_image_data, val_image_data = get_image_data(tmpfile.name)
    else:
        train_image_data, val_image_data = get_image_data(data_index)

    train_dataset, val_dataset, train_loader, val_loader = get_dataset_dataloader(
        train_image_data, 
        val_image_data, 
        class2idx, 
        model_preprocess.resize_size if hasattr(model_preprocess, "resize_size") else 256, 
        batch_size,
        device, 
        dtype
    )
    augmentation = get_training_augmentation()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    example_image = train_dataset[random.choice(range(len(train_dataset)))][0].clone().float().cpu()

    axs[0].imshow(example_image.permute(1,2,0))
    axs[1].imshow(augmentation(example_image).permute(1,2,0))

    plt.savefig("example_augmentation.png")
    plt.close()

    # Define training "hyperparameters"
    epochs = epochs
    start_epoch = 0
    lr_warmup_epochs = warmup_epochs
    lr = 0.0001
    label_smoothing = 0.1
    if lr_warmup_epochs > epochs:
        raise ValueError(f'Number of warmup epochs ({lr_warmup_epochs}) must be lower than number of total epochs ({epochs}).')

    parameters = set_weight_decay(model, 1e-3)
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(epochs - lr_warmup_epochs) * len(train_loader), 
        eta_min=lr * 10**-6
    )
    if lr_warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=10**-2, 
            total_iters=lr_warmup_epochs * len(train_loader)
        )
    
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
        milestones=[lr_warmup_epochs * len(train_loader)]
    ) if lr_warmup_epochs > 0 else main_lr_scheduler

    if checkpoint is not None:
        checkpoint_files = checkpoint
        if isinstance(checkpoint_files, list) and len(checkpoint_files) == 1:
            checkpoint_files = checkpoint_files[0]
        if isinstance(checkpoint_files, str):
            checkpoint = torch.load(checkpoint_files, device)
        else:
            checkpoint = average_checkpoints(checkpoint_files)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"]

    # Run training
    train(
        model, 
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
        output_dir=output_dir
    )

    # Save result model
    model.eval()
    save_on_master(model.state_dict(), os.path.join(output_dir, f"{name}.pt"))

if __name__ == "__main__":  
    parser = ArgumentParser(
        prog = "train",
        description = "Train a simple classifier"
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
        "-C", "--class_index", type=str, default="class_index.json", required=False,
        help="path to a JSON file containing the class name to index mapping. If it doesn't exist, one will be created based on the directories found under `input`."
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
    main(**parser.parse_args())

