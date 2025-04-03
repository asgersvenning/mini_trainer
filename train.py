import json
import os
import random
from argparse import ArgumentParser
from functools import partial
from random import choice, random, seed
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
from utils import set_weight_decay

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

def write_metadata(directory : str, dst : str, train_proportion : float=0.9):
    data = {
        "path" : [],
        "class" : [],
        "split" : []
    }
    for cls in CLASSES:
        this_dir = os.path.join(directory, cls)
        for file in map(os.path.basename, os.listdir(this_dir)):
            data["path"].append(os.path.join(this_dir, file))
            data["class"].append(cls)
            data["split"].append("train" if random() < train_proportion else "validation")
    with open(dst, "w") as f:
        json.dump(data, f)

def is_image(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(16)  # read enough bytes for JPEG and PNG signatures
    except Exception:
        return False

    # JPEG files start with: 0xFF, 0xD8
    if header.startswith(b'\xff\xd8'):
        return True

    # PNG files start with: 0x89, 'PNG', CR, LF, 0x1A, LF
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return True

    return False

def get_image_data(path : str):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Meta data file ({path}) for training split not found. Please provide a JSON with the following keys: "path", "class", "split".')
        with open(path, "rb") as f:
            _image_data = {k : np.array(v) for k, v in json.load(f).items()}
        image_data = {k : v[np.array([is_image(f) for f in _image_data["path"]])] for k, v in _image_data.items()}
        train_image_data = {k : v[image_data["split"] == np.array("train")] for k, v in image_data.items()}
        test_image_data = {k : v[image_data["split"] == np.array("validation")] for k, v in image_data.items()}
        return train_image_data, test_image_data

def get_dataset_dataloader(train_image_data : Dict, val_image_data : Dict, class2idx : Dict[str, int], resize_size : Union[int, Tuple[int, int]], device=torch.device("cpu"), dtype=torch.float32):
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
        batch_size=ARGS.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=ARGS.batch_size, 
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

if __name__ == "__main__":  
    parser = ArgumentParser(
        prog = "train",
        description = "Train a simple classifier"
    )  
    parser.add_argument(
        "--model", type=str, default="efficientnet_v2_s", required=False,
        help="name of the model type from the torchvision model zoo (https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights). Not case-sensitive."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="path to a directory containing a subdirectory for each class, where the name of each subdirectory should correspond to the name of the class."
    )
    parser.add_argument(
        "--class_index", type=str, default="class_index.json", required=False,
        help="path to a JSON file containing the class name to index mapping. If it doesn't exist, one will be created based on the directories found under `input`."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False,
        help="Model checkpoint (weights) used to initialize model before training."
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
        "--warmup_epochs", type=int, default=2, required=False,
        help="Number of warmup epochs (default=2)."
    )
    parser.add_argument(
        "--name", type=str, required=False,
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
    ARGS = parser.parse_args()

    # Prepare state
    if ARGS.name is None:
        NAME = f'{ARGS.model}_{"fine_tune" if ARGS.fine_tune else "full"}_e{ARGS.epochs}'
    else:
        NAME = ARGS.name
    
    if ARGS.seed is not None:
        seed(ARGS.seed)

    device = torch.device(ARGS.device)
    dtype = getattr(torch, ARGS.dtype)

    if not os.path.exists(ARGS.class_index):
        _class2idx = {cls : i for i, cls in enumerate(sorted([f for f in map(os.path.basename, os.listdir(ARGS.input)) if os.path.isdir(os.path.join(ARGS.input, f))]))}
        with open(ARGS.class_index, "w") as f:
            json.dump(_class2idx, f)
    with open(ARGS.class_index, "rb") as f:
        class2idx = json.load(f)
    CLASSES = list(class2idx.keys())
    idx2class = {v : k for k, v in class2idx.items()}
    num_classes = len(idx2class)

    # Prepare model
    model, head_name, model_preprocess = get_model(ARGS.model)
    num_embeddings = getattr(model, head_name)[1].in_features
    if ARGS.checkpoint is not None:
        model = Classifier.load(ARGS.model, ARGS.checkpoint, device=device, dtype=torch.float32)
    else:
        setattr(model, head_name, Classifier(num_embeddings, num_classes))
        model.to(device, torch.float32)
    if ARGS.fine_tune:
        for name, param in model.named_parameters():
            if param.requires_grad and not head_name in name:
                param.requires_grad_(False)

    # Prepare datasets/dataloaders
    with NamedTemporaryFile() as tmpfile:
        write_metadata(ARGS.input, tmpfile.name, train_proportion=0.9)
        train_image_data, val_image_data = get_image_data(tmpfile.name)

    train_dataset, val_dataset, train_loader, val_loader = get_dataset_dataloader(
        train_image_data, 
        val_image_data, 
        class2idx, 
        model_preprocess.resize_size if hasattr(model_preprocess, "resize_size") else 256, 
        device, 
        dtype
    )
    augmentation = get_training_augmentation()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    example_image = train_dataset[choice(range(len(train_dataset)))][0].clone().float().cpu()

    axs[0].imshow(example_image.permute(1,2,0))
    axs[1].imshow(augmentation(example_image).permute(1,2,0))

    plt.savefig("example_augmentation.png")
    plt.close()

    # Define training "hyperparameters"
    epochs = ARGS.epochs
    lr_warmup_epochs = ARGS.warmup_epochs

    parameters = set_weight_decay(model, 1e-3)
    optimizer = torch.optim.AdamW(parameters, lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs - lr_warmup_epochs, 
        eta_min=0
    )
    if lr_warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=lr_warmup_epochs
        )
    
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
        milestones=[lr_warmup_epochs]
    ) if lr_warmup_epochs > 0 else main_lr_scheduler

    # Run training
    train(
        model, 
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler,
        epochs,
        0,
        model_preprocess,
        augmentation,
        device=device,
        dtype=dtype
    )

    # Save result model
    model.eval()
    torch.save(model.state_dict(), f"{NAME}.pt")

