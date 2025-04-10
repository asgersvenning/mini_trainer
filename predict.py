import json
import os
from argparse import ArgumentParser
from glob import glob
from math import ceil
from typing import (Any, Callable, Concatenate, Dict, Iterable, Optional,
                    Tuple, Union)

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize
from tqdm import tqdm as TQDM

from train import base_load_model, convert2bf16, convert2fp16, convert2fp32
from utils import confusion_matrix, is_image, parse_class_index


class ImageDataset(Dataset):
    def __init__(self, func : Callable[[str], torch.Tensor], items : list[str]):
        self.func = func
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.func(self.items[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ImageLoader:
    def __init__(self, preprocessor, dtype, device):
        self.dtype, self.device = dtype, device
        self.preprocessor = preprocessor
        match self.dtype:
            case torch.float16:
                self.converter = convert2fp16
            case torch.float32:
                self.converter = convert2fp32
            case torch.bfloat16:
                self.converter = convert2bf16
            case _:
                raise ValueError("Only fp16 supported for now.")
        size = self.preprocessor.resize_size if hasattr(self.preprocessor, "resize_size") else 256
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str):
            proc_img : torch.Tensor = self.preprocessor(resize(convert2fp32(decode_image(x, ImageReadMode.RGB)), self.shape)).to(self.device, self.dtype)
            return proc_img
        return ImageDataset(self, x)

def find_images(root : str):
    paths = glob(os.path.join(root, "**"), recursive=True)
    return list(filter(is_image, paths))

def main(
    model : str,
    weights : str,
    input : str,
    output : str="result.json",
    class_index : str="class_index.json",
    training_format : bool=False,
    batch_size : int=32,
    num_workers : Optional[int]=None,
    device : str="cuda:0",
    dtype : str="float16",
    load_model : Callable[
        Concatenate[str, int, Optional[str], bool, torch.dtype, torch.device, ...], 
        Tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    ]=base_load_model,
    load_model_kwargs : Dict[str, Any]={}
) -> None:
    """
    Predict with a classifier.

    Args:
        model (str):
            Name of the model type from the torchvision model zoo.
            See: https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights.
            Not case-sensitive. (required)

        weights (str):
            Path to the model weights file used for inference. (required)

        input (str):
            Path to a directory containing images for prediction, optionally structured
            with subdirectories named after class labels. (required)

        output (str, optional):
            Path where inference results will be stored.
            Default is 'result.json'.

        class_index (str):
            Path to a JSON file containing the mapping from class names to indices.
            If it does not exist, one will be created based on subdirectories under `input`. (required)

        training_format (bool, optional):
            Indicates if the images in `input` are organized into class-specific subdirectories.
            If True, accuracy statistics will be computed. Default is False.

        batch_size (int, optional):
            Batch size used during inference. Larger values require more VRAM.
            Default is 32.

        num_workers (Optional[int], optional):
            Number of worker threads/processes used for loading images.
            Defaults to the number of physical CPU cores if None.

        device (str, optional):
            Device to perform inference on (e.g., 'cuda:0', 'cpu').
            Default is 'cuda:0'.

        dtype (str, optional):
            PyTorch data type used for inference (e.g., 'float16').
            Default is 'float16'.
        
        **kwargs: Additional arguments to be documented. 
            All additional arguments are not available from the commandline, but exist to enable usage of 
            the `train.py` and `predict.py` functionality with custom models, loss functions, data loaders etc.

    Returns:
        None
    """
    # Prepare state
    device : torch.device = torch.device(device)
    dtype : torch.dtype = getattr(torch, dtype)

    input_dir = os.path.abspath(input)

    workers = num_workers
    if workers is None:
        workers = os.cpu_count() - 1
        workers -= workers % 2
    batch_size = batch_size

    classes, class2idx, idx2class, num_classes = parse_class_index(class_index)

    # Prepare model
    nn_model, model_preprocess = load_model(model, num_classes, weights, False, dtype, device, **load_model_kwargs)
        
    # Prepare image loader
    image_loader = ImageLoader(
        model_preprocess,
        dtype,
        torch.device("cpu")
    )
    images = find_images(input_dir)
    ds = image_loader(images)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        pin_memory_device=device.type,
        shuffle=False,
        drop_last=False
    )

    # Inference
    results = {
        "path" : [],
        "pred" : [],
        "conf" : []
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for batch_i, batch in TQDM(enumerate(dl), desc="Running inference...", total=ceil(len(images) / batch_size), leave=True):
            i = batch_i * batch_size
            prediction = nn_model(batch.to(device))
            results["path"].extend(images[i:(i+len(batch))])
            results["pred"].extend([idx2class[idx] for idx in prediction.argmax(1).tolist()])
            results["conf"].extend(prediction.softmax(1).max(0).values.tolist())
    
    # Write results
    with open(output, "w") as f:
        json.dump(results, f)

    print(f'Outputs written to {os.path.abspath(output)}')
    end.record()
    torch.cuda.synchronize(device)
    inf_time = start.elapsed_time(end) / 1000
    print(f'Inference took {inf_time:.1f}s ({len(images)/inf_time:.1f} img/s)')

    if training_format:
        results["gt"] = [f.split(os.sep)[-2] for f in results["path"]]

        confusion_matrix(results, idx2class)

if __name__ == "__main__":  
    parser = ArgumentParser(
        prog = "predict",
        description = "Predict with a classifier"
    )  
    parser.add_argument(
        "-m", "--model", type=str, default="efficientnet_v2_s", required=True,
        help="name of the model type from the torchvision model zoo (https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights). Not case-sensitive."
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True,
        help="Model weights for inference."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="path to a directory containing a subdirectory for each class, where the name of each subdirectory should correspond to the name of the class."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="result.json", required=False,
        help='Path used to store inference results (default="result.json").'
    )
    parser.add_argument(
        "-C", "--class_index", type=str, default="class_index.json", required=True,
        help="path to a JSON file containing the class name to index mapping. If it doesn't exist, one will be created based on the directories found under `input`."
    )
    parser.add_argument(
        "--training_format", action="store_true", required=False,
        help="Are the images in `input` stored in subfolders named by their class? If so, we can calculate accuracy statistics."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, required=False,
        help="Batch size used for inference (default=32). Higher requires more VRAM."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, required=False,
        help="Number of workers used for reading/loading images for inference. Default is set to number of physical CPU cores." 
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", required=False,
        help='Device used for inference (default="cuda:0").'
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", required=False,
        help="PyTorch data type used for inference (default=float16)."
    )
    main(**parser.parse_args())
