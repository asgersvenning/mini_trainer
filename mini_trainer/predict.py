import json
import os
from argparse import ArgumentParser
from math import ceil
from typing import Any, Dict, Optional, Type

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as TQDM

from .builders import BaseBuilder
from .utils import BaseResultCollector, ImageLoader, find_images


def main(
    input : str,
    output : str="result.json",
    class_index : str="class_index.json",
    batch_size : int=32,
    num_workers : Optional[int]=None,
    n_max : Optional[int]=None,
    device : str="cuda:0",
    dtype : str="float16",
    builder : Type[BaseBuilder]=BaseBuilder,
    spec_model_dataloader_kwargs : Dict[str, Any]={},
    model_builder_kwargs : Dict[str, Any]={},
    result_collector=BaseResultCollector,
    result_collector_kwargs : Dict[str, Any]={"training_format" : False}
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

        batch_size (int, optional):
            Batch size used during inference. Larger values require more VRAM.
            Default is 32.

        num_workers (Optional[int], optional):
            Number of worker threads/processes used for loading images.
            Defaults to the number of physical CPU cores if None.

        n_max (Optional[int], optional):
            Maximum number of images to run inference on.

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

    # Load additional information for model and dataloader instantiation
    # e.g. number of classes, class-to-index dictionary
    extra_model_kwargs, extra_dataloader_kwargs = builder.spec_model_dataloader(
        path=class_index, 
        dir=input_dir,
        **spec_model_dataloader_kwargs
    )

    # Prepare model
    nn_model, model_preprocess = builder.build_model( 
        device=device, 
        dtype=dtype, 
        **{**extra_model_kwargs, **model_builder_kwargs}
    )
    nn_model.eval()
        
    # Prepare image loader
    image_loader = ImageLoader(
        model_preprocess,
        dtype,
        torch.device("cpu")
    )
    images = find_images(input_dir)
    if n_max is not None and len(images) > n_max:
        images = images[:n_max]
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
    results = result_collector(**{**extra_dataloader_kwargs, **result_collector_kwargs})
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for batch_i, batch in TQDM(enumerate(dl), desc="Running inference...", total=ceil(len(images) / batch_size), leave=True):
            i = batch_i * batch_size
            if len(batch.shape) == 3:
                batch = batch.unsqueeze(0)
            with torch.autocast(device_type=device.type, dtype=dtype):
                prediction = nn_model(batch.to(device))
            results.collect(
                paths = images[i:(i+len(batch))],
                predictions = prediction
            )
    
    # Write results
    with open(output, "w") as f:
        json.dump(results.data, f)

    print(f'Outputs written to {os.path.abspath(output)}')
    end.record()
    torch.cuda.synchronize(device)
    inf_time = start.elapsed_time(end) / 1000
    print(f'Inference took {inf_time:.1f}s ({len(images)/inf_time:.1f} img/s)')

    results.evaluate()

def cli():
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
    args = vars(parser.parse_args())
    args["model_builder_kwargs"] = {
        "model_type" : args.pop("model"),
        "weights" : args.pop("weights")
    }
    args["result_collector_kwargs"] = {"training_format" : args.pop("training_format")}
    main(**args)

if __name__ == "__main__":  
    cli()
