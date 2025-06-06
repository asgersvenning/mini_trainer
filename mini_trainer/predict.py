import json
import os
import re
import warnings
from argparse import ArgumentParser
from math import ceil
from typing import Any, Optional, Type

import torch
from torch.utils.data import DataLoader

from mini_trainer import TQDM, Formatter
from mini_trainer.builders import BaseBuilder, AutoEmbedder
from mini_trainer.utils.data import find_images
from mini_trainer.utils.io import ImageLoader
from mini_trainer.utils.logging import BaseResultCollector


def main(
    input : Optional[str],
    output : Optional[str]=None,
    name : Optional[str]=None,
    class_index : str="class_index.json",
    data_index : Optional[str]=None,
    split : Optional[str]="test",
    embeddings : bool=False,
    batch_size : int=32,
    num_workers : Optional[int]=None,
    n_max : Optional[int]=None,
    device : str="cuda:0",
    dtype : str="bfloat16",
    verbose : bool=False,
    builder : Type[BaseBuilder]=BaseBuilder,
    spec_model_dataloader_kwargs : dict[str, Any]={},
    model_builder_kwargs : dict[str, Any]={},
    result_collector=BaseResultCollector,
    result_collector_kwargs : dict[str, Any]={"training_format" : False, "verbose" : False}
) -> None:
    """
    Predict with a classifier.

    Args:
        input (str):
            Path to a directory containing images for prediction, optionally structured
            with subdirectories named after class labels. (required unless `data_index` is passsed)

        output (str, optional):
            Path to the directory where the results should be stored, 
            if passed and the directory does not already exist it is created.
        
        name (str, optional):
            Name of the prediction run, used as a prefix for the result files.
            The name should only contain alphanumeric ASCII characters and underscores.
            If not supplied the basename of the input directory is used (all non-ASCII alphanumeric/underscore characters are removed).

        class_index (str):
            Path to a JSON file containing the mapping from class names to indices.
            If it does not exist, one will be created based on subdirectories under `input`. (required)

        data_index (str, optional):
            Path to a JSON file containing three arrays with keys 'path', 'split',
            and 'class', representing a structured dataset.
            Default is None.

        split (str, optional):
            Name of the split to predict on (default="test"). Only applies when `data_index` is passed.

        embeddings (bool):
            Compute and store the embeddings (the embedding layer is automatically guessed from the architecture). 
            Default is False.

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
            PyTorch data type used for inference (e.g., 'bfloat16').
            Default is 'bfloat16'.
        
        verbose (bool, optional):
            Print additional logging messages to the terminal.
        
        **kwargs: Additional arguments to be documented. 
            All additional arguments are not available from the commandline, but exist to enable usage of 
            the `train.py` and `predict.py` functionality with custom models, loss functions, data loaders etc.

    Returns:
        None
    """
    # Prepare state
    device : torch.device = torch.device(device)
    dtype : torch.dtype = getattr(torch, dtype)

    if name is None:
        name = re.sub("[^a-zA-Z0-9_]", "", "pred" if input is None else os.path.basename(input))
    else:
        if re.search("[^a-zA-Z0-9_]", name) and verbose:
            warnings.warn(f'Found non-standard characters (non-ASCII alphanumeric and underscore) in the supplied name; "{name}".')

    if input is not None:
        input_dir = os.path.abspath(input)
        if not os.path.isdir(input_dir):
            raise OSError(f'Supplied input directory ("{input_dir}") is not a valid directory.')
    else:
        input_dir = None
    output_dir = os.path.abspath(output) if isinstance(output, str) else None
    if isinstance(output_dir, str) and not os.path.isdir(output_dir):
        raise OSError(f'Supplied output directory ("{output_dir}") is not a valid directory.')

    workers = num_workers
    if workers is None:
        workers = os.cpu_count() - 1
        workers -= workers % 2
    batch_size = batch_size

    # Load additional information for model and dataloader instantiation
    # e.g. number of classes, class-to-index dictionary
    extra_model_kwargs, extra_dataloader_kwargs = builder.spec_model_dataloader(
        path=class_index, 
        dir=None,
        **spec_model_dataloader_kwargs
    )

    # Prepare model
    nn_model, model_preprocess = builder.build_model( 
        device=device, 
        dtype=dtype, 
        **{**extra_model_kwargs, **model_builder_kwargs}
    )
    nn_model.eval()
    if embeddings:
        nn_model = AutoEmbedder(nn_model)
        
    # Prepare image loader
    image_loader = ImageLoader(
        model_preprocess,
        dtype,
        torch.device("cpu")
    )

    if data_index:
        assert split is not None, ValueError(f'Prediction `split` must be passed when `data_index` is used.')
        split = split.strip().lower()
        with open(data_index, "r") as f:
            data_index_data = json.load(f)
            images = [im for im, spl in zip(data_index_data["path"], data_index_data["split"]) if spl.strip().lower() == split]
    else:
        assert input_dir is not None, ValueError(f'Input directory ({input_dir}) must be a valid path.')
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
                if embeddings:
                    prediction, embedding = nn_model(batch.to(device))
                    kwargs = {"embeddings" : embedding}
                else:
                    prediction = nn_model(batch.to(device))
                    kwargs = {}
            results.collect(
                paths = images[i:(i+len(batch))],
                predictions = prediction,
                **kwargs
            )
    
    end.record()
    torch.cuda.synchronize(device)
    inf_time = start.elapsed_time(end) / 1000
    if verbose:
        print(f'Inference took {inf_time:.1f}s ({len(images)/inf_time:.1f} img/s)')

    # Write results
    if output_dir is not None:
        with open(os.path.join(output_dir, f'{name}_result.json'), "w") as f:
            json.dump(results.data, f)

    results.evaluate(outdir=output_dir, prefix=f'{name}_')
    if verbose:
        print(f'Outputs written to {os.path.abspath(output_dir)}')

def cli(description="Predict with a classifier", **kwargs):
    parser = ArgumentParser(
        prog="predict",
        description=description,
        formatter_class=Formatter
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", required=False,
        help="Print the prediction results to the terminal? Disabled by default."
    )

    if kwargs:
        for argname, args in kwargs.items():
            parser.add_argument(f'--{argname}', **args)

    input_args = parser.add_argument_group("Input [mandatory]")
    input_args.add_argument(
        "-m", "--model", type=str, default="efficientnet_v2_s", required=True,
        help=
        "Name of the model type from the torchvision model zoo (not case-sensitive):\n"
        "https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights"
    )
    input_args.add_argument(
        "-C", "--class_index", type=str, default="class_index.json", required=True,
        help="Path to a JSON file containing the class name to index mapping."
    )
    input_args.add_argument(
        "-w", "--weights", type=str, required=True,
        help="Model weights for inference."
    )
    input_args.add_argument(
        "-i", "--input", type=str, required=False,
        help=
        "Path to a directory containing a subdirectory for each class,\n"
        "where the name of each subdirectory should correspond to the name of the class."
    )
    input_args.add_argument(
        "-D", "--data_index", type=str, required=False,
        help=
        "JSON file containing three arrays with keys 'path', 'split' and 'class'.\n"
        "The arrays should all have equal lengths and can be considered \"columns\" in a table.\n"
        "The 'split' column should contain values 'train', 'validation' or other,\n"
        "and the 'class' column' should contain the the class *names* (not indices) for each file/path."
    )
    input_args.add_argument(
        "--split", type=str, default="test", required=False,
        help="Which split to perform inference on (default='test'). Only applies if `--data_index` is passed."
    )
    out_args = parser.add_argument_group("Output [optional]")
    out_args.add_argument(
        "-o", "--output", type=str, required=False,
        help='Path to the directory where the results should be stored.'
    )
    out_args.add_argument(
        "-n", "--name", type=str, required=False,
        help=
        "Name of the prediction run, used as a prefix for the result files.\n"
        "The name should only contain alphanumeric ASCII characters and underscores.\n"
        "If not supplied the basename of the input directory is used (removing non-ASCII alphanumeric/underscore characters)."
    )
    out_args.add_argument(
        "--training_format", action="store_true", required=False,
        help= \
        "Are the images in `input` stored in subfolders named by their class? "
        "If so, we can calculate accuracy statistics."
    )
    out_args.add_argument(
        "-e", "--embeddings", action="store_true", required=False,
        help="Compute and store embeddings as well."
    )
    cfg_args = parser.add_argument_group("Config [optional]")
    cfg_args.add_argument(
        "--batch_size", type=int, default=32, required=False,
        help="Batch size used for inference (default=32). Higher requires more VRAM."
    )
    cfg_args.add_argument(
        "--num_workers", type=int, default=None, required=False,
        help= \
        "Number of workers used for reading/loading images for inference. "
        "Default is set to number of physical CPU cores." 
    )
    cfg_args.add_argument(
        "--device", type=str, default="cuda:0", required=False,
        help='Device used for inference (default="cuda:0").'
    )
    cfg_args.add_argument(
        "--dtype", type=str, default="bfloat16", required=False,
        help="PyTorch data type used for inference (default=bfloat16)."
    )
    args = vars(parser.parse_args())
    args["model_builder_kwargs"] = {
        "model_type" : args.pop("model"),
        "weights" : args.pop("weights")
    }
    args["result_collector_kwargs"] = {
        "training_format" : args.pop("training_format", False), 
        "verbose" : args.get("verbose", False)
    }
    if args["embeddings"]:
        args["result_collector_kwargs"]["additional_attributes"] = ["embeddings"]
    return args

if __name__ == "__main__":  
    main(**cli())
