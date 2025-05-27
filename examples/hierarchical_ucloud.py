import json
import os
import warnings
from argparse import ArgumentParser

import pandas as pd
from hierarchical.base.integration import HierarchicalBuilder

from mini_trainer import Formatter
from mini_trainer.train import main as mt_train
from mini_trainer.utils import increment_name_dir


def parquet_to_dataindex(
        path : str, 
        dir : str, 
        class_index : str,
        test_split : tuple[int, ...]=(0,),
        train_prop : float=0.9
    ):
    data = pd.read_parquet(path)

    flds, fld2spl = data["set"].tolist(), dict()
    # Allocate folds to splits
    train = val = total = test = 0
    for fld in set(flds):
        split = "test"
        if isinstance(fld, str):
            clean_fld = fld.strip()
        else:
            clean_fld = fld
        if (isinstance(clean_fld, int) or clean_fld.isdigit()) and not (int(clean_fld) in test_split):
            split = "val"
            if train == 0 or (train / max(1, train + val)) < train_prop:
                split = "train"
            # Ensure we have at least one fold in train/val (if there is more than 1 fold)
            if train > 0 and val == 0:
                split = "val"
            if split == "train":
                train += 1
            if split == "val":
                val += 1
        else:
            test += 1
        total += 1
        fld2spl[fld] = split
    if (train + val + test) != total:
        raise RuntimeError(f'Inconsistent fold-to-split allocation: {train + val + test=} != {total=}')
    if train == 0:
        warnings.warn("No folds allocated to training! May cause issues with training.")
    if val == 0:
        warnings.warn("No folds allocated to validation! May cause issues with training.")
    if test == 0:
        warnings.warn("No folds allocated to testing! Likely has no effect on training, but is likely incorrect.")
    # Translate folds to splits
    spl = [fld2spl[fld] for fld in flds]
    
    # Construct image paths by joining the data directory with the species and file name.
    paths = [os.path.join(dir, sp, fn) for sp, fn in zip(data["speciesKey"].tolist(), data["filename"].tolist())]

    # Construct index-based class labels
    with open(class_index, "r") as f:
        cls2idx = json.load(f)["cls2idx"]
    cls = [[cls2idx[str(lvl)][c] for lvl, c in enumerate(sgf)] for sgf in zip(*[data[f'{tl}Key'] for tl in ["species", "genus", "family"]])]
    
    return {
        "split" : spl,
        "class" : cls,
        "path"  : paths
    }

def parquet_to_combinations(path : str):
    data = pd.read_parquet(path)
    combinations = set()
    for sgf in zip(*[data[f'{tl}Key'] for tl in ["species", "genus", "family"]]):
        combinations.add(tuple(sgf))
    return [list(sgf) for sgf in sorted(combinations)]

def tensorboard_logger_kwargs(name : str, output : str, resume : bool=False):
    from torch.utils.tensorboard.writer import SummaryWriter

    from mini_trainer.utils import increment_name_dir
    from mini_trainer.utils.logging import MetricLoggerWrapper
    from mini_trainer.utils.tensorboard import TensorboardLogger
    
    tensorboard_dir = os.path.join(output, "tensorboard")
    if resume:
        run_name = name
    else:
        run_name = increment_name_dir(name, tensorboard_dir)
    tensorboard_writer = SummaryWriter(os.path.join(tensorboard_dir, run_name), flush_secs=30)
    
    return {
        "verbose" : True,
        "logger_cls" : [MetricLoggerWrapper, TensorboardLogger],
        "logger_cls_extra_kwargs" : [{}, {"writer" : tensorboard_writer}]
    }

def cli():
    parser = ArgumentParser(
        prog="hierarchical_train",
        description="Train a hierarchical model.",
        formatter_class=Formatter
    )
    input_args = parser.add_argument_group("Input [mandatory]")
    input_args.add_argument(
        "-i", "--input", type=str, required=True,
        help=
        "Path to a directory containing a subdirectory for each class,\n" 
        "where the name of each subdirectory should correspond to the name of the class."
    )
    input_args.add_argument(
        "-P", "--parquet", type=str, required=True,
        help="Path to the parquet metadata file produced by `gbifxdl`."
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
    train_args = parser.add_argument_group("Training [optional]")
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
        "--fine-tune", action="store_true", required=False,
        help="OBS: This should probably not be used. Update only the classifier weights."
    )
    cfg_args = parser.add_argument_group("Config [optional]")
    cfg_args.add_argument(
        "--device", type=str, default="cuda:0"
    )
    cfg_args.add_argument(
        "--dtype", type=str, default="float16"
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
        "batch_size" : args.pop("batch_size"),
        "resize_size" : 256, 
        "train_proportion" : 0.9
    }
    args["optimizer_builder_kwargs"] = {
        "lr" : args.pop("lr"),
        "weight_decay" : 1e-4
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
        args["logger_builder_kwargs"] = tensorboard_logger_kwargs(args["name"], output=args["output"], resume=bool(args["checkpoint"]))

    # Create class and data index from parquet
    class_index_path = os.path.join(args["output"], "class_index.json")
    HierarchicalBuilder.spec_model_dataloader(
        class_index_path,
        args["parquet"],
        parquet_to_combinations
    )
    args["class_index"] = class_index_path
    data_index_path = os.path.join(args["output"], "data_index.json")
    with open(data_index_path, "w") as f:
        json.dump(
            obj=parquet_to_dataindex(
                path=args["parquet"],
                dir=args["input"],
                class_index=class_index_path
            ),
            fp=f
        )
    args["dataloader_builder_kwargs"]["data_index"] = data_index_path

    args.pop("parquet")
    
    # Call the Python training API
    mt_train(
        builder=HierarchicalBuilder,
        **args
    )

if __name__ == "__main__":
    cli()