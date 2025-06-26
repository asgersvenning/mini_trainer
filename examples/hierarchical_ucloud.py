import json
import os
import warnings

import pandas as pd
from tqdm.contrib.concurrent import thread_map

from mini_trainer import TQDM
from mini_trainer.hierarchical.integration import HierarchicalBuilder
from mini_trainer.hierarchical.train import cli as mth_train_args
from mini_trainer.train import main as mt_train
from collections import defaultdict


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
            split = "validation"
            if train == 0 or (train / max(1, train + val)) < train_prop:
                split = "train"
            # Ensure we have at least one fold in train/val (if there is more than 1 fold)
            if train > 0 and val == 0:
                split = "validation"
            if split == "train":
                train += 1
            if split == "validation":
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
    data["di_split"] = [fld2spl[fld] for fld in flds]
    
    # Construct image paths by joining the data directory with the species and file name.
    data["di_path"] = [os.path.join(dir, sp, fn) for sp, fn in zip(data["speciesKey"].tolist(), data["filename"].tolist())]

    # Construct index-based class labels
    with open(class_index, "r") as f:
        cls2idx = json.load(f)["cls2idx"]
    data["di_cls"] = [[cls2idx[str(lvl)].get("c", None) for lvl, c in enumerate(sgf)] for sgf in zip(*[data[f'{tl}Key'] for tl in ["species", "genus", "family"]])]
    
    data = data[[cls[0] is not None for cls in data["di_cls"]]]
    data = data[thread_map(os.path.exists, data["di_path"], tqdm_class=TQDM, desc="Checking parquet paths...")]

    return {
        "split" : list(data["di_split"]),
        "class" : list(data["di_cls"]),
        "path"  : list(data["di_path"])
    }

def parquet_to_combinations(path : str):
    data = pd.read_parquet(path)
    combinations = defaultdict(lambda : 0)
    for sgf in zip(*[data[f'{tl}Key'] for tl in ["species", "genus", "family"]]):
        combinations[tuple(sgf)] += 1
    return [list(sgf) for sgf, count in sorted(combinations.items(), key=lambda x : x[1]) if count > 25]

def cli():
    kwargs = mth_train_args(
        parquet={
            None : "-P",
            "type" : str, 
            "required" : True,
            "help" : "Path to the parquet metadata file produced by `gbifxdl`."
        }
    )

    # Create class and data index from parquet
    class_index_path = os.path.join(kwargs["output"], "class_index.json")
    HierarchicalBuilder.spec_model_dataloader(
        class_index_path,
        kwargs["parquet"],
        parquet_to_combinations
    )
    kwargs["class_index"] = class_index_path
    data_index_path = os.path.join(kwargs["output"], "data_index.json")
    with open(data_index_path, "w") as f:
        json.dump(
            obj=parquet_to_dataindex(
                path=kwargs["parquet"],
                dir=kwargs["input"],
                class_index=class_index_path
            ),
            fp=f
        )
    kwargs["dataloader_builder_kwargs"]["data_index"] = data_index_path
    kwargs.pop("parquet")
    
    # Call the Python training API
    mt_train(
        **kwargs,
        builder=HierarchicalBuilder,
    )

if __name__ == "__main__":
    cli()