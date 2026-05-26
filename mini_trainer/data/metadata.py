import json
import os
import random
from collections import OrderedDict
from glob import glob
from typing import cast

import numpy as np

from mini_trainer.data import is_image
from mini_trainer.integrations import (
    create_taxonomy,
    get_metadata_from_parquet,
    is_taxonomical_cls2idx,
    labels_from_taxonomy,
    parquet_to_class_spec,
)


def find_images(root: str):
    """Find all images within root."""
    paths = glob(os.path.join(root, "**"), recursive=True)
    check = is_image(paths)
    return [p for p, f in zip(paths, check) if f]


def auto_find_images(src: str, **kwargs) -> tuple[list[int] | list[list[int]], list[str]]:
    """Find images in source and possibly create training metadata."""
    metadata = labels = images = None
    if os.path.isfile(src):
        if src.endswith(".parquet"):
            metadata = get_metadata_from_parquet(src, **kwargs)
        else:
            images = [src]
    elif os.path.isdir(src):
        contains_only_dirs = all([os.path.isdir(os.path.join(src, p)) for p in os.listdir(src)])
        if contains_only_dirs:
            metadata = create_metadata(src, **{**kwargs, **{"train_proportion": 0, "val_proportion": 0, "labels": None}})
        else:
            images = find_images(src)
    else:
        raise ValueError(f"Image source must be a file (image or gbifxdl parquet) or directory with images, not {src}.")
    if metadata is not None:
        images = [p for p, s in zip(metadata["path"], metadata["split"]) if s == "test"]
        labels = [c for c, s in zip(metadata["label"], metadata["split"]) if s == "test"]
    assert images is not None
    if labels is None:
        labels = []
    return labels, images


# TODO: Unfortunately, this function has some functionality for the hierarchical submodule
# even though the core mini_trainer module and the hierarchical submodule are
# supposed to be entirely compartmentalized. Difficulty to fix: very high.
def create_metadata(
    directory: str,
    cls2idx: dict[str, int] | dict[str, dict[str, int]],
    labels: OrderedDict[str, str] | OrderedDict[str, tuple[str, ...]] | list[str] | None,
    train_proportion: float = 0.9,
    val_proportion: float = 0.5,
    **kwargs,
):
    """Create training metadata.

    TODO: This function has too many responsibilities, and crosses semantic boundaries.
    """
    if directory.endswith(".parquet"):
        return get_metadata_from_parquet(directory, cls2idx=cls2idx)
    metadata = {"path": [], "class": [], "split": [], "label": []}
    if labels is None:
        # If no labels are supplied we just assume that the images are put into
        # folders named after the class
        if isinstance(cls2idx.get("0", None), dict):
            if not is_taxonomical_cls2idx(cls2idx):
                raise ValueError("Hierarchical class index passed without labels and is not taxonomical.")
            dirs = [
                name
                for name in os.listdir(directory)
                if os.path.isdir(subdir := os.path.join(directory, name)) and len(os.listdir(subdir)) > 0
            ]
            tax = create_taxonomy(dirs, len(cls2idx))
            labels = labels_from_taxonomy(tax)
            del dirs, tax
        else:
            labels = OrderedDict(
                (name, name)
                for name in sorted(os.listdir(directory))
                if os.path.isdir(subdir := os.path.join(directory, name)) and len(os.listdir(subdir)) > 0
            )
    elif isinstance(labels, list):
        # Same if it is a list, in this case we just assume the folders
        # are named after the (leaf) class
        labels = OrderedDict([(lab[0] if isinstance(lab, (list, tuple)) else lab, lab) for lab in labels])
    for dir, cls in labels.items():
        if isinstance(cls, str):
            cls2idx = cast(dict[str, int], cls2idx)
            idx = cls2idx.get(cls, None)
        else:
            cls2idx = cast(dict[str, dict[str, int]], cls2idx)
            idx = [cls2idx[str(lvl)].get(c, None) for lvl, c in enumerate(cls)]
        this_dir = os.path.join(directory, str(dir))
        for image_path in find_images(this_dir):
            metadata["path"].append(image_path)
            metadata["class"].append(idx)
            metadata["split"].append(
                "train" if random.random() < train_proportion else "validation" if random.random() < val_proportion else "test"
            )
            metadata["label"].append(cls)
    return metadata


def get_metadata(
    path: str,
    splits: tuple[str, ...] = ("train", "validation"),
    check_integrity: bool = False,
    cls2idx: dict[str, int] | dict[str, dict[str, int]] | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Load training metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Meta data file ({path}) for training split not found. "
            'Please provide a JSON with the following keys: "path", "class" or "label", "split".'
        )
    with open(path, "rb") as f:
        data = json.load(f)
        if "path" in data:
            base_dir = os.path.dirname(path) or "."
            data["path"] = [os.path.relpath(os.path.join(base_dir, p)) for p in data["path"]]
        metadata = {k: np.array(v) for k, v in data.items()}
    if check_integrity:
        integrity_mask = np.array(is_image(metadata["path"]))
        metadata = {k: v[integrity_mask] for k, v in metadata.items()}
    if "class" not in metadata:
        if "label" not in metadata:
            raise KeyError(f"No 'class's or 'label's found in {path}")
        if cls2idx is None:
            raise TypeError(f"Found 'label's in {path}, but no 'cls2idx' was supplied.")
        multilabel = isinstance(list(cls2idx.values())[0], dict)
        if not multilabel:
            cls2idx = {"0": cast(dict[str, int], cls2idx)}
        cls2idx = cast(dict[str, dict[str, int]], cls2idx)
        levels = sorted(cls2idx.keys(), key=int)
        metadata["class"] = np.array(
            [[cls2idx[level][lab] for level, lab in zip(levels, labs if labs.size > 1 else [labs])] for labs in metadata["label"]]
        )
        if not multilabel:
            metadata["class"] = np.array([c[0] for c in metadata["class"]])
    return metadata


def parse_class_spec(path: str | None = None, dir: str | None = None) -> dict[str, dict[str, int]]:
    """Create or load (flat) class specification."""
    if path is None or not os.path.exists(path):
        if dir is None or not os.path.isdir(dir):
            # Special: If `dir` is a parquet, in this case we assume
            # it is a parquet generated by `gbifxdl`
            if isinstance(dir, str) and dir.endswith(".parquet"):
                retval = parquet_to_class_spec(dir)
            else:
                raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not "{dir}".')
        else:
            cls2idx = {
                cls: i
                for i, cls in enumerate(
                    sorted(filter(lambda f: os.path.isdir(os.path.join(dir, f)), map(os.path.basename, os.listdir(dir))))
                )
            }
            retval = {"cls2idx": cls2idx, "num_classes": len(cls2idx)}
        if path is not None:
            with open(path, "w") as f:
                json.dump(retval, f)
        else:
            return retval
    with open(path, "rb") as f:
        return json.load(f)
