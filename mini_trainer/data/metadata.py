import csv
import json
import os
import random
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, cast

import numpy as np

from mini_trainer.integrations import (
    create_taxonomy,
    get_metadata_from_parquet,
    is_taxonomical_cls2idx,
    labels_from_taxonomy,
    parquet_to_class_spec,
    resolve_name_or_id,
)
from mini_trainer.integrations.parquet import KCOLUMNS, iter_parquet, path_from_class

from .io import is_image


def find_images(root: str):
    """Find all images within root."""
    paths = sorted(glob(os.path.join(root, "**"), recursive=True))
    check = is_image(paths)
    return [p for p, f in zip(paths, check) if f]


def _samples_from_dict(data: dict, base_dir: Path | str | None = None) -> list[tuple[str, str]]:
    """Extract (image_path, raw_class_label) pairs from a dictionary or loaded file."""
    if "data_index" in data and isinstance(data["data_index"], (str, Path)) and os.path.exists(str(data["data_index"])):
        return collect_samples_from_source(str(data["data_index"]))

    path_key = next((k for k in ("path", "filename", "filepath", "image", "file") if k in data), None)
    label_key = next((k for k in ("label", "labels", "class", "classes", "species", "speciesKey") if k in data), None)

    if path_key is None or label_key is None:
        raise KeyError(f"Dict must contain path ({('path', 'filename')}) and label ({('label', 'class')}) keys.")

    paths = data[path_key]
    labels = data[label_key]

    if not isinstance(paths, (list, tuple, np.ndarray)) or not isinstance(labels, (list, tuple, np.ndarray)):
        raise TypeError("Path and label entries in dict must be sequences of values.")

    samples = []
    for p, lbl in zip(paths, labels):
        p_str = str(p)
        if base_dir is not None and not os.path.isabs(p_str):
            p_str = os.path.normpath(os.path.join(base_dir, p_str))
        lbl_str = str(lbl[0] if isinstance(lbl, (list, tuple, np.ndarray)) and len(lbl) > 0 else lbl)
        samples.append((p_str, lbl_str))

    return samples


def collect_samples_from_source(source: str | Path | dict) -> list[tuple[str, str]]:
    """Collect (image_path, raw_class_label) pairs from various dataset formats."""
    if isinstance(source, dict):
        return _samples_from_dict(source)

    src_path = Path(source).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset source '{src_path}' does not exist.")

    if src_path.is_dir():
        if (src_path / "data_index.json").exists():
            return collect_samples_from_source(src_path / "data_index.json")

        subdirs = sorted(d for d in src_path.iterdir() if d.is_dir() and not d.name.startswith("."))
        if subdirs:
            samples = []
            for subdir in subdirs:
                cls_name = subdir.name
                for img in find_images(str(subdir)):
                    samples.append((img, cls_name))
            if samples:
                return samples

        # Flat directory containing images
        images = find_images(str(src_path))
        if images:
            return [(img, src_path.name) for img in images]
        raise RuntimeError(f"Directory '{src_path}' does not contain any images or class subdirectories.")

    ext = src_path.suffix.lower()
    match ext:
        case ".parquet":
            samples = []
            root_dir = str(src_path.parent)
            for row in iter_parquet(str(src_path), ("filename", KCOLUMNS[0])):
                gid = str(int(row[KCOLUMNS[0]]))
                fn = str(row["filename"])
                img_path = path_from_class(file=fn, gid=int(gid), dir=root_dir)
                samples.append((img_path, gid))
            return samples

        case ".json":
            with open(src_path, encoding="utf8") as f:
                data = json.load(f)
            return _samples_from_dict(data, base_dir=src_path.parent)

        case ".csv":
            with open(src_path, encoding="utf8") as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader)]
                data = {h: [] for h in headers}
                for row in reader:
                    for h, val in zip(headers, row):
                        data[h].append(val.strip())
            return _samples_from_dict(data, base_dir=src_path.parent)

        case _:
            raise ValueError(f"Unsupported file format '{ext}' for dataset source '{src_path}'.")


def partition_class_samples(
    samples: list[str],
    proportions: dict[str, float],
    min_freqs: dict[str, int],
    rng: random.Random | None = None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Partition samples of a single class into splits respecting target proportions and minimum frequencies.

    Returns:
        split_samples: mapping from split_name -> list of sample paths.
        violations: mapping from split_name -> count deficit below min_freq (0 if met).
    """
    if rng is None:
        rng = random.Random()
    n = len(samples)
    splits = list(proportions.keys())
    total_p = sum(proportions.values())
    norm_p = {s: proportions[s] / total_p for s in splits}

    shuffled = list(samples)
    rng.shuffle(shuffled)

    total_min = sum(min_freqs.get(s, 0) for s in splits)

    if n <= total_min:
        counts = {s: 0 for s in splits}
        remaining = n
        sorted_splits = sorted(splits, key=lambda s: (min_freqs.get(s, 0), s == "test"), reverse=True)
        for s in sorted_splits:
            need = min_freqs.get(s, 0)
            alloc = min(need, remaining)
            counts[s] = alloc
            remaining -= alloc
        if remaining > 0:
            for s in sorted_splits:
                counts[s] += remaining
                break
    else:
        counts = {s: min_freqs.get(s, 0) for s in splits}
        remainder = n - total_min

        ideal_extra = {s: max(0.0, n * norm_p[s] - min_freqs.get(s, 0)) for s in splits}
        sum_extra = sum(ideal_extra.values())
        extra_ratios = {s: ideal_extra[s] / sum_extra for s in splits} if sum_extra > 0 else norm_p

        exact_alloc = {s: remainder * extra_ratios[s] for s in splits}
        floor_alloc = {s: int(exact_alloc[s]) for s in splits}
        unassigned = remainder - sum(floor_alloc.values())

        remainders = sorted(splits, key=lambda s: exact_alloc[s] - floor_alloc[s], reverse=True)
        for s in remainders[:unassigned]:
            floor_alloc[s] += 1

        for s in splits:
            counts[s] += floor_alloc[s]

    violations = {s: max(0, min_freqs.get(s, 0) - counts[s]) for s in splits}

    split_samples = {}
    offset = 0
    for s in splits:
        cnt = counts[s]
        split_samples[s] = shuffled[offset : offset + cnt]
        offset += cnt

    return split_samples, violations


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
            metadata = create_metadata(src, **{**kwargs, **{"train_proportion": 0, "val_proportion": 0}})
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


def parse_class_spec(path: str | None = None, dir: str | None = None, species: bool = False) -> dict[str, Any]:
    """Create or load (flat) class specification."""
    if path is not None and os.path.exists(path):
        with open(path, "rb") as f:
            data: dict[str, Any] = json.load(f)
        assert isinstance(data, dict)
    else:
        if dir is None or not os.path.isdir(dir):
            # Special: If `dir` is a parquet, in this case we assume
            # it is a parquet generated by `gbifxdl`
            if isinstance(dir, str) and dir.endswith(".parquet"):
                data = parquet_to_class_spec(dir)
            else:
                raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not "{dir}".')
        else:
            cls2idx = {
                cls: i
                for i, cls in enumerate(
                    sorted(filter(lambda f: os.path.isdir(os.path.join(dir, f)), map(os.path.basename, os.listdir(dir))))
                )
            }
            data = {"cls2idx": cls2idx, "num_classes": len(cls2idx)}
    if species:
        cls2idx = data["cls2idx"]
        assert isinstance(cls2idx, dict)
        cls = list(cls2idx.keys())
        taxs = resolve_name_or_id(cls)
        ids_list = [[id for id, _ in tax.values()] for tax in taxs]
        cls = [c for c, ids in sorted(zip(cls, ids_list), key=lambda kv: kv[1][::-1])]
        data["cls2idx"] = {c: i for i, c in enumerate(cls)}
    if path is not None:
        with open(path, "w") as f:
            json.dump(data, f)
    return data
