import csv
import json
import os
import random
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import Any, cast

import numpy as np

from mini_trainer.integrations import (
    cls2idx_from_labels,
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


SPLIT_DIR_MAP: dict[str, str] = {
    "train": "train",
    "training": "train",
    "trn": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "eval": "validation",
    "evaluation": "validation",
    "test": "test",
    "testing": "test",
    "tst": "test",
}


def _samples_from_dict(data: dict, base_dir: Path | str | None = None) -> list[tuple[str, str, str | None]]:
    """Extract (image_path, raw_class_label, split) tuples from a dictionary or loaded file."""
    if "data_index" in data and isinstance(data["data_index"], (str, Path)) and os.path.exists(str(data["data_index"])):
        return collect_samples_from_source(str(data["data_index"]))

    path_key = next((k for k in ("path", "filename", "filepath", "image", "file") if k in data), None)
    label_key = next((k for k in ("label", "labels", "class", "classes", "species", "speciesKey") if k in data), None)
    split_key = next((k for k in ("split", "splits", "set", "subset") if k in data), None)

    if path_key is None or label_key is None:
        raise KeyError(f"Dict must contain path ({('path', 'filename')}) and label ({('label', 'class')}) keys.")

    paths = data[path_key]
    labels = data[label_key]
    splits = data[split_key] if split_key is not None else None

    if not isinstance(paths, (list, tuple, np.ndarray)) or not isinstance(labels, (list, tuple, np.ndarray)):
        raise TypeError("Path and label entries in dict must be sequences of values.")

    samples = []
    for idx, (p, lbl) in enumerate(zip(paths, labels)):
        p_str = str(p)
        if base_dir is not None and not os.path.isabs(p_str):
            p_str = os.path.normpath(os.path.join(base_dir, p_str))
        lbl_str = str(lbl[0] if isinstance(lbl, (list, tuple, np.ndarray)) and len(lbl) > 0 else lbl)
        s_str = str(splits[idx]).strip().lower() if splits is not None else None
        samples.append((p_str, lbl_str, s_str))

    return samples


def collect_samples_from_source(source: str | Path | dict | Sequence[Any]) -> list[tuple[str, str, str | None]]:
    """Collect (image_path, raw_class_label, split) tuples from various dataset formats."""
    if isinstance(source, (list, tuple)):
        if len(source) > 0 and isinstance(source[0], (tuple, list)):
            return [
                (s[0], s[1], s[2] if len(s) >= 3 else None)
                for s in source
            ]
        return [s for src in source for s in collect_samples_from_source(src)]

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
            # Check if this is a pre-split directory structure (e.g. train/, valid/, test/)
            if all(d.name.lower() in SPLIT_DIR_MAP for d in subdirs) and any(
                any(child.is_dir() for child in d.iterdir() if not child.name.startswith(".")) for d in subdirs
            ):
                samples = []
                for split_dir in subdirs:
                    assigned_split = SPLIT_DIR_MAP[split_dir.name.lower()]
                    split_samples = collect_samples_from_source(split_dir)
                    for item in split_samples:
                        samples.append((item[0], item[1], assigned_split))
                if samples:
                    return samples

            samples = []
            for subdir in subdirs:
                cls_name = subdir.name
                for img in find_images(str(subdir)):
                    samples.append((img, cls_name, None))
            if samples:
                return samples

        # Flat directory containing images
        images = find_images(str(src_path))
        if images:
            return [(img, src_path.name, None) for img in images]
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
                samples.append((img_path, gid, None))
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

        case ".txt" | ".tsv":
            samples = []
            with open(src_path, encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t") if "\t" in line else line.split()
                    if len(parts) >= 2:
                        p_str = parts[0]
                        if not os.path.isabs(p_str):
                            p_str = os.path.normpath(os.path.join(src_path.parent, p_str))
                        lbl_str = parts[1]
                        s_str = parts[2].lower() if len(parts) >= 3 else None
                        samples.append((p_str, lbl_str, s_str))
            return samples

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


def label_to_class_idx(
    label: str | tuple[str, ...] | list[str] | np.ndarray,
    cls2idx: dict[str, int] | dict[str, dict[str, int]],
) -> int | list[int] | None:
    """Map a label (flat string or hierarchical tuple/list) to class index integer(s)."""
    if isinstance(label, (list, tuple, np.ndarray)):
        cls2idx_hier = cast(dict[str, dict[str, int]], cls2idx)
        if "0" in cls2idx_hier and isinstance(cls2idx_hier["0"], dict):
            return [cls2idx_hier[str(lvl)].get(str(c), None) for lvl, c in enumerate(label)]
        return cls2idx.get(tuple(map(str, label)), None) or cls2idx.get(label, None)
    cls2idx_flat = cast(dict[str, int], cls2idx)
    return cls2idx_flat.get(str(label), None) if str(label) in cls2idx_flat else cls2idx_flat.get(label, None)


# TODO: Unfortunately, this function has some functionality for the hierarchical submodule
# even though the core mini_trainer module and the hierarchical submodule are
# supposed to be entirely compartmentalized. Difficulty to fix: very high.
def create_metadata(
    directory: str | Path | dict | list[str | Path | dict],
    cls2idx: dict[str, int] | dict[str, dict[str, int]] | None = None,
    labels: OrderedDict[str, str] | OrderedDict[str, tuple[str, ...]] | list[str] | None = None,
    train_proportion: float = 0.9,
    val_proportion: float = 0.5,
    test_proportion: float | None = None,
    min_freqs: dict[str, int] | None = None,
    output: str | Path | None = None,
    relative_paths: bool = False,
    seed: int | None = None,
    **kwargs,
) -> dict[str, list]:
    """Create or generate dataset metadata / data index."""
    if isinstance(directory, (str, Path)) and str(directory).endswith(".parquet"):
        return get_metadata_from_parquet(str(directory), cls2idx=cls2idx or {})

    # Hierarchical taxonomy construction if applicable
    if labels is None and isinstance(cls2idx, dict) and isinstance(cls2idx.get("0", None), dict) and isinstance(directory, (str, Path)):
        dir_str = str(directory)
        if not is_taxonomical_cls2idx(cls2idx):
            raise ValueError("Hierarchical class index passed without labels and is not taxonomical.")
        dirs = [
            name
            for name in os.listdir(dir_str)
            if os.path.isdir(os.path.join(dir_str, name)) and len(os.listdir(os.path.join(dir_str, name))) > 0
        ]
        tax = create_taxonomy(dirs, len(cls2idx))
        labels = labels_from_taxonomy(tax)

    if isinstance(labels, list):
        dir_str = str(directory) if isinstance(directory, (str, Path)) else "."
        labels_map = OrderedDict([(lab[0] if isinstance(lab, (list, tuple)) else lab, lab) for lab in labels])
        samples = [
            (img, tuple(cls) if isinstance(cls, list) else cls, None)
            for d, cls in labels_map.items()
            for img in find_images(os.path.join(dir_str, str(d)))
        ]
    elif isinstance(labels, OrderedDict):
        dir_str = str(directory) if isinstance(directory, (str, Path)) else "."
        samples = [
            (img, tuple(cls) if isinstance(cls, list) else cls, None)
            for d, cls in labels.items()
            for img in find_images(os.path.join(dir_str, str(d)))
        ]
    else:
        raw_samples = collect_samples_from_source(directory)
        samples = []
        for item in raw_samples:
            p, lbl = item[0], item[1]
            s = item[2] if len(item) >= 3 else None
            resolved_lbl = labels.get(lbl, lbl) if isinstance(labels, dict) else lbl
            norm_lbl = tuple(resolved_lbl) if isinstance(resolved_lbl, list) else resolved_lbl
            samples.append((p, norm_lbl, s))

    has_presplit = len(samples) > 0 and all(s[2] is not None for s in samples)
    unique_classes = sorted(set(s[1] for s in samples))
    if cls2idx is not None:
        resolved_cls2idx = cls2idx
    elif unique_classes and isinstance(unique_classes[0], (list, tuple)):
        resolved_cls2idx = cls2idx_from_labels(
            OrderedDict([(f"cls_{i}", tuple(map(str, lab))) for i, lab in enumerate(unique_classes)])
        )
    else:
        resolved_cls2idx = {str(c): i for i, c in enumerate(unique_classes)}
    metadata: dict[str, list] = {"path": [], "class": [], "split": [], "label": []}
    out_base_dir = Path(output).parent if output is not None else None

    if has_presplit:
        target_splits = ["train", "test"] if (test_proportion is not None and val_proportion == 0.0) else ["train", "validation", "test"]
        for p, lbl, raw_split in samples:
            raw_s = str(raw_split).strip().lower()
            if raw_s.startswith("train") or raw_s == "trn":
                norm_split = "train"
            elif raw_s.startswith("test") or raw_s == "tst":
                norm_split = "test"
            else:
                norm_split = "validation"
            if "validation" not in target_splits and norm_split == "validation":
                norm_split = "test"
            cls_idx = label_to_class_idx(lbl, resolved_cls2idx)
            p_out = os.path.relpath(p, out_base_dir) if (relative_paths and out_base_dir is not None) else p
            metadata["path"].append(p_out)
            metadata["class"].append(cls_idx)
            metadata["split"].append(norm_split)
            metadata["label"].append(lbl)
    else:
        # Group samples by class label
        class_samples: dict[Any, list[str]] = defaultdict(list)
        for p, lbl, _ in samples:
            class_samples[lbl].append(p)

        # Proportions: train, validation, test
        p_train = float(train_proportion)
        if test_proportion is not None:
            p_test = float(test_proportion)
            p_val = float(val_proportion) if val_proportion < 1.0 else max(0.0, 1.0 - p_train - p_test)
        else:
            p_val = float((1.0 - train_proportion) * val_proportion)
            p_test = max(0.0, 1.0 - p_train - p_val)

        proportions = {"train": p_train, "validation": p_val, "test": p_test}
        min_freqs_map = min_freqs or {}
        rng = random.Random(seed) if seed is not None else None

        for lbl, paths in class_samples.items():
            split_dict, _ = partition_class_samples(paths, proportions=proportions, min_freqs=min_freqs_map, rng=rng)
            cls_idx = label_to_class_idx(lbl, resolved_cls2idx)
            for split, split_paths in split_dict.items():
                for p in split_paths:
                    p_out = os.path.relpath(p, out_base_dir) if (relative_paths and out_base_dir is not None) else p
                    metadata["path"].append(p_out)
                    metadata["class"].append(cls_idx)
                    metadata["split"].append(split)
                    metadata["label"].append(lbl)

    if output is not None:
        out_file = Path(output)
        if out_file.is_dir() or not out_file.suffix:
            out_file = out_file / "data_index.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2)

    return metadata


def get_metadata(
    path: str | Path | dict,
    splits: tuple[str, ...] = ("train", "validation"),
    check_integrity: bool = False,
    cls2idx: dict[str, int] | dict[str, dict[str, int]] | None = None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Load training metadata."""
    if isinstance(path, dict):
        data = {k: list(v) for k, v in path.items()}
    else:
        src_path = Path(path).expanduser().resolve()
        if not src_path.exists():
            raise FileNotFoundError(
                f"Meta data file ({src_path}) for training split not found. "
                'Please provide a JSON with the following keys: "path", "class" or "label", "split".'
            )
        if src_path.suffix.lower() == ".parquet":
            return {k: np.array(v) for k, v in get_metadata_from_parquet(str(src_path), cls2idx=cls2idx or {}, **kwargs).items()}

        with open(src_path, "rb") as f:
            data = json.load(f)
        if "path" in data:
            base_dir = src_path.parent
            data["path"] = [os.path.relpath(os.path.join(base_dir, p)) if not os.path.isabs(p) else p for p in data["path"]]

    metadata = {k: np.array(v) for k, v in data.items()}
    if check_integrity and "path" in metadata:
        integrity_mask = np.array(is_image(metadata["path"]))
        metadata = {k: v[integrity_mask] for k, v in metadata.items()}

    if "class" not in metadata or len(metadata["class"]) == 0 or metadata["class"][0] is None:
        if "label" not in metadata:
            raise KeyError(f"No 'class' or 'label' found in {path}")
        if cls2idx is None:
            raise TypeError(f"Found 'label's in {path}, but no 'cls2idx' was supplied.")
        metadata["class"] = np.array([label_to_class_idx(lbl, cls2idx) for lbl in metadata["label"]])

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
