import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, cast

from tqdm import tqdm

_HAS_PYARROW = True
try:
    import pyarrow.compute as pc
    import pyarrow.parquet as pp
except ImportError:
    _HAS_PYARROW = False


def _check_pyarrow():
    if not _HAS_PYARROW:
        raise ImportError(
            "Parquet integration requires the optional dependency: pyarrow. Install with `pip install mini_trainer[recommended]`."
        )


KCOLUMNS = ("speciesKey", "genusKey", "familyKey", "orderKey", "classKey", "phylumKey", "kingdomKey")

COLUMNS = ("filename", "set", *KCOLUMNS)


def nrow(path: str):
    _check_pyarrow()
    return sum(p.count_rows() for p in pp.ParquetDataset(path).fragments)


def iter_parquet_batches(path: str, columns=COLUMNS):
    """Iterate lazily over batches in ``gbifxdl`` parquet."""
    _check_pyarrow()
    # 1. Ensure "set" is loaded so we can filter by it
    read_columns = list(columns)
    if "set" not in read_columns:
        read_columns.append("set")

    for batch in pp.ParquetFile(path).iter_batches(columns=read_columns):
        filtered = batch.filter(
            pc.match_substring_regex(pc.field("set"), pattern="^\\d+$")  # type: ignore
        ).select(
            list(columns)  # 2. Select ONLY the originally requested columns
        )
        if filtered.num_rows > 0:
            yield filtered


def iter_parquet(path: str, columns=COLUMNS):
    """Iterate lazily over individual rows in ``gbifxdl`` parquet."""
    for batch in iter_parquet_batches(path, columns=columns):
        yield from batch.to_pylist()


def set2split(set: int):
    """Map: 0=test, 1=validation, *=train."""
    match set:
        case 0:
            return "test"
        case 1:
            return "validation"
        case _:
            return "train"


def path_from_class(file: str, gid: int, dir: str):
    """Compose full path from basename, class, and root directory."""
    return os.path.join(dir, "images", str(gid), file)


def get_keys(row: dict[str, Any]):
    """Get class values from ``gbifxdl` row."""
    return [str(int(row[k].strip())) for k in KCOLUMNS]


def combine_dicts(dicts: Sequence[dict]):
    """Combine dictionaries with shared keys by stacking values in lists."""
    if not isinstance(dicts, list):
        dicts = list(dicts)
    if len(dicts) == 0:
        return dict()
    retval = {k: [] for k in dicts[0].keys()}
    for d in dicts:
        for k, v in d.items():
            retval[k].append(v)
    return retval


def get_metadata_from_parquet(
    path: str,
    cls2idx: dict[str, int] | dict[str, dict[str, int]],
    **kwargs,
) -> dict[str, list]:
    root_dir = os.path.dirname(os.path.abspath(path))
    base_dir = os.path.join(root_dir, "images") + os.sep

    first_val = cls2idx[next(iter(cls2idx))]
    is_hierarchical = isinstance(first_val, dict)

    out_split: list[str] = []
    out_class: list[int | tuple[int | None, ...] | None] = []
    out_path: list[str] = []
    out_label: list[str | tuple[str, ...]] = []

    def _to_str(x: object) -> str:
        return "" if x is None else str(x)

    def _to_int_or_minus1(x: object) -> int:
        if x is None:
            return -1
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x) if x.is_integer() else -1
        try:
            s = str(x).strip()
            if s == "":
                return -1
            return int(float(s))
        except Exception:
            return -1

    path_str = path if len(path) < (25 + 3) else ("..." + path[-min(25, len(path)) :])
    with tqdm(
        total=nrow(path),
        desc=f"Parsing metadata from {path_str}",
    ) as pbar:
        for batch in iter_parquet_batches(path):
            names = batch.schema.names
            idx = {name: i for i, name in enumerate(names)}

            missing = [c for c in ("filename", "set") if c not in idx]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")

            for k in KCOLUMNS:
                if k not in idx:
                    raise KeyError(f"Missing required key column: {k}")

            filename_list = batch.column(idx["filename"]).to_pylist()
            set_list = batch.column(idx["set"]).to_pylist()

            gid_list = batch.column(idx[KCOLUMNS[0]]).to_pylist()

            n = len(filename_list)
            if not (len(set_list) == len(gid_list) == n):
                raise ValueError("Column lengths mismatch inside a parquet batch.")

            split_batch: list[str] = []
            path_batch: list[str] = []
            for gid, fn, s in zip(gid_list, filename_list, set_list, strict=True):
                si = _to_int_or_minus1(s)
                split_batch.append(set2split(si))
                path_batch.append(base_dir + _to_str(gid) + os.sep + _to_str(fn))

            if is_hierarchical:
                level_maps: list[dict[str, int]] = []
                for level in range(len(cls2idx)):
                    lm = cls2idx.get(str(level))
                    if not isinstance(lm, dict):
                        raise TypeError(f"Expected hierarchical cls2idx['{level}'] to be dict[str,int], got {type(lm)}")
                    level_maps.append(lm)

                key_lists = [batch.column(idx[KCOLUMNS[level]]).to_pylist() for level in range(len(level_maps))]

                class_batch = []
                label_batch = []
                for i in range(n):
                    labels = tuple(_to_str(key_lists[level][i]).strip() for level in range(len(level_maps)))
                    classes = tuple(level_maps[level].get(labels[level]) for level in range(len(level_maps)))
                    label_batch.append(labels)
                    class_batch.append(classes)

                out_label.extend(label_batch)
                out_class.extend(class_batch)
            else:
                mapping = cast(dict[str, int], cls2idx)
                key_list = batch.column(idx[KCOLUMNS[0]]).to_pylist()

                label_batch = [_to_str(v).strip() for v in key_list]
                class_batch = [mapping.get(lbl) for lbl in label_batch]

                out_label.extend(label_batch)
                out_class.extend(class_batch)

            out_split.extend(split_batch)
            out_path.extend(path_batch)

            pbar.update(n)

    return {
        "split": out_split,
        "class": out_class,
        "path": out_path,
        "label": out_label,
    }


def parquet_to_class_spec(path: str):
    """Create flat class specification from ``gbifxdl`` parquet."""
    clss = set([row["speciesKey"].strip() for row in iter_parquet(path, ("speciesKey",))])
    return {"cls2idx": {cls: i for i, cls in enumerate(clss)}, "num_classes": len(clss)}


def parquet_to_class_spec_hierarchical(
    path: str, levels: int = 3
) -> dict[str, dict[str, dict[str, int]] | OrderedDict[str, tuple[str, ...]] | list[int]]:
    """Create hierarchical class specification from ``gbifxdl`` parquet."""
    combs = OrderedDict(
        (v[0], v) for v in sorted(set([tuple(get_keys(row)[:levels]) for row in iter_parquet(path, KCOLUMNS)]), key=lambda x: x[::-1])
    )
    cls2idx: dict[str, dict[str, int]] = dict()
    for level in range(levels):
        clss = set()
        this_cls2idx: dict[str, int] = dict()
        for _, comb in combs.items():
            cls = comb[level]
            if cls in clss:
                continue
            this_cls2idx[cls] = len(clss)
            clss.add(cls)
        cls2idx[str(level)] = this_cls2idx
    num_classes = [len(cls2idx[str(i)]) for i in range(len(cls2idx))]
    return {"cls2idx": cls2idx, "labels": combs, "num_classes": num_classes}
