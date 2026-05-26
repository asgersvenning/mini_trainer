import csv
import os
from glob import glob
from itertools import repeat
from typing import Any

MISSING_VALUE = "NA"


def write_csv_from_dict(d: dict[str, Any], path: str):
    """Write to existing or new CSV from dict."""
    headers = tuple(d.keys())
    nrow = -1
    for h in headers:
        v = d[h]
        if not hasattr(v, "__len__"):
            raise TypeError(f"All values must be sequences, but {h} is {type(v)}")
        if nrow == -1:
            nrow = len(v)
        if nrow != len(v):
            raise ValueError(f"All values must have the same length, but {h} has length {len(v)}, and expected {nrow}")
    mode = "a" if os.path.exists(path) else "w"
    if mode == "a":
        with open(path, newline="", encoding="utf-8") as f:
            try:
                existing_headers = tuple(next(csv.reader(f)))
            except StopIteration:
                existing_headers = ()
        if not all([h in existing_headers for h in headers]):
            raise RuntimeError(f"Mismatching columns in {path}, found:\n\t{existing_headers}\nbut expected:\n\t{headers}")
        if len(headers) != len(existing_headers):
            for h in existing_headers:
                if h not in headers:
                    d[h] = repeat(MISSING_VALUE)
        headers = existing_headers
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(headers)
        writer.writerows(zip(*(d[h] for h in headers)))


def increment_name_dir(name: str, dir: str | None = None, max_iter: int = 1000):  # noqa: D103
    if name is None:
        raise ValueError("A name must be specified.")
    if not isinstance(name, str):
        raise TypeError(f"Invalid type `{type(name)}` used for the name. Only `str` is accepted.")
    if len(name) == 0:
        raise ValueError("Invalid zero-length name specified.")
    if dir is None:
        return name

    def _name(i: int):
        if i < 0:
            raise RuntimeError(f"Invalid name iteration {i} specified.")
        if i == 0:
            return name
        return f"{name}_{i}"

    fs = set([os.path.splitext(os.path.basename(f))[0] for f in glob(name + "*", root_dir=dir)])
    i0 = 0
    if "_" in name and (parts := name.split("_"))[-1].isdigit():
        i0 = int(parts[-1])
        name = "_".join(parts[:-1])
    for i in range(i0, max_iter + 1):
        if (this := _name(i)) not in fs:
            return this

    raise RuntimeError(
        f"Unable to create a new model name from {name} in {dir}, "
        f"the maximum number of model iterations with the same base name {max_iter} has been reached. "
        "OBS: The name check is file-extension agnostic!"
    )
