import csv
import os
import re
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


def increment_name_dir(name: str, dir: str | None = None, max_iter: int = 1000) -> str:
    """
    Sanitizes a base name and appends an incrementing integer if the name 
    already exists in the target directory.
    """
    if not isinstance(name, str):
        raise TypeError(f"Invalid type `{type(name).__name__}` used for the name. Only `str` is accepted.")

    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', "_", name)
    name = name.strip("_")

    if not name:
        raise ValueError("The sanitized name is empty. It must contain at least one valid character (A-Z, a-z, 0-9).")

    if dir is None:
        return name

    base_name = name
    i0 = 0
    if "_" in name:
        parts = name.split("_")
        if parts[-1].isdigit():
            i0 = int(parts[-1])
            base_name = "_".join(parts[:-1])

    fs = set()
    if os.path.exists(dir):
        with os.scandir(dir) as it:
            for entry in it:
                if entry.name.startswith(base_name):
                    fs.add(os.path.splitext(entry.name)[0])

    for i in range(i0, max_iter + 1):
        current_name = base_name if i == 0 else f"{base_name}_{i}"
        if current_name not in fs:
            return current_name

    raise RuntimeError(
        f"Unable to create a new model name from '{name}' in '{dir}'. "
        f"The maximum number of model iterations ({max_iter}) has been reached. "
        "OBS: The name check is file-extension agnostic!"
    )
