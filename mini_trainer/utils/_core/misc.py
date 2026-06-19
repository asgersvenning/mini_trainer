import shutil
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, TypeVarTuple

import numpy as np
import torch
from torchvision.transforms.v2 import ToDtype
from tqdm.auto import tqdm

from mini_trainer.utils import get_rank

TERMINAL_WIDTH, _ = shutil.get_terminal_size()

X = TypeVar("X")
R = TypeVar("R")
Ks = TypeVarTuple("Ks")


def make_empty_ndarray(s: int) -> np.typing.NDArray[np.float64]:
    """Create a 1-dimensional array filled with ``np.nan``."""
    arr = np.empty((s,))
    arr[:] = np.nan
    return arr


def filter_ordered_dict[K, V](od: OrderedDict[K, V], keys: Iterable[K]):  # noqa: UP047
    """Filter an `OrderedDict` by keys and maintain order."""
    return OrderedDict([(k, od[k]) for k in keys])


def make_convert_dtype(dtype: torch.dtype, scale: bool = True):
    """See `torchvision.transforms.v2.ToDtype`."""
    return ToDtype(dtype=dtype, scale=scale)


def recursive_dfs_attr(obj: Any, attr: str, predicate: Callable[[Any], bool] = lambda x: True) -> Any:
    """Helper function to search for a specific attribute in an object,
    or any attached objects.
    """
    seen = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if hasattr(current, attr):
            value = getattr(current, attr)
            if predicate(value):
                return value
        if isinstance(current, Iterable) and not isinstance(current, (str, bytes, bytearray)):
            try:
                stack.extend(current)
            except TypeError:
                pass  # non-iterable despite isinstance claiming so (rare, but safe)
    raise StopIteration(f"No attribute '{attr}' found passing predicate.")


class TQDM(tqdm):
    """Wrapper around tqdm.auto.tqdm that automatically disables output on non-zero DDP ranks."""

    def __new__(cls, *args, **kwargs):
        if get_rank() > 0:
            kwargs["disable"] = True
        return super().__new__(cls, *args, **kwargs)
