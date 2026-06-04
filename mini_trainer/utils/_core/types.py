import math
import shutil
import tempfile
from typing import Any

import psutil
import torch


def dtype_to_string(dtype: torch.dtype | str) -> str:
    dtype_str = str(dtype)
    return dtype_str.replace("torch.", "")


def string_to_dtype(dtype_str: str) -> torch.dtype:
    name = dtype_str.replace("torch.", "")
    return getattr(torch, name)


def device_to_string(device: torch.device | str | int) -> str:
    return str(torch.device(device))


def string_to_device(device_str: str) -> torch.device:
    return torch.device(device_str)


def memory_proportion(shape: tuple[int, ...], device: torch.types.Device, dtype: torch.dtype):
    """Compute the proportion of available space on device that would be used by a given tensor."""
    numel = math.prod(shape)
    # bytes per element
    bpe = torch.empty(0, dtype=dtype).element_size()
    required = numel * bpe
    if isinstance(device, str):
        device = device.lower().strip()
    if isinstance(device, str) and "disk" in device:
        free = shutil.disk_usage(tempfile.gettempdir()).free
    else:
        dev = torch.device(device)
        if dev.type == "cuda":
            free, _ = torch.cuda.mem_get_info(dev)
        else:
            free = psutil.virtual_memory().available

    return required / free


def validate_type(obj: Any, expected_cls: Any, allow_none: bool = False) -> None:
    if obj is None and allow_none:
        return
    if not isinstance(obj, expected_cls):
        if isinstance(expected_cls, tuple):
            expected = " or ".join(c.__qualname__ for c in expected_cls)
        else:
            expected = expected_cls.__qualname__
        raise TypeError(f"Expected an instance of {expected}, but got {type(obj).__qualname__}.")
