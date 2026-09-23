"""Campaign image recipe on CPU, shared by both runtimes."""

from pathlib import Path

import numpy as np
from PIL import Image

RECIPE = {
    "id": "nearest-square-uint8-bilinear-center-imagenet-v1",
    "decode": "RGB; discard alpha; ignore EXIF orientation",
    "input": "uint8 CHW/BCHW or image paths; float images must be in [0,1]",
    "square_size": 384,
    "resize_size": 438,
    "crop_size": 384,
    "nearest_coordinates": "floor(float32(output_index) * float32(source_size/384))",
    "bilinear": "half-pixel coordinates; edge clamp; round to uint8 before normalization",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "output": "float32 NCHW; RGB/255 then channel normalization",
}


def _rgb(item):
    if isinstance(item, (str, Path)):
        with Image.open(item) as im:
            array = np.asarray(im.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
    elif isinstance(item, Image.Image):
        array = np.asarray(item.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
    else:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        array = np.asarray(item)
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3 or array.shape[0] not in (1, 3, 4):
            raise ValueError("Images must be CHW with 1, 3 or 4 channels; transpose HWC arrays explicitly")
        if array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError("Image arrays must be uint8 or finite floating point in [0,1]")
            array = np.rint(array * 255).astype(np.uint8)
        if array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        array = array[:3]
    if min(array.shape[1:]) < 1:
        raise ValueError("Empty image")
    return array


def preprocess(item):
    image = _rgb(item)
    size, resized = 384, 438
    yy = np.minimum((np.arange(size, dtype=np.float32) * np.float32(image.shape[1] / size)).astype(int), image.shape[1] - 1)
    xx = np.minimum((np.arange(size, dtype=np.float32) * np.float32(image.shape[2] / size)).astype(int), image.shape[2] - 1)
    image = np.ascontiguousarray(image[:, yy][:, :, xx], dtype=np.float32)
    # Upsampling uses a bilinear support of one pixel (no downsampling antialias filter).
    coordinates = np.maximum((np.arange(resized, dtype=np.float32) + 0.5) * np.float32(size / resized) - 0.5, 0)
    offset = (resized - size) // 2
    coordinates = coordinates[offset : offset + size]
    lo = np.floor(coordinates).astype(int)
    hi = np.minimum(lo + 1, size - 1)
    fraction = coordinates - lo
    rows = image[:, lo] * (1 - fraction)[None, :, None] + image[:, hi] * fraction[None, :, None]
    rows = np.ascontiguousarray(rows)
    pixels = rows[:, :, lo] * (1 - fraction)[None, None, :] + rows[:, :, hi] * fraction[None, None, :]
    pixels = np.ascontiguousarray(np.rint(pixels).astype(np.float32) / 255)
    return np.ascontiguousarray(
        (pixels - np.array(RECIPE["mean"], dtype=np.float32)[:, None, None]) / np.array(RECIPE["std"], dtype=np.float32)[:, None, None]
    )


def image_items(value):
    if isinstance(value, (str, Path, Image.Image)):
        return iter([value])
    if hasattr(value, "ndim") and value.ndim in (2, 3):
        return iter([value])
    return iter(value)
