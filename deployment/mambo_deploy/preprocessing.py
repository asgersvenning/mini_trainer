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


# The model-space interpolation grid is fixed; source-size indexing is per image.
_SIZE, _RESIZED = RECIPE["crop_size"], RECIPE["resize_size"]
_GRID = np.arange(_SIZE, dtype=np.float32)
_COORD = np.maximum((np.arange(_RESIZED, dtype=np.float32) + 0.5) * np.float32(_SIZE / _RESIZED) - 0.5, 0)
_COORD = _COORD[(_RESIZED - _SIZE) // 2 : (_RESIZED + _SIZE) // 2]
_LO = np.floor(_COORD).astype(np.intp)
_HI = np.minimum(_LO + 1, _SIZE - 1)
# Keep release rounding at half-integer pixels; constants are cached once.
_FRACTION = _COORD - _LO
_MEAN = np.array(RECIPE["mean"], dtype=np.float32)[:, None, None]
_STD = np.array(RECIPE["std"], dtype=np.float32)[:, None, None]


def preprocess(item, out=None):
    """Apply the release geometry and normalize directly into optional batch storage."""
    image = _rgb(item)
    yy = np.minimum((_GRID * np.float32(image.shape[1] / _SIZE)).astype(np.intp), image.shape[1] - 1)
    xx = np.minimum((_GRID * np.float32(image.shape[2] / _SIZE)).astype(np.intp), image.shape[2] - 1)
    image = np.ascontiguousarray(image[:, yy[:, None], xx[None, :]], dtype=np.float32)
    rows = image[:, _LO] * (1 - _FRACTION)[None, :, None] + image[:, _HI] * _FRACTION[None, :, None]
    pixels = rows[:, :, _LO] * (1 - _FRACTION)[None, None, :] + rows[:, :, _HI] * _FRACTION[None, None, :]
    if out is None:
        out = np.empty((3, _SIZE, _SIZE), dtype=np.float32)
    np.rint(pixels, out=out)
    out /= 255
    out -= _MEAN
    out /= _STD
    return out


def prepare_batch(items, pool=None, transform=None):
    """Fill one contiguous batch without per-image output allocations and stacking."""
    output = np.empty((len(items), 3, _SIZE, _SIZE), dtype=np.float32)

    def fill(index):
        item = items[index]
        if transform is not None:
            item = transform(item.copy())
        preprocess(item, out=output[index])

    if pool is not None and len(items) > 1:
        for _ in pool.map(fill, range(len(items))):
            pass
    else:
        for index in range(len(items)):
            fill(index)
    return output


def image_items(value):
    if isinstance(value, (str, Path, Image.Image)):
        return iter([value])
    if hasattr(value, "ndim") and value.ndim in (2, 3):
        return iter([value])
    return iter(value)
