"""Release image geometry with portable CPU and batched Torch finishing."""

import io
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
    if isinstance(item, (str, Path, bytes)):
        with Image.open(io.BytesIO(item) if isinstance(item, bytes) else item) as im:
            return _rgb(im)
    if isinstance(item, Image.Image):
        array = np.asarray(item if item.mode == "RGB" else item.convert("RGB"), dtype=np.uint8).transpose(2, 0, 1)
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
# FP32 is sufficient for image interpolation; avoid promoting every temporary to FP64.
_FRACTION = _COORD - _LO.astype(np.float32)
_MEAN = np.array(RECIPE["mean"], dtype=np.float32)[:, None, None]
_STD = np.array(RECIPE["std"], dtype=np.float32)[:, None, None]


def _nearest_indices(length, padding=0):
    pad = int(np.ceil(length * padding))
    return np.clip((_GRID * np.float32((length + 2 * pad) / _SIZE)).astype(np.intp) - pad, 0, length - 1)


def _square(item, padding=0):
    image = _rgb(item)
    height, width = image.shape[1:]
    if height == width == _SIZE and padding == 0:
        return image
    return image[:, _nearest_indices(height, padding)[:, None], _nearest_indices(width, padding)[None, :]]


def prepare_uint8(item, out=None, *, padding=0):
    """Select compact pixels; virtual edge padding avoids a full-size padded image."""
    square = _square(item, padding)
    if out is None:
        return np.ascontiguousarray(square)
    out[...] = square
    return out


def preprocess(item, out=None, *, padding=0):
    """Finish the release geometry and normalization in FP32 on CPU."""
    image = np.ascontiguousarray(_square(item, padding), dtype=np.float32)
    rows = image[:, _LO] * (1 - _FRACTION)[None, :, None] + image[:, _HI] * _FRACTION[None, :, None]
    pixels = rows[:, :, _LO] * (1 - _FRACTION)[None, None, :] + rows[:, :, _HI] * _FRACTION[None, None, :]
    if out is None:
        out = np.empty((3, _SIZE, _SIZE), dtype=np.float32)
    np.rint(pixels, out=out)
    out /= 255
    out -= _MEAN
    out /= _STD
    return out


def prepare_batch(items, pool=None, transform=None, *, compact=False, decode=_rgb):
    """Fill one contiguous batch without per-image output allocations and stacking."""
    output = np.empty((len(items), 3, _SIZE, _SIZE), dtype=np.uint8 if compact else np.float32)
    prepare = prepare_uint8 if compact else preprocess

    def fill(index):
        item = decode(items[index])
        if transform is not None:
            item = transform(item.copy())
        prepare(item, out=output[index])

    if pool is not None and len(items) > 1:
        for _ in pool.map(fill, range(len(items))):
            pass
    else:
        for index in range(len(items)):
            fill(index)
    return output


class TorchDecode:
    """Native CPU JPEG/PNG decoding, using the Torch backend's existing dependency."""

    def __init__(self, torch):
        from torchvision.io import ImageReadMode, decode_image

        self.torch, self.decode, self.mode = torch, decode_image, ImageReadMode.RGB

    def __call__(self, item):
        if isinstance(item, (str, Path)):
            item = Path(item).read_bytes()
        if isinstance(item, bytes) and item.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")):
            # Writable encoded storage avoids a read-only tensor view; only compressed
            # bytes are copied. Decoded pixels stay in native storage shared with NumPy.
            encoded = self.torch.frombuffer(bytearray(item), dtype=self.torch.uint8)
            decoded = self.decode(encoded, mode=self.mode, apply_exif_orientation=False)
            if decoded.dtype == self.torch.uint8:
                return decoded.numpy()
            # Preserve Pillow RGB conversion for high-bit-depth PNGs.
        return _rgb(item)


class TorchPreprocess:
    """Finish a batch of uint8 squares with native operations on its Torch device."""

    def __init__(self, torch, device):
        self.torch = torch
        self.mean = torch.as_tensor(_MEAN, device=device)
        self.std = torch.as_tensor(_STD, device=device)

    def __call__(self, images):
        torch = self.torch
        with torch.autocast(images.device.type, enabled=False):
            values = torch.nn.functional.interpolate(
                images.float(), size=(_RESIZED, _RESIZED), mode="bilinear", align_corners=False, antialias=False
            )
            start = (_RESIZED - _SIZE) // 2
            values = values[..., start : start + _SIZE, start : start + _SIZE]
            return values.round_().div_(255).sub_(self.mean).div_(self.std)


def image_items(value):
    if isinstance(value, (str, Path, Image.Image)):
        return iter([value])
    if hasattr(value, "ndim") and value.ndim in (2, 3):
        return iter([value])
    return iter(value)
