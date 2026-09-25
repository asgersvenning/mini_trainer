"""Outer, runtime-independent TTA over decoded CHW images."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .preprocessing import RECIPE, _nearest_indices, _rgb, _square, prepare_uint8, preprocess


@dataclass(frozen=True)
class View:
    """Fractional crop (top, left, bottom, right), quarter turns, then reflection."""

    crop: tuple = (0, 0, 1, 1)
    quarter_turns: int = 0
    hflip: bool = False

    def __post_init__(self):
        top, left, bottom, right = self.crop
        if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
            raise ValueError("View crop must be a nonempty fractional box within [0,1]")
        if not isinstance(self.quarter_turns, int):
            raise ValueError("quarter_turns must be an integer")

    def __call__(self, image):
        top, left, bottom, right = self.crop
        height, width = image.shape[1:]
        y, x = int(top * height), int(left * width)
        result = image[:, y : max(y + 1, int(bottom * height)), x : max(x + 1, int(right * width))]
        result = np.rot90(result, self.quarter_turns, axes=(1, 2))
        return result[..., ::-1] if self.hflip else result


@dataclass(frozen=True)
class EdgePad:
    """Pad each source edge by a fraction of its axis, preserving original pixels."""

    fraction: float

    def __post_init__(self):
        if not np.isfinite(self.fraction) or self.fraction < 0:
            raise ValueError("Padding fraction must be finite and nonnegative")

    def __call__(self, image):
        height, width = image.shape[1:]
        y, x = int(np.ceil(height * self.fraction)), int(np.ceil(width * self.fraction))
        return np.pad(image, ((0, 0), (y, y), (x, x)), mode="edge")


@dataclass(frozen=True)
class RotatePad:
    """Rotate on an expanded canvas, then edge-pad before ordinary preprocessing."""

    degrees: float
    padding: float = 0.25

    def __post_init__(self):
        if not np.isfinite(self.degrees):
            raise ValueError("Rotation must be finite")
        EdgePad(self.padding)

    def rotate(self, image):
        return _rotated(image, self.degrees)

    def __call__(self, image):
        return EdgePad(self.padding)(self.rotate(image))


def _rotated(image, degrees, padding=None):
    """Sample an expanded rotation, optionally only at the final square's pixels.

    Preserve rotate-to-uint8 THEN nearest selection, including virtual edge pad.
    Sampling a resized source instead would change the augmentation geometry.
    """
    angle = degrees % 360
    if angle % 90 == 0:
        rotated = np.rot90(image, int(angle // 90), axes=(1, 2))
        return rotated.copy() if padding is None else _square(rotated, padding)
    height, width = image.shape[1:]
    a, b = round(math.cos(-math.radians(angle)), 15), round(math.sin(-math.radians(angle)), 15)
    c, f = a * (-width / 2) + b * (-height / 2) + width / 2, -b * (-width / 2) + a * (-height / 2) + height / 2
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    xx, yy = zip(*[(a * x + b * y + c, -b * x + a * y + f) for x, y in corners], strict=True)
    nw, nh = math.ceil(max(xx)) - math.floor(min(xx)), math.ceil(max(yy)) - math.floor(min(yy))
    c += a * (-(nw - width) / 2) + b * (-(nh - height) / 2)
    f += -b * (-(nw - width) / 2) + a * (-(nh - height) / 2)
    x, y = np.arange(nw), np.arange(nh)
    if padding is not None:
        # Upscaling and edge padding repeat pixels: interpolate each only once.
        x, columns = np.unique(_nearest_indices(nw, padding), return_inverse=True)
        y, rows = np.unique(_nearest_indices(nh, padding), return_inverse=True)
    x, y = x[None, :] + 0.5, y[:, None] + 0.5
    sx, sy = a * x + b * y + c, -b * x + a * y + f
    outside = (sx < 0) | (sx >= width) | (sy < 0) | (sy >= height)
    sx, sy = sx - 0.5, sy - 0.5
    ix, iy = np.floor(sx).astype(np.intp), np.floor(sy).astype(np.intp)
    dx, dy = sx - ix, sy - iy
    x0, x1 = np.clip(ix, 0, width - 1), np.clip(ix + 1, 0, width - 1)
    y0, y1 = np.clip(iy, 0, height - 1), np.clip(iy + 1, 0, height - 1)
    top, bottom = image[:, y0, x0].astype(np.float64), image[:, y1, x0].astype(np.float64)
    top += (image[:, y0, x1] - top) * dx
    bottom += (image[:, y1, x1] - bottom) * dx
    top += (bottom - top) * dy
    result = top.astype(np.uint8)
    result[:, outside] = np.array([124, 116, 104], dtype=np.uint8)[:, None]
    return result if padding is None else result[:, rows[:, None], columns[None, :]]


@dataclass(frozen=True)
class SaltAndPepper:
    """Deterministic image-keyed noise; one black/white pixel mask shared by RGB."""

    proportion: float = 0.01
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.proportion <= 1:
            raise ValueError("Noise proportion must be in [0,1]")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Noise seed must be a nonnegative integer")

    def __call__(self, image):
        image = np.ascontiguousarray(image)
        fingerprint = int.from_bytes(hashlib.blake2b(image.data, digest_size=8).digest(), "little")
        rng = np.random.default_rng([self.seed, fingerprint])
        draws = rng.random(image.shape[1:])
        result = image.copy()
        result[:, draws < self.proportion / 2] = 0
        result[:, (draws >= self.proportion / 2) & (draws < self.proportion)] = 255
        return result


@dataclass(frozen=True)
class TTA:
    """Named finite transforms; custom callables receive isolated uint8 CHW copies."""

    transforms: tuple
    name: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if not self.transforms or not all(callable(view) for view in self.transforms):
            raise ValueError("TTA requires one or more callable transforms")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TTA name must be a nonempty string")


DEFAULT_TTA = "rotation30_pad25_3"
PROFILES = (
    "none",
    "rotation30_pad25_3",
    "wide_rotation_mixed_padding_5",
    "padded_scale",
    "hflip",
    "five_crop",
    "ten_crop",
    "d4",
    "light_noise",
)


def resolve_tta(value):
    if value is True:
        value = DEFAULT_TTA
    elif value is False:
        value = "none"
    if isinstance(value, TTA):
        return value
    if value not in PROFILES:
        raise ValueError(f"tta must be a TTA object or one of {PROFILES}")
    if value == "none":
        return None
    if value == "rotation30_pad25_3":
        views = (View(), RotatePad(-30), RotatePad(30))
    elif value == "wide_rotation_mixed_padding_5":
        views = (View(), RotatePad(-10, 0.15), RotatePad(10, 0.15), RotatePad(-30), RotatePad(30))
    elif value == "padded_scale":
        views = (View(), EdgePad(0.08), EdgePad(0.15))
    elif value == "hflip":
        views = (View(), View(hflip=True))
    elif value == "light_noise":
        views = (View(), SaltAndPepper(seed=0), SaltAndPepper(seed=1))
    elif value == "d4":
        views = tuple(View(quarter_turns=k, hflip=flip) for flip in (False, True) for k in range(4))
    else:
        # Original framing plus four corner crops covering 90% on each axis.
        boxes = ((0, 0, 1, 1), (0, 0, 0.9, 0.9), (0, 0.1, 0.9, 1), (0.1, 0, 1, 0.9), (0.1, 0.1, 1, 1))
        views = tuple(View(crop=box, hflip=flip) for flip in ((False, True) if value == "ten_crop" else (False,)) for box in boxes)
    return TTA(views, value)


def _prepare_view(image, transform, out=None, *, compact=False):
    prepare = prepare_uint8 if compact else preprocess
    # Exact built-in types are non-mutating; subclasses/custom callables retain isolation.
    if type(transform) is RotatePad:
        return prepare(_rotated(image, transform.degrees, transform.padding), out=out)
    if type(transform) is EdgePad:
        return prepare(image, out=out, padding=transform.fraction)
    return prepare(transform(image if type(transform) in (View, SaltAndPepper) else image.copy()), out=out)


def prepared_views(items, tta, pool=None, *, compact=False, decode=_rgb):
    """Decode once and lazily prepare views in recipe order."""

    def mapped(fn, values):
        return list(pool.map(fn, values)) if pool and len(values) > 1 else [fn(item) for item in values]

    decoded = mapped(decode, items)

    def batches():
        for transform in tta.transforms:
            output = np.empty((len(decoded), 3, RECIPE["crop_size"], RECIPE["crop_size"]), dtype=np.uint8 if compact else np.float32)

            def fill(index):
                _prepare_view(decoded[index], transform, out=output[index], compact=compact)

            mapped(fill, range(len(decoded)))
            yield output

    return batches()


def infer_augmented(runtime, items, tta, embeddings=False, pool=None):
    return infer_prepared(runtime, prepared_views(items, tta, pool), len(tta.transforms), embeddings)


def infer_prepared(runtime, views, view_count, embeddings=False):
    """Aggregate prepared views in recipe order, identically for streaming and prefetched inputs."""
    leaves, vectors = None, None
    for prepared in views:
        scores, embedding = runtime(prepared, embeddings)
        scores = scores.astype(np.float32) / np.float32(view_count)
        leaves = scores if leaves is None else leaves + scores
        if embeddings:
            embedding = embedding.astype(np.float32) / np.float32(view_count)
            vectors = embedding if vectors is None else vectors + embedding
    if embeddings:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).eps):
            raise RuntimeError("TTA produced an undefined mean embedding")
        vectors /= norms
    return leaves, vectors
