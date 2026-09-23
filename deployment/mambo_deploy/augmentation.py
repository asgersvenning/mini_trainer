"""Outer, runtime-independent TTA over decoded CHW images."""

import hashlib
from dataclasses import dataclass
from functools import partial

import numpy as np

from .preprocessing import _rgb, preprocess


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
    """Named finite transforms; each callable receives its own uint8 CHW image copy."""

    transforms: tuple
    name: str = "custom"

    def __post_init__(self):
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if not self.transforms or not all(callable(view) for view in self.transforms):
            raise ValueError("TTA requires one or more callable transforms")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("TTA name must be a nonempty string")


PROFILES = ("none", "hflip", "five_crop", "ten_crop", "d4", "light_noise")


def resolve_tta(value):
    if isinstance(value, TTA):
        return value
    if value not in PROFILES:
        raise ValueError(f"tta must be a TTA object or one of {PROFILES}")
    if value == "none":
        return None
    if value == "hflip":
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


def _prepare_view(image, transform):
    return preprocess(transform(image.copy()))


def infer_augmented(runtime, items, tta, embeddings=False, pool=None):
    """Generate a view, run ordinary preprocessing/inference, then aggregate leaves."""

    def mapped(fn, values):
        return list(pool.map(fn, values)) if pool and len(values) > 1 else [fn(item) for item in values]

    decoded = mapped(_rgb, items)
    leaves, vectors = None, None
    for transform in tta.transforms:
        prepared = np.stack(mapped(partial(_prepare_view, transform=transform), decoded))
        scores, embedding = runtime(prepared, embeddings)
        scores = scores.astype(np.float32) / np.float32(len(tta.transforms))
        leaves = scores if leaves is None else leaves + scores
        if embeddings:
            embedding = embedding.astype(np.float32) / np.float32(len(tta.transforms))
            vectors = embedding if vectors is None else vectors + embedding
    if embeddings:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).eps):
            raise RuntimeError("TTA produced an undefined mean embedding")
        vectors /= norms
    return leaves, vectors
