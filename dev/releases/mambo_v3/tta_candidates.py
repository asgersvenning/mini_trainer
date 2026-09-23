"""Literature-informed, extent-preserving candidate policies for bounded qualification.

These use the public callable interface; they are not additional release defaults.
See docs/mambo-tta.md for sources, magnitudes and the distinction from policy tuning.
"""

import hashlib
from functools import partial

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from deployment.mambo_deploy import TTA, View

CANDIDATES = ("brightness", "contrast", "gamma", "gaussian_noise", "padded_scale", "mild_blur", "padded_rotation")


def photometric(image, kind, factor):
    if kind == "gamma":
        return np.rint((image.astype(np.float32) / 255) ** factor * 255).astype(np.uint8)
    pil = Image.fromarray(image.transpose(1, 2, 0))
    enhancer = ImageEnhance.Brightness if kind == "brightness" else ImageEnhance.Contrast
    return enhancer(pil).enhance(factor)


def gaussian(image, seed):
    fingerprint = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(image).data, digest_size=8).digest(), "little")
    rng = np.random.default_rng([seed, fingerprint])
    # Independent RGB sensor-like perturbations; sigma is 0.005 on [0,1].
    return np.rint(np.clip(image.astype(np.float32) + rng.normal(0, 0.005 * 255, image.shape), 0, 255)).astype(np.uint8)


def pad(image, fraction):
    height, width = image.shape[1:]
    y, x = max(1, int(np.ceil(height * fraction))), max(1, int(np.ceil(width * fraction)))
    return np.pad(image, ((0, 0), (y, y), (x, x)), mode="edge")


def blur(image, radius):
    return Image.fromarray(image.transpose(1, 2, 0)).filter(ImageFilter.GaussianBlur(radius))


def rotate(image, degrees):
    pil = Image.fromarray(image.transpose(1, 2, 0)).rotate(
        degrees, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(124, 116, 104)
    )
    # Keep the expanded canvas through the release recipe's center crop.
    return pad(np.asarray(pil).transpose(2, 0, 1), 0.08)


def candidate_policy(name):
    if name in ("brightness", "contrast", "gamma"):
        transforms = [partial(photometric, kind=name, factor=factor) for factor in (0.9, 1.1)]
    elif name == "gaussian_noise":
        transforms = [partial(gaussian, seed=seed) for seed in (0, 1)]
    elif name == "padded_scale":
        transforms = [partial(pad, fraction=fraction) for fraction in (0.08, 0.15)]
    elif name == "mild_blur":
        transforms = [partial(blur, radius=radius) for radius in (0.25, 0.5)]
    elif name == "padded_rotation":
        transforms = [partial(rotate, degrees=degrees) for degrees in (-10, 10)]
    else:
        raise ValueError(f"Unknown candidate {name}")
    return TTA((View(), *transforms), name=name)
