"""Square-crop reading and affine resampling shared by data loaders and model preprocessing.

Both reproduce fastai's image pipeline exactly, so a recipe ported from fastai receives byte-identical
inputs: ``read_square_crop`` matches ``Resize(size, method="crop")`` and ``affine_resample`` matches
the ``grid_sample`` step of ``AffineCoordTfm`` (including its area pre-filter rule).
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def square_crop_box(width: int, height: int, size: int, position: tuple[float, float] = (0.5, 0.5)) -> tuple[int, int, int, int]:
    """Return the ``(left, top, right, bottom)`` box of the largest square, placed at ``position``.

    ``position`` is the relative offset along each axis; ``(0.5, 0.5)`` centres the square. The side is
    ``int(m * size)`` with ``m = min(width, height) / size``, as fastai computes it, so floating-point
    rounding can make it one pixel shorter than the short side.
    """
    m = width / size if (width / size) < (height / size) else height / size
    side = int(m * size)
    left, top = int(position[0] * (width - side)), int(position[1] * (height - side))
    return left, top, left + side, top + side


def read_square_crop(path: str, size: int, position: tuple[float, float] | None = None) -> torch.Tensor:
    """Read an RGB image, crop its largest square and resize it to ``size`` with PIL bilinear filtering.

    ``position=None`` draws a uniform random offset per axis from Python's ``random`` module (seeded per
    DataLoader worker). PIL's bilinear filter antialiases when downscaling. EXIF orientation is not
    applied. Returns a contiguous ``uint8`` tensor of shape ``[3, size, size]``.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
    if position is None:
        position = (random.random(), random.random())
    image = image.crop(square_crop_box(*image.size, size, position)).resize((size, size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).contiguous()


def affine_resample(
    images: torch.Tensor,
    theta: torch.Tensor | None,
    size: int,
    mode: str = "bilinear",
    padding_mode: str = "reflection",
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample float ``images`` (``[N, C, H, W]``) on an affine grid of ``size`` x ``size``.

    ``theta`` holds ``[N, 2, 3]`` matrices in normalised coordinates, or ``None`` for the identity (a
    plain resize). Like fastai, bilinear sampling first area-downsamples the input when
    ``d = min(H / size, W / 2) / 2`` exceeds both 1 and the grid's zoom; fastai compares the input height
    with the grid's width dimension and the input width with its coordinate dimension, reproduced here.
    """
    n, c = images.shape[:2]
    if theta is None:
        theta = torch.eye(2, 3, device=images.device, dtype=images.dtype).expand(n, 2, 3)
    grid = F.affine_grid(theta, [n, c, size, size], align_corners=align_corners)
    if mode == "bilinear":
        d = min(images.shape[-2] / grid.shape[-2], images.shape[-1] / grid.shape[-1]) / 2
        if d > 1 and d > 2 / (grid.max() - grid.min()).item():
            images = F.interpolate(images, scale_factor=1 / d, mode="area", recompute_scale_factor=True)
    return F.grid_sample(images, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
