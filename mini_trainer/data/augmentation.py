import math
import os
import warnings
from collections.abc import Callable
from random import sample

import torch
from matplotlib import pyplot as plt

from mini_trainer.utils import affine_resample, make_convert_dtype


def debug_augmentation(
    augmentation: Callable[[torch.Tensor], torch.Tensor],
    dataset: torch.utils.data.Dataset,
    output_dir: str | None = None,
    strict: bool = True,
):
    """Helper to debug augmentation pipeline."""
    if output_dir is None or int(os.environ.get("RANK", 0)) > 0:
        return
    convert2fp32 = make_convert_dtype(torch.float32)
    fig = None
    try:
        n = min(3, len(dataset))
        fig = plt.figure(figsize=(10, 5))
        axs = fig.subplots(n, 3, squeeze=False)

        def prep_for_imshow(img_tensor):
            img_tensor = convert2fp32(img_tensor.permute(1, 2, 0))
            min_v = img_tensor.min()
            max_v = img_tensor.max()
            if max_v > min_v:
                return (img_tensor - min_v) / (max_v - min_v)
            return torch.clamp(img_tensor, 0.0, 1.0)

        for j, i in enumerate(sample(range(len(dataset)), n)):
            example_image: torch.Tensor = dataset[i][0].clone().cpu()

            axs[j, 0].imshow(prep_for_imshow(example_image))
            axs[j, 1].imshow(prep_for_imshow(augmentation(example_image)))
            axs[j, 2].imshow(prep_for_imshow(augmentation(example_image)))
            for ax in axs[j, :]:
                ax.axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "example_augmentation.png"))
    except Exception as e:
        e_msg = (
            "Error while attempting to create debug augmentation image."
            "Perhaps the supplied dataloader doesn't return items (image, label) in the expected format."
        )
        e.add_note(e_msg)
        if strict:
            raise
        warnings.warn(e_msg, UserWarning)
        return False
    finally:
        if fig is not None:
            plt.close(fig)
    return True


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def salt_and_pepper(img: torch.Tensor, proportion: tuple[float, float] = (0, 0.5), probability: float = 1):
    """Functional salt and pepper augmentation implementation."""
    apply_aug = torch.rand([], device=img.device) < probability
    p = torch.rand([], device=img.device) * (proportion[1] - proportion[0]) + proportion[0]
    if len(img.shape) <= 3:
        r = torch.rand((img.shape[-2], img.shape[-1]), dtype=torch.float16, device=img.device)
    else:
        r = torch.rand((img.shape[0], 1, img.shape[-2], img.shape[-1]), dtype=torch.float16, device=img.device)
    max_val = 1 if img.is_floating_point() else torch.iinfo(img.dtype).max
    m = r < p
    s = r < (p / 2)
    noisy_img = img.clone()
    noisy_img.masked_fill_(m & s, 0)
    noisy_img.masked_fill_(m & ~s, max_val)
    return torch.where(apply_aug, noisy_img, img)


_DIHEDRAL = torch.tensor(
    [
        [1, -1, 1, -1, -1, 1, 1, -1],  # x sign
        [1, 1, -1, 1, -1, -1, 1, -1],  # y sign
        [1, 1, 1, 0, 1, 0, 0, 0],  # keep axes
        [0, 0, 0, 1, 0, 1, 1, 1],  # swap axes
    ],
    dtype=torch.float32,
)


class RandomAffineResize(torch.nn.Module):
    """Per-image flips, rotation and zoom composed into one affine and resampled to ``size``.

    Reproduces the geometric part of fastai's ``aug_transforms(size=size, ...)`` with
    ``batch=False``: every image draws its own transform, and the affine and the resize share one
    bilinear ``grid_sample`` with reflection padding (``mini_trainer.utils.affine_resample``).
    ``dihedral=True`` draws one of the 8 dihedral elements (index 0 is the identity) with
    probability ``flip_p``, otherwise a horizontal flip; rotations are uniform in
    ``[-max_rotate, max_rotate]`` degrees and zooms in ``[min_zoom, max_zoom]`` around a uniform
    centre, each applied with probability ``affine_p``. Accepts uint8 (scaled by 1/255) or float
    images, ``[N, C, H, W]`` or ``[C, H, W]``, and returns float32.
    """

    def __init__(
        self,
        size: int,
        dihedral: bool = True,
        flip_p: float = 0.5,
        max_rotate: float = 10.0,
        min_zoom: float = 1.0,
        max_zoom: float = 1.1,
        affine_p: float = 0.75,
        padding_mode: str = "reflection",
        align_corners: bool = True,
    ):  # noqa: D107
        super().__init__()
        if min_zoom > max_zoom:
            raise ValueError("min_zoom must not exceed max_zoom.")
        self.size = int(size)
        self.dihedral, self.flip_p = dihedral, float(flip_p)
        self.max_rotate, self.min_zoom, self.max_zoom, self.affine_p = float(max_rotate), float(min_zoom), float(max_zoom), float(affine_p)
        self.padding_mode, self.align_corners = padding_mode, align_corners

    @staticmethod
    def _keep(n: int, p: float, device: torch.device) -> torch.Tensor:
        return torch.ones(n, device=device) if p >= 1 else torch.empty(n, device=device).bernoulli_(p)

    def matrices(self, n: int, device: torch.device) -> torch.Tensor:
        """Draw ``n`` composed ``[3, 3]`` affine matrices: flip, then rotation, then zoom (fastai's order)."""
        zero, one = torch.zeros(n, device=device), torch.ones(n, device=device)

        def affine(a, b, c, d, e, f):
            return torch.stack([torch.stack([a, b, c], 1), torch.stack([d, e, f], 1), torch.stack([zero, zero, one], 1)], 1)

        matrices = torch.eye(3, device=device).repeat(n, 1, 1)
        if self.flip_p > 0:
            if self.dihedral:
                index = (torch.randint(0, 8, (n,), device=device) * self._keep(n, self.flip_p, device)).long()
                xs, ys, keep, swap = _DIHEDRAL.to(device)[:, index]
                matrices = matrices @ affine(xs * keep, xs * swap, zero, ys * swap, ys * keep, zero)
            else:
                sign = one - 2 * self._keep(n, self.flip_p, device)
                matrices = matrices @ affine(sign, zero, zero, zero, one, zero)
        if self.max_rotate > 0:
            degrees = torch.empty(n, device=device).uniform_(-self.max_rotate, self.max_rotate) * self._keep(n, self.affine_p, device)
            cos, sin = torch.cos(degrees * math.pi / 180), torch.sin(degrees * math.pi / 180)
            matrices = matrices @ affine(cos, sin, zero, -sin, cos, zero)
        if self.min_zoom < 1 or self.max_zoom > 1:
            zoom = torch.empty(n, device=device).uniform_(self.min_zoom, self.max_zoom)
            scale = 1 / ((zoom - 1) * self._keep(n, self.affine_p, device) + 1)
            col, row = torch.rand(n, device=device), torch.rand(n, device=device)
            matrices = matrices @ affine(scale, zero, (1 - scale) * (2 * col - 1), zero, scale, (1 - scale) * (2 * row - 1))
        return matrices

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        single = images.ndim == 3
        x = images.unsqueeze(0) if single else images
        x = x.float().div(255) if x.dtype == torch.uint8 else x.float()
        theta = self.matrices(len(x), x.device)
        height, width = x.shape[-2:]
        theta[:, 0, 1] *= height / width
        theta[:, 1, 0] *= width / height
        x = affine_resample(x, theta[:, :2], self.size, padding_mode=self.padding_mode, align_corners=self.align_corners)
        return x[0] if single else x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size}, dihedral={self.dihedral}, flip_p={self.flip_p}, max_rotate={self.max_rotate}, "
            f"zoom=({self.min_zoom}, {self.max_zoom}), affine_p={self.affine_p}, padding_mode={self.padding_mode!r})"
        )


class SaltAndPepper(torch.nn.Module):
    """Salt and pepper augmentation module."""

    def __init__(self, proportion: tuple[float, float] = (0.05, 0.33), probability: float = 0.5):  # noqa: D107
        super().__init__()
        self.proportion = proportion
        self.probability = probability

    def forward(self, img):
        return salt_and_pepper(img, proportion=self.proportion, probability=self.probability)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(proportion={self.proportion}, probability={self.probability})"
