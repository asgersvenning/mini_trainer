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


class _BatchTransform(torch.nn.Module):
    """Applied to the whole batch with probability ``p``, with one draw of its parameters (fastai's ``RandTransform``).

    Accepts float images ``[N, C, H, W]`` or ``[C, H, W]``; meant to follow ``RandomAffineResize`` (values in
    ``[0, 1]``) and to precede normalization, as in lepinet's ``batch_tfms``.
    """

    def __init__(self, p: float):  # noqa: D107
        super().__init__()
        self.p = float(p)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not (self.p >= 1 or torch.rand(()).item() < self.p):
            return images
        single = images.ndim == 3
        x = self._transform(images.unsqueeze(0) if single else images)
        return x[0] if single else x

    def extra_repr(self) -> str:
        return f"p={self.p}"


class RandomMotionBlur(_BatchTransform):
    """Box blur along a random horizontal or vertical line of odd length 3 to ``max_kernel`` (lepinet's trap preset)."""

    def __init__(self, p: float = 0.25, max_kernel: int = 9):  # noqa: D107
        super().__init__(p)
        self.max_kernel = int(max_kernel)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(3, self.max_kernel + 1, ()).item()) | 1
        kernel = torch.zeros(1, 1, k, k, device=x.device, dtype=x.dtype)
        if torch.rand(()).item() < 0.5:
            kernel[..., k // 2, :] = 1.0 / k
        else:
            kernel[..., :, k // 2] = 1.0 / k
        channels = x.shape[1]
        return torch.nn.functional.conv2d(x, kernel.expand(channels, 1, k, k), padding=k // 2, groups=channels)


class RandomLowLight(_BatchTransform):
    """Darken with a gamma in ``[1, max_gamma]`` and add Gaussian noise of std up to ``max_noise`` (night frames)."""

    def __init__(self, p: float = 0.25, max_gamma: float = 2.2, max_noise: float = 0.05):  # noqa: D107
        super().__init__(p)
        self.max_gamma, self.max_noise = float(max_gamma), float(max_noise)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + torch.rand(()).item() * (self.max_gamma - 1.0)
        sigma = torch.rand(()).item() * self.max_noise
        lo, hi = x.min(), x.max()
        unit = (x - lo) / (hi - lo + 1e-6)
        out = unit.clamp_min(0).pow(gamma) + torch.randn_like(x) * sigma
        return out.clamp(0, 1) * (hi - lo) + lo


class RandomJPEGish(_BatchTransform):
    """Quantize intensities to ``min_levels``-63 levels, a GPU stand-in for heavy JPEG compression."""

    def __init__(self, p: float = 0.2, min_levels: int = 12):  # noqa: D107
        super().__init__(p)
        self.min_levels = int(min_levels)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        levels = int(torch.randint(self.min_levels, 64, ()).item())
        lo, hi = x.min(), x.max()
        unit = (x - lo) / (hi - lo + 1e-6)
        return (unit * levels).round() / levels * (hi - lo) + lo


class RandomRotatePad(torch.nn.Module):
    """Per-image rotation on an expanded canvas, then zoom-out by edge padding, resampled to the input size.

    The training counterpart of ``mambo_deploy``'s ``RotatePad``/``EdgePad`` test-time views: each image is rotated
    by a uniform angle in ``[-max_degrees, max_degrees]`` on a canvas expanded to hold the whole rotated image (the
    corners filled with ``fill``, an RGB colour in 0-255), each canvas edge is then extended by a uniform fraction in
    ``[0, max_pad]`` of its axis by repeating the canvas border, and the result is resampled (bilinear, one
    ``grid_sample``) back to the input's height and width, so the object shrinks. Applied with probability ``p``
    per image. Accepts uint8 (scaled by 1/255) or float images in ``[0, 1]``, ``[N, C, H, W]`` or ``[C, H, W]``,
    and returns float32; meant to precede ``RandomAffineResize``.
    """

    def __init__(self, max_degrees: float = 45.0, max_pad: float = 0.25, fill: tuple[int, int, int] = (124, 116, 104), p: float = 1.0):  # noqa: D107
        super().__init__()
        if max_degrees < 0 or max_pad < 0 or not 0 <= p <= 1:
            raise ValueError("max_degrees and max_pad must be nonnegative and p in [0, 1].")
        self.max_degrees, self.max_pad, self.fill, self.p = float(max_degrees), float(max_pad), tuple(fill), float(p)

    def grid(self, degrees: torch.Tensor, pad: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Sampling grid ``[N, H, W, 2]`` (``grid_sample``, ``align_corners=False``) for the given angles and pads."""
        device = degrees.device
        theta = degrees * math.pi / 180
        cos, sin = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]
        canvas_w = width * cos.abs() + height * sin.abs()
        canvas_h = width * sin.abs() + height * cos.abs()
        scale = 1 + 2 * pad[:, None, None]
        # Output pixel centres in centred pixel units of the padded canvas, clamped to the centres of the canvas's
        # border pixels (edge padding repeats them; clamping to the outer edge would blend in the fill colour).
        u = (torch.arange(width, device=device) + 0.5) / width * 2 - 1
        v = (torch.arange(height, device=device) + 0.5) / height * 2 - 1
        x = torch.maximum(torch.minimum(u[None, None, :] * canvas_w * scale / 2, canvas_w / 2 - 0.5), 0.5 - canvas_w / 2)
        y = torch.maximum(torch.minimum(v[None, :, None] * canvas_h * scale / 2, canvas_h / 2 - 0.5), 0.5 - canvas_h / 2)
        # Inverse of PIL's counter-clockwise rotation (image y axis points down), back to normalised source coordinates.
        src_x = (x * cos - y * sin) / (width / 2)
        src_y = (x * sin + y * cos) / (height / 2)
        return torch.stack([src_x, src_y], dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        single = images.ndim == 3
        x = images.unsqueeze(0) if single else images
        x = x.float().div(255) if x.dtype == torch.uint8 else x.float()
        n, channels, height, width = x.shape
        keep = torch.ones(n, device=x.device) if self.p >= 1 else torch.empty(n, device=x.device).bernoulli_(self.p)
        degrees = torch.empty(n, device=x.device).uniform_(-self.max_degrees, self.max_degrees) * keep
        pad = torch.empty(n, device=x.device).uniform_(0, self.max_pad) * keep
        fill = torch.tensor(self.fill, device=x.device, dtype=x.dtype)[:channels].view(1, -1, 1, 1) / 255
        grid = self.grid(degrees, pad, height, width)
        # Outside the source image (inside the rotated canvas) is the fill colour: sample (x - fill) with zero padding.
        out = torch.nn.functional.grid_sample(x - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + fill
        return out[0] if single else out

    def extra_repr(self) -> str:
        return f"max_degrees={self.max_degrees}, max_pad={self.max_pad}, fill={self.fill}, p={self.p}"


def sample_augmentation(name: str | None) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """Return an opt-in per-sample transform for training reads (uint8 ``[C, H, W]``), run in loader workers.

    ``"trivialwide"`` is torchvision's ``TrivialAugmentWide``, one independent draw per image. ``None`` returns
    ``None``. Workers apply it to each decoded training image, so it costs CPU rather than training-step time.
    """
    if not name:
        return None
    if name == "trivialwide":
        from torchvision.transforms import v2

        return v2.TrivialAugmentWide()
    raise ValueError(f"Unknown sample augmentation {name!r}; known: ['trivialwide']")


INPUT_AUGMENTATIONS: dict[str, Callable[[], list[torch.nn.Module]]] = {
    "geometric": lambda: [RandomRotatePad()],
}


def split_domain_augmentation(spec: str | None) -> tuple[list[torch.nn.Module], list[torch.nn.Module]]:
    """Resolve a ``+``-joined augmentation spec into transforms before and after the recipe's affine resize.

    ``"geometric"`` (``RandomRotatePad``) acts on the read images, before ``RandomAffineResize``; the camera-trap
    presets of ``domain_augmentation`` follow it. For example ``"geometric+trap"``. ``None`` or ``""`` returns no
    transforms. (TrivialAugmentWide runs per sample in the loader workers: ``sample_augmentation``.)
    """
    before: list[torch.nn.Module] = []
    after: list[torch.nn.Module] = []
    for name in (spec or "").split("+"):
        if not name:
            continue
        if name in INPUT_AUGMENTATIONS:
            before += INPUT_AUGMENTATIONS[name]()
        else:
            after += domain_augmentation(name)
    return before, after


def domain_augmentation(preset: str | None) -> list[torch.nn.Module]:
    """Return lepinet's opt-in camera-trap transforms by preset name; ``None`` returns none.

    ``"trap"`` is lepinet's ``domain_aug: trap`` (motion blur 0.25, low light 0.25, JPEG-like 0.2, in that
    order); ``"blur"`` and ``"lowlight"`` are its single-transform ablations at p 0.35.
    """
    presets = {
        "trap": lambda: [RandomMotionBlur(p=0.25), RandomLowLight(p=0.25), RandomJPEGish(p=0.2)],
        "blur": lambda: [RandomMotionBlur(p=0.35)],
        "lowlight": lambda: [RandomLowLight(p=0.35)],
    }
    if not preset:
        return []
    if preset not in presets:
        raise ValueError(f"Unknown domain augmentation {preset!r}; known: {sorted(presets) + sorted(INPUT_AUGMENTATIONS)}")
    return presets[preset]()
