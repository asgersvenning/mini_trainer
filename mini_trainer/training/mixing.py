"""Opt-in batch regularization for training steps: timm-style random erasing and Mixup/CutMix.

Both act on the preprocessed (normalized) training batch, after the recipe's augmentation, as timm does (its loader
erases normalized tensors; its training loop mixes them). Evaluation never sees them.
"""

import math

import numpy as np
import torch

from mini_trainer.utils import get_rank

_TAG = 0x6D1C  # separates this stream from other seeded draws of the same run


def mixed_loss(loss, partner_loss, lam: float):
    """``lam * loss + (1 - lam) * partner_loss`` for a scalar loss or a list of per-rank losses.

    For cross-entropy (with or without label smoothing) this equals the loss against the soft targets
    ``lam * onehot(target) + (1 - lam) * onehot(partner)`` at every taxonomic rank (``soft_targets``).
    """
    if isinstance(loss, (list, tuple)):
        return [lam * a + (1 - lam) * b for a, b in zip(loss, partner_loss)]
    return lam * loss + (1 - lam) * partner_loss


def soft_targets(target: torch.Tensor, partner: torch.Tensor, lam: float, num_classes: int | list[int]) -> list[torch.Tensor]:
    """The mixed targets a mixed batch is trained on, one ``[N, C]`` tensor per rank (targets ``[N]`` or ``[N, R]``)."""
    if target.ndim == 1:
        target, partner = target[:, None], partner[:, None]
    sizes = [num_classes] if isinstance(num_classes, int) else list(num_classes)
    return [
        lam * torch.nn.functional.one_hot(target[:, r], n).float() + (1 - lam) * torch.nn.functional.one_hot(partner[:, r], n).float()
        for r, n in enumerate(sizes)
    ]


class BatchRegularizer:
    """Random erasing then Mixup/CutMix of a normalized training batch (each opt-in; both off is the identity).

    Random erasing (timm ``RandomErasing``, mode ``"pixel"``, one box): each image independently with probability
    ``erase_p`` gets one box of 2-33 % of its area, aspect ratio 0.3-3.3 (log-uniform), filled with standard-normal
    noise per pixel and channel.

    Mixup/CutMix (timm ``Mixup``, mode ``"batch"``): with probability ``mix_prob`` the batch is mixed with itself in
    reverse order, by CutMix (probability ``switch_prob``, lam ~ Beta(cutmix_alpha), corrected to the pasted box's
    area) or else by Mixup (lam ~ Beta(mixup_alpha)). An alpha of 0 disables that method. The loss is then
    ``lam * criterion(logits, target) + (1 - lam) * criterion(logits, target.flip(0))`` (``mixed_loss``), the soft
    targets of every rank. Each process mixes its own local batch.

    Draws come from a generator seeded with (``seed``, process rank, epoch) at ``set_epoch``, so an epoch replays
    after a resume and the global RNGs are untouched.
    """

    def __init__(
        self,
        erase_p: float = 0.0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        switch_prob: float = 0.5,
        seed: int | None = None,
    ):
        for name, value in (("erase_p", erase_p), ("mix_prob", mix_prob), ("switch_prob", switch_prob)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], not {value}.")
        if mixup_alpha < 0 or cutmix_alpha < 0:
            raise ValueError(f"Mixup and CutMix alphas must be non-negative, not {mixup_alpha} and {cutmix_alpha}.")
        self.erase_p = float(erase_p)
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.mix_prob = float(mix_prob)
        self.switch_prob = float(switch_prob)
        self.seed = 0 if seed is None else int(seed)
        self.set_epoch(0)

    @property
    def mixes(self) -> bool:
        return self.mix_prob > 0 and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)

    def set_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng([self.seed, get_rank(), int(epoch), _TAG])
        self._noise_seed = int(self.rng.integers(2**62))
        self._noise: dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._noise:
            self._noise[device] = torch.Generator(device=device).manual_seed(self._noise_seed + len(self._noise))
        return self._noise[device]

    def erase(self, x: torch.Tensor) -> torch.Tensor:
        """Erase one random box in each image with probability ``erase_p`` (in place on a copy)."""
        if self.erase_p <= 0:
            return x
        x = x.clone()
        n, c, height, width = x.shape
        area = height * width
        for i in np.flatnonzero(self.rng.random(n) < self.erase_p):
            for _ in range(10):  # timm's attempts
                target_area = self.rng.uniform(0.02, 1 / 3) * area
                aspect = math.exp(self.rng.uniform(math.log(0.3), math.log(1 / 0.3)))
                h = int(round(math.sqrt(target_area * aspect)))
                w = int(round(math.sqrt(target_area / aspect)))
                if w < width and h < height:
                    top, left = int(self.rng.integers(0, height - h + 1)), int(self.rng.integers(0, width - w + 1))
                    noise = torch.randn((c, h, w), generator=self._generator(x.device), device=x.device, dtype=torch.float32)
                    x[i, :, top : top + h, left : left + w] = noise.to(x.dtype)
                    break
        return x

    def mix(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Mix the batch with its reverse; returns the mixed batch and the weight ``lam`` of the original labels."""
        if not self.mixes or self.rng.random() >= self.mix_prob:
            return x, 1.0
        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            cutmix = bool(self.rng.random() < self.switch_prob)
        else:
            cutmix = self.cutmix_alpha > 0
        lam = float(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha) if cutmix else self.rng.beta(self.mixup_alpha, self.mixup_alpha))
        if not cutmix:
            return lam * x + (1 - lam) * x.flip(0), lam
        height, width = x.shape[-2:]
        ratio = math.sqrt(1 - lam)
        cut_h, cut_w = int(height * ratio), int(width * ratio)
        cy, cx = int(self.rng.integers(0, height)), int(self.rng.integers(0, width))
        top, bottom = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        left, right = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        x = x.clone()
        x[..., top:bottom, left:right] = x.flip(0)[..., top:bottom, left:right]
        return x, 1.0 - (bottom - top) * (right - left) / (height * width)  # timm's correct_lam

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Erase, then mix: returns the batch to train on and ``lam`` (1.0: not mixed)."""
        return self.mix(self.erase(x))

    def __repr__(self) -> str:
        return (
            f"BatchRegularizer(erase_p={self.erase_p}, mixup_alpha={self.mixup_alpha}, cutmix_alpha={self.cutmix_alpha}, "
            f"mix_prob={self.mix_prob}, switch_prob={self.switch_prob}, seed={self.seed})"
        )
