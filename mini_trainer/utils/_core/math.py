import math

import torch


def float_signif_decimal(value: float, digits: int = 3):
    """Compute the number of decimals needed for rounding to the given decimals."""
    if value == 0 or not math.isfinite(value):
        return 0
    min_b10 = math.log10(abs(value))
    if abs(min_b10 - abs(min_b10)) < 1e-1:
        min_b10 = round(min_b10)
    else:
        min_b10 = math.floor(min_b10)
    return -min(-1, min_b10 - digits + 1)


def decimals(value: float, tol: int = 6):
    """Heuristic to determine the number of significant digits in a float."""
    fv = f"{value}"
    if "." not in fv:
        return 0
    dec = fv.index(".") + 1
    for d, i in enumerate(range(dec, len(fv))):
        trunc_value = float(fv[:i])
        if abs(value - trunc_value) < 10 ** (-(i + tol)):
            return d
    return d + 1


def cosine_schedule_with_warmup(total: int, warmup: int, start: float, end: float):
    """Factory for creating a function that parametrizes a cosine with warmup schedule shape function."""

    def _shape_fn(step: int):
        if warmup > 0 and step < warmup:
            # linear warm-up from start_factor -> 1.0
            return start + (1.0 - start) * step / warmup
        # cosine decay from 1.0 -> min_factor
        progress = (step - warmup) / max(1, total - warmup)
        return end + 0.5 * (1.0 - end) * (1.0 + math.cos(math.pi * progress))

    return _shape_fn


def cosine_to_zscore(cosine: torch.Tensor, ndim: int, eps: float = 1e-7):
    r"""Converts a cosine (or inner product) between random unit vectors in D dimensions to z-score.

            ::math::`Z(x) = \sqrt(D-2) * (cos^{-1}(-x) - \frac{\pi}{2})`

    This function is a *very* good approximation for transforming the distribution
    given by the inner product between random unit vectors in D dimensions
    to the standard normal distribution - i.e. if the embeddings and weights are random
    then the output logits here will follow a normal distribution.
    """
    z_var: float = 1 / (float(ndim) - 2) ** 0.5
    z_mu = torch.pi / 2
    z_rel = torch.acos(-cosine.clamp(-1 + eps, 1 - eps))
    return (z_rel - z_mu) / z_var


def kl_distill(
    logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    target_logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    T: float = 1.0,
):
    """Compute KL-divergence between online model-logits and target logits."""
    if isinstance(logits, (list, tuple)) or isinstance(target_logits, (list, tuple)):
        return torch.stack([kl_distill(lg, target_lg, T=T) for lg, target_lg in zip(logits, target_logits)]).mean()
    orig_dtype = logits.dtype
    logits = (logits / T).float().log_softmax(-1)
    target_logits = (target_logits / T).float().log_softmax(-1).detach()
    return (torch.nn.functional.kl_div(logits, target_logits, log_target=True, reduction="batchmean") * T**2).to(dtype=orig_dtype)
