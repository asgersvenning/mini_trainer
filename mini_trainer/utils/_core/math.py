import math


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
