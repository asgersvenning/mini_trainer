"""WARNING: This module is scheduled for deprecation and will be removed in the future!"""

import math
from collections import Counter


def prior_logit_adjustment(counts: list[int], C: float = 1.0, eps: float = 1e-7) -> list[float]:
    """Return mean-centered biases from b_i = -C * log(K * p_i).

    Frequencies are clamped below by eps before taking logs.
    Reference: https://arxiv.org/abs/2007.07314
    """
    total_samples = sum(counts)
    ncls = len(counts)

    biases = [-C * math.log(ncls * max(c / total_samples, eps)) for c in counts]

    mean_bias = sum(biases) / ncls
    centered_biases = [b - mean_bias for b in biases]

    return centered_biases


def prior_ldam_shift(counts: list[int], C: float = 1.0, eps: float = 1e-7) -> list[float]:
    """Return mean-centered biases from C * (N_i**(-1/4) - N_max**(-1/4)).

    Counts are clamped below by eps in the first term.
    Reference: https://arxiv.org/abs/1906.07413
    """
    n_max = max(counts)
    biases = [C * ((max(c, eps) ** -0.25) - (n_max**-0.25)) for c in counts]

    mean_bias = sum(biases) / len(biases)
    centered_biases = [b - mean_bias for b in biases]

    return centered_biases


def prior_scratch(counts: list[int], **kwargs):
    """Return negative standardized log-counts (sample standard deviation).

    Experimental: zero counts map to log(1), and equal log-counts or a single
    class cause division by zero.
    """
    prior = [math.log(c) if c > 0 else 0 for c in counts]
    pmu = sum(prior) / len(prior)
    pvar = sum([(p - pmu) ** 2 for p in prior]) / (len(prior) - 1)
    psig = pvar**0.5
    retval = [-(p - pmu) / psig for p in prior]
    return retval


def get_prior_method(method: str):
    match method.lower().strip():
        case "adjust":
            return prior_logit_adjustment
        case "ldam":
            return prior_ldam_shift
        case "custom":
            return prior_scratch
        case _:
            raise NotImplementedError(
                f'Class frequency prior implementations currently include: "adjust", "ldam", and "custom", not: {method}'
            )


def prior_from_labels(labels: list[int | list[int]], cls2idx: dict, method: str = "adjust", **kwargs):
    if isinstance(labels[0], (list, tuple)):
        labels = [lab[0] for lab in labels]
        ncls = len(cls2idx["0"])
    else:
        ncls = len(cls2idx)
    counts = Counter(labels)
    counts = [counts.get(i, 0) for i in range(ncls)]
    method = get_prior_method(method)
    return method(counts, **kwargs)
