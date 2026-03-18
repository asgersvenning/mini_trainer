import math
from collections import Counter

import torch


def cosine_to_zscore(cosine : torch.Tensor, ndim : int):
    r"""Converts a cosine (or inner product) between random unit vectors in D dimensions to z-score.
    
            ::math::`Z(x) = \sqrt(D-2) * (cos^{-1}(-x) - \frac{\pi}{2})`
    
    This function is a *very* good approximation for transforming the distribution
    given by the inner product between random unit vectors in D dimensions 
    to the standard normal distribution - i.e. if the embeddings and weights are random
    then the output logits here will follow a normal distribution.
    """
    z_var : float = 1 / (float(ndim) - 2)**0.5
    z_mu = torch.pi / 2
    z_rel = torch.acos(-cosine.clamp(-1 + 1e-7, 1 - 1e-7))
    return (z_rel - z_mu) / z_var


def prior_logit_adjustment(counts: list[int], C: float = 1.0, eps: float = 1e-7) -> list[float]:
    """Computes dimension-independent biases based on Bayesian Logit Adjustment.
    Formula: b_i = -C * log(K * p_i)

    Ref: https://arxiv.org/abs/2007.07314
    """
    total_samples = sum(counts)
    ncls = len(counts)
    
    biases = [-C * math.log(ncls * max(c / total_samples, eps)) for c in counts]
        
    # Optional but recommended: Center the biases so their mean is 0. 
    # This keeps the initial Softmax logits numerically stable.
    mean_bias = sum(biases) / ncls
    centered_biases = [b - mean_bias for b in biases]
    
    return centered_biases


def prior_ldam_shift(counts: list[int], C: float = 1.0, eps: float = 1e-7) -> list[float]:
    """Computes dimension-independent biases using LDAM generalization bounds.
    Formula: b_i = C * (N_i^{-1/4} - N_max^{-1/4})

    Ref: https://arxiv.org/abs/1906.07413
    """
    n_max = max(counts)
    biases = [C * ((max(c, eps) ** -0.25) - (n_max ** -0.25)) for c in counts]
        
    # Again, centering helps network initialization stability
    mean_bias = sum(biases) / len(biases)
    centered_biases = [b - mean_bias for b in biases]
    
    return centered_biases


def prior_scratch(counts : list[int], **kwargs):
    """Computes dimension-independent biases using Z-scored negative log-frequencies.
    Formula: b_i = -(log(N_i) - mu) / sigma
    
    Note: This is an experimental ad-hoc method. It standardizes the log-counts 
    to have a mean of 0 and a variance of 1. While it correctly penalizes majority 
    classes, it can become numerically unstable if the dataset is perfectly balanced 
    (sigma approaches 0) and maps zero-counts to the same value as singletons (since log(1) == 0).
    """ 
    prior = [math.log(c) if c > 0 else 0 for c in counts]
    pmu = sum(prior) / len(prior)
    pvar = sum([(p - pmu)**2 for p in prior]) / (len(prior) - 1)
    psig = pvar ** 0.5
    retval = [-(p - pmu) / psig for p in prior]
    return retval


def get_prior_method(method : str):
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


def prior_from_labels(labels : list[int | list[int]], cls2idx : dict, method : str="adjust", **kwargs):
    if isinstance(labels[0], (list, tuple)):
        labels = [l[0] for l in labels]
        ncls = len(cls2idx["0"])
    else:
        ncls = len(cls2idx)
    counts = Counter(labels)
    counts = [counts.get(i, 0) for i in range(ncls)]
    method = get_prior_method(method)
    return method(counts, **kwargs)
