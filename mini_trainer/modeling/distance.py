import torch
from torch import nn

from mini_trainer.utils import cosine_to_zscore

from .classifier import classification_module


def _class_similarity(W: torch.Tensor, cdf: bool = True) -> torch.Tensor:
    if getattr(W, "_is_quantized_training", False):
        W = W.dequantize()
    W = W.detach().clone().float()
    WN = W.norm(2, 1, True)
    Z = cosine_to_zscore((W @ W.T) / (WN @ WN.T), W.shape[1])
    Z = (Z + Z.T) / 2
    if cdf:
        Z = torch.distributions.Normal(0, 1).cdf(Z).fill_diagonal_(1.0)
    return Z


@torch.no_grad()
def class_similarity(model: nn.Module, cdf: bool = False):
    W = classification_module(model).last_layer_weights
    if isinstance(W, torch.Tensor):
        W = [W]
    return [_class_similarity(w, cdf=cdf) for w in W]


@torch.no_grad()
def class_log_similarity(model: nn.Module, complement: bool = False):
    """Return float32 log probabilities without materializing the Normal CDF.

    Uses the existing prototype z-score. By default returns log Phi(z), with
    diagonal 0. With complement=True returns log Phi(-z) = log(1 - Phi(z)),
    with diagonal -inf (the exact zero self-dissimilarity). The complementary
    log representation retains strongly aligned pairs even when Phi(z) rounds
    to 1. No probability floor is applied. This is a representation for analysis
    and plotting, not a replacement distance to pass directly into linkage.

    Like the existing diagnostics, returns one matrix per prototype level and
    does not propagate gradients. Disable enclosing autocast for the diagnostic
    computation; no float64 promotion is needed.
    """
    weights = classification_module(model).last_layer_weights
    if isinstance(weights, torch.Tensor):
        weights = [weights]
    results = []
    for weight in weights:
        with torch.autocast(device_type=weight.device.type, enabled=False):
            z = _class_similarity(weight, cdf=False)
            log_probability = torch.special.log_ndtr(-z if complement else z)
            log_probability.fill_diagonal_(-torch.inf if complement else 0.0)
        results.append(log_probability)
    return results


@torch.no_grad()
def class_distance(model: nn.Module, eps: float | None = None):
    return [
        ((-sim.clamp_(min=torch.finfo(sim.dtype).eps if eps is None else eps, max=1.0).log_()).clamp_min_(0).fill_diagonal_(0.0))
        for sim in class_similarity(model, cdf=True)
    ]
