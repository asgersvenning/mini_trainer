import torch
from torch import nn

from .classifier import last_layer_weights
from .generic import cosine_to_zscore


@torch.no_grad()
def class_similarity(model: nn.Module, cdf: bool = False):
    W = last_layer_weights(model)
    W = W.detach().clone().float()
    WN = W.norm(2, 1, True)
    Z = cosine_to_zscore((W @ W.T) / (WN @ WN.T), W.shape[1])
    if cdf:
        Z = torch.distributions.Normal(0, 1).cdf(Z).fill_diagonal_(1.0)
    return Z


@torch.no_grad()
def class_distance(model: nn.Module, eps: float | None = None):
    sim = class_similarity(model, cdf=True)
    if eps is None:
        eps = torch.finfo(sim.dtype).eps
    return (-sim.clamp_(min=eps, max=1.0).log_()).clamp_min_(0)
