import torch
from torch import nn

from mini_trainer.utils import cosine_to_zscore

from .classifier import classification_module


def _class_similarity(W: torch.Tensor, cdf: bool = True) -> torch.Tensor:
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
def class_distance(model: nn.Module, eps: float | None = None): 
    return [
        (
            (-sim.clamp_(min=torch.finfo(sim.dtype).eps if eps is None else eps, max=1.0).log_())
            .clamp_min_(0)
            .fill_diagonal_(0.0)
        ) 
        for sim in class_similarity(model, cdf=True)
    ]
