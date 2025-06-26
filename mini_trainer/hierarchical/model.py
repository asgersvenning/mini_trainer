from typing import Optional

import torch
from torch import nn as nn

from mini_trainer.classifier import Classifier


class HierarchicalClassifier(Classifier):
    def __init__(self, masks : Optional[list[torch.Tensor]]=None, **kwargs):
        super().__init__(**kwargs)

        # Store masks
        self.masks = masks or []
        [m.requires_grad_(False) for m in self.masks]

    def _apply(self, fn):
        self.masks = [fn(m) for m in self.masks]
        return super()._apply(fn)

    def forward(self, x):
        # Compute the normalized log probabilities for the leaf nodes (level 0)
        ys = [nn.functional.log_softmax(super().forward(x), dim = 1)]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(nn.functional.log_softmax(torch.logsumexp(ys[-1].unsqueeze(2) + mask, dim = 1), dim=1))
        return ys