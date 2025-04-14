## REFERENCE IMPLEMENTATION
# loss_weights = [species_weight, genus_weight, family_weight]
# loss_weights = [len(loss_weights) * i / sum(loss_weights) for i in loss_weights]

# class_counts = full_dataset.class_counts
# class_weights = [cc.to(dtype) for cc in class_counts]
# class_weights = [1 / (w + w.mean()) for w in class_weights]
# class_weights = [w / w.mean() for w in class_weights]
# # class_weights = [torch.ones_like(i) for i in class_weights] # Uncomment this line to disable class weighting
# for i in class_weights:
#     i.requires_grad = False
# loss_fn = [
#     nn.CrossEntropyLoss(
#         weight=class_weights[i], 
#         reduction="none", 
#         # label_smoothing=1/class_weights[i].shape[0]
#     ) for i in range(N_LEVELS)
# ]
# item_weights = [class_weights[i][labels[i]] for i in range(N_LEVELS)]
# [(loss_fn[i](pred[i], labels[i].to(device)) * item_weights[i]).mean() for i in range(N_LEVELS)]

# Imports
from typing import List, Union
import torch
from torch import nn
from torch._prims_common import DeviceLikeType
from torch.types import _dtype

# Class based implemtation
class MultiLevelWeightedCrossEntropyLoss:
    def __init__(
            self, 
            weights : Union[List[Union[float, int]], torch.Tensor], 
            class_counts : List[torch.Tensor], 
            device : DeviceLikeType, 
            dtype : _dtype, 
            label_smoothing : float = 0.0
        ):
        self.device = device
        self.dtype = dtype

        self.weights = torch.tensor(weights).to(device=device, dtype=dtype)
        self.n_levels = len(weights)
        self.class_counts = class_counts
        self.label_smoothing = label_smoothing

        self.class_weights = [cc.to(self.device, self.dtype) for cc in class_counts]
        self.class_weights = [1 / (w + w.mean()) for w in self.class_weights]
        self.class_weights = [w / w.mean() for w in self.class_weights]
        for i in self.class_weights:
            i.requires_grad = False
        
        self._loss_fns = [
            nn.CrossEntropyLoss(
                weight=None, #self.class_weights[i], 
                reduction="none", 
                label_smoothing=label_smoothing
            ) for i in range(self.n_levels)
        ]

    def __call__(
            self, 
            preds : torch.Tensor, 
            targets : torch.Tensor
        ) -> "MultiLevelLoss":
        targets = targets.transpose(0, 1)
        item_weights = [self.class_weights[i][targets[i]] for i in range(self.n_levels)]
        return MultiLevelLoss([(self._loss_fns[i](preds[i], targets[i].to(self.device)) * item_weights[i]).mean() for i in range(self.n_levels)], self.weights)
    
class MultiLevelLoss:
    def __init__(
            self, 
            losses : List[torch.Tensor], 
            weights : List[Union[float, int]]
        ):
        self.losses = losses
        self.weights = weights
        if any([w < 0 for w in weights]):
            raise ValueError("Weights must be non-negative.")

    def aggregate(self) -> torch.Tensor:
        return sum([self.losses[i] * self.weights[i] for i in range(len(self.weights)) if self.weights[i] > 0])
    
    def __getitem__(self, idx : Union[int, slice]) -> torch.Tensor:
        return self.losses[idx]
    
    def __len__(self) -> int:
        return len(self.losses)

    def __iter__(self):
        return iter(self.losses)
