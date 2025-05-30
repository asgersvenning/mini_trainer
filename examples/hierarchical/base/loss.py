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
from typing import Union
import torch
from torch import nn
from torch._prims_common import DeviceLikeType
from torch.types import _dtype

class MultiLevelCrossEntropyLoss(torch.nn.modules.loss._Loss):
    def __init__(
            self, 
            weights : Union[list[Union[float, int]], torch.Tensor], 
            label_smoothing : float = 0.0
        ):

        self.weights = [float(v) for v in weights]
        self.n_levels = len(weights)
        self.label_smoothing = label_smoothing
        
        self._loss_fns = [
            nn.CrossEntropyLoss(
                weight=None, 
                reduction="none", 
                label_smoothing=label_smoothing
            ) for i in range(self.n_levels)
        ]

    def __call__(
            self, 
            preds : torch.Tensor, 
            targets : torch.Tensor
        ):
        targets = targets.transpose(0, 1)
        return list(MultiLevelLoss(
            [
                (self._loss_fns[i](preds[i], targets[i])).mean()
                for i, w in enumerate(self.weights) if w > 0
            ], 
            [w for w in self.weights if w > 0]
        ))

class MultiLevelWeightedCrossEntropyLoss(torch.nn.modules.loss._Loss):
    def __init__(
            self, 
            weights : Union[list[Union[float, int]], torch.Tensor], 
            # class_counts : list[torch.Tensor], 
            class_weights : list[torch.Tensor],
            device : DeviceLikeType, 
            dtype : _dtype, 
            label_smoothing : float = 0.0
        ):
        self.device = device
        self.dtype = dtype

        self.weights = torch.tensor(weights).to(device=device, dtype=dtype)
        self.n_levels = len(weights)
        # self.class_counts = class_counts
        self.class_weights = [w.to(device=device, dtype=dtype) for w in class_weights]
        self.label_smoothing = label_smoothing

        # self.class_weights = [cc.to(self.device, self.dtype) for cc in class_counts]
        # self.class_weights = [1 / (w + w.mean()) for w in self.class_weights]
        # self.class_weights = [w / w.mean() for w in self.class_weights]
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
        return list(MultiLevelLoss(
            [
                (self._loss_fns[i](preds[i], targets[i].to(self.device)) * item_weights[i]).mean()
                for i in range(self.n_levels)
            ], 
             self.weights
        ))
    
class MultiLevelLoss:
    def __init__(
            self, 
            losses : list[torch.Tensor], 
            weights : list[Union[float, int]]
        ):
        self.losses = losses
        self.weights = weights
        if any([w < 0 for w in weights]):
            raise ValueError("Weights must be non-negative.")

    def aggregate(self) -> torch.Tensor:
        return sum([self.losses[i] * self.weights[i] for i in range(len(self.weights)) if self.weights[i] > 0])
    
    def __getitem__(self, idx : Union[int, slice]) -> torch.Tensor:
        return self.losses[idx] * self.weights[idx]
    
    def __len__(self) -> int:
        return sum([int(w > 0 for w in self.weights)])

    def __iter__(self):
        for w, l in zip(self.weights, self.losses):
            if w == 0:
                continue
            yield w * l
    
    def __repr__(self):
        return f'Losses: [{", ".join([f"{loss.item():.1f}" for loss in self.losses])}]\nWeights: [{", ".join([f"{weight:.1f}" for weight in self.weights])}]'
