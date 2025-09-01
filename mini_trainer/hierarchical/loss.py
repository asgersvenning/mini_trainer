
import torch
from torch import nn
from torch._prims_common import DeviceLikeType
from torch.types import _dtype


class MultiLevelWeightedCrossEntropyLoss(torch.nn.modules.loss._Loss):
    def __init__(
            self, 
            weights : list[float | int] | torch.Tensor,
            device : DeviceLikeType, 
            dtype : _dtype, 
            class_weights : list[torch.Tensor] | None=None,
            label_smoothing : float = 0.0
        ):
        self.device = device
        self.dtype = dtype

        self.weights = torch.tensor(weights).to(device=device, dtype=dtype)
        self.n_levels = len(weights)
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = [w.to(device=device, dtype=dtype) for w in class_weights]
            for i in self.class_weights:
                i.requires_grad = False

        # The adjustment: ls(L)=1-(1-ls(0))^(1/(L+1)), ls(0)=k
        # is to avoid a situation where the model gives the target probability for the correct leaf class,
        # e.g. if ls=0.1, the model predicts P(Correct_0 | Model, Data) = 1 - ls = 0.9, and distributes the remaining 
        # probability mass to the correct class siblings (i.e. other species in the correct genus), then the model must 
        # give a higher confidence for the correct parent (child): P(Correct_1 | Model, Data) > P(Correct_0 | Model, Data)
        # (if it gives any confidence to the sibling classes), meaning that the model is encouraged NOT to give any confidence
        # to the sibling classes, which is counter to the point of hierarchical learning
        self.label_smoothing = [1 - (1 - label_smoothing)**(1/(i+1)) for i in range(self.n_levels)]
        
        self._loss_fns = [
            nn.CrossEntropyLoss(
                weight=None, #self.class_weights[i], 
                reduction="none", 
                label_smoothing=label_smoothing
            ) for _ in range(self.n_levels)
        ]

    def __call__(
            self, 
            preds : torch.Tensor, 
            targets : torch.Tensor
        ) -> "MultiLevelLoss":
        targets = targets.transpose(0, 1)
        if self.class_weights is None:
            item_weights = [targets[i].new_ones(targets[i].shape) for i in range(self.n_levels)]
        else:
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
            weights : list[float | int]
        ):
        self.losses = losses
        self.weights = weights
        if any([w < 0 for w in weights]):
            raise ValueError("Weights must be non-negative.")

    def aggregate(self) -> torch.Tensor:
        return sum([self.losses[i] * self.weights[i] for i in range(len(self.weights)) if self.weights[i] > 0])
    
    def __getitem__(self, idx : int | slice) -> torch.Tensor:
        return self.losses[idx] * self.weights[idx]
    
    def __len__(self) -> int:
        return sum([int(w > 0 for w in self.weights)])

    def __iter__(self):
        for weight, loss in zip(self.weights, self.losses):
            if weight == 0:
                continue
            yield weight * loss
    
    def __repr__(self):
        return f'Losses: [{", ".join([f"{loss.item():.1f}" for loss in self.losses])}]\nWeights: [{", ".join([f"{weight:.1f}" for weight in self.weights])}]'
