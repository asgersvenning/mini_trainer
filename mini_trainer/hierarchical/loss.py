import torch
from torch import nn
from torch._prims_common import DeviceLikeType
from torch.types import _dtype


class MultiLevelWeightedCrossEntropyLoss(torch.nn.modules.loss._Loss):  # noqa: D101 TODO
    def __init__(  # noqa: D107
        self,
        weights: list[float | int] | torch.Tensor,
        device: DeviceLikeType,
        dtype: _dtype,
        label_smoothing: float = 0.0,
        loss_cls: type[nn.CrossEntropyLoss] = nn.CrossEntropyLoss,
        **kwargs,
    ):
        self.device = device
        self.dtype = dtype

        self.weights = torch.tensor(weights).to(device=device, dtype=dtype)
        self.n_levels = len(weights)

        # The adjustment:
        #   ls(L)=1-(1-ls(0))^(1/(L+1)), ls(0)=k
        # is to avoid a situation where the model gives the target probability for the correct leaf class,
        # e.g. if ls=0.1, the model predicts P(Correct_0 | Model, Data) = 1 - ls = 0.9, and distributes the remaining
        # probability mass to the correct class siblings (i.e. other species in the correct genus), then the model must
        # give a higher confidence for the correct parent (child):
        #   P(Correct_1 | Model, Data) > P(Correct_0 | Model, Data)
        # (if it gives any confidence to the sibling classes), meaning that the model is encouraged NOT to give any
        # confidence to the sibling classes, which is counter to the point of hierarchical learning
        self.label_smoothing = [1 - (1 - label_smoothing) ** (1 / (i + 1)) for i in range(self.n_levels)]
        kwargs["label_smoothing"] = self.label_smoothing

        # Construct marginal loss functions and handle vectorized (per-level) keyword arguments
        self._loss_fns: list[nn.CrossEntropyLoss] = []
        for lvl in range(self.n_levels):
            lvlkw = dict()
            for k, v in kwargs.items():
                if not isinstance(v, list) and len(v) == self.n_levels:
                    lvlkw[k] = v
                else:
                    lvlkw[k] = v[lvl]
            self._loss_fns.append(loss_cls(**lvlkw))

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        targets = targets.transpose(0, 1)
        losses: list[torch.Tensor] = [self._loss_fns[i](preds[i], targets[i]) for i in range(self.n_levels)]
        return [loss * weight for loss, weight in zip(losses, self.weights)]
