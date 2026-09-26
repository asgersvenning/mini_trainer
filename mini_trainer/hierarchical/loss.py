import torch
from torch import nn


class MultiLevelWeightedCrossEntropyLoss(torch.nn.modules.loss._Loss):  # noqa: D101 TODO
    def __init__(  # noqa: D107
        self,
        num_classes: list[int],
        device: torch.device,
        dtype: torch.dtype,
        weights: list[float] | list[int] | torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        loss_cls: type[nn.CrossEntropyLoss] = nn.CrossEntropyLoss,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.loss_cls = loss_cls
        self.n_levels = len(num_classes)
        if weights is None:
            weights = [1] * self.n_levels
        self.weights = torch.tensor(weights).to(device=self.device, dtype=self.dtype)

        # Smoothing decreases toward the root: ls(i) = 1 - (1 - ls(0))**(1/(i+1)).
        # Correct parents receive probability mass from multiple sibling classes.
        self.label_smoothing = [1 - (1 - label_smoothing) ** (1 / (i + 1)) for i in range(self.n_levels)]
        kwargs["label_smoothing"] = self.label_smoothing

        # Construct marginal loss functions and handle vectorized (per-level) keyword arguments
        self._loss_fns: torch.nn.ModuleList = torch.nn.ModuleList()
        for lvl in range(self.n_levels):
            lvlkw = dict()
            for k, v in kwargs.items():
                if not isinstance(v, list) and len(v) == self.n_levels:
                    lvlkw[k] = v
                else:
                    lvlkw[k] = v[lvl]
            self._loss_fns.append(self.loss_cls(**lvlkw))

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        targets = targets.transpose(0, 1)
        losses: list[torch.Tensor] = [self._loss_fns[i](preds[i], targets[i]) for i in range(self.n_levels)]
        return [loss * weight for loss, weight in zip(losses, self.weights)]

    def __repr__(self):
        return "{}({})[{}]".format(self.__class__.__name__, self.loss_cls.__name__, " x ".join(f"{w:.1f}" for w in self.weights.tolist()))
