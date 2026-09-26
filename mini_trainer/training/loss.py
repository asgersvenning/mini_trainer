import math

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss

from mini_trainer.utils import cosine_to_zscore


class EvenCrossEntropyLoss(CrossEntropyLoss):
    """Cross entropy divided by the log of the number of classes."""

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        max_CE = input.new_full((1,), input.size(1), requires_grad=False).log()
        return super().forward(input=input, target=target) / max_CE


class EMLACrossEntropy(torch.nn.CrossEntropyLoss):
    """Cross entropy with a detached, entropy-gated class-frequency adjustment.

    Add ``(1 - H(softmax(input)) / log(C)) * adjustments`` to each sample's
    logits, where ``C`` is its class count and ``adjustments`` contains centered
    log class counts. Uncertain predictions receive less adjustment; uniform
    counts produce no adjustment. The gate does not contribute gradients.

    Adapts logit adjustment from Menon et al. (2021), arXiv:2007.07314, with
    this implementation's entropy gate. Quality trade-offs require evaluation
    on the intended dataset; see docs/training-feature-validation.md.
    """

    def __init__(
        self,
        class_frequencies: list[int] | list[float] | np.ndarray | torch.Tensor,
        flatten: float = 0.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        device: torch.types.Device = None,
    ) -> None:
        """Initialize adjustments from rounded class counts, clamped to at least one.

        ``flatten`` replaces counts ``c`` with ``flatten * sum(c) +
        (1 - flatten) * c``; zero preserves counts and one makes them uniform.
        ``device`` optionally fixes the adjustment device; otherwise forward
        uses the input device. Other arguments pass through to CrossEntropyLoss.
        """
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
        self._device = device
        if isinstance(self._device, (int, str)):
            self._device = torch.device(self._device)

        if isinstance(class_frequencies, np.ndarray):
            class_frequencies = torch.from_numpy(class_frequencies)
        if isinstance(class_frequencies, (list, tuple)):
            class_frequencies = torch.tensor(class_frequencies)
        if isinstance(class_frequencies, torch.Tensor):
            counts = class_frequencies.round().long()

        counts = torch.clamp(counts, min=1)
        if flatten != 0:
            assert flatten > 0 and flatten <= 1
            counts = flatten * counts.sum() + (1 - flatten) * counts
        log_counts = torch.log(counts)
        log_priors = log_counts - log_counts.mean()
        if self._device is not None:
            log_priors = log_priors.to(device=self._device)

        self.register_buffer("adjustments", log_priors)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply the detached adjustment before cross entropy."""
        # Uncertainty gate: 1.0 when confident, 0.0 when uncertain
        with torch.no_grad():
            log_probs = input.log_softmax(dim=-1)
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1, keepdim=True)
            evenness = 1.0 - entropy / math.log(input.size(-1))

        adjustments = self.adjustments
        assert isinstance(adjustments, torch.Tensor)
        if self._device is None:
            adjustments = adjustments.to(input.device)

        return super().forward(input + (evenness * adjustments), target)


def class_weight_distribution_regularization(W: torch.Tensor, sparse: bool = True):
    """Calculates a regularization term on the assumption that the weights should be uniformly distributed unit vectors.
    If the weights are not unit vectors, they will be normalized before the computation.

    Args:
        W: Tensor of shape [num_classes, num_embeddings],
            typically the weights of the final linear layer.
        sparse: Use a sparse set of classes to compute the regularization over.
            Samples a bounded random subset on each call.

    Returns:
        A scalar tensor representing the regularization loss.
    """
    if getattr(W, "_is_quantized_training", False):
        W = W.dequantize()
    # Select a subset of classes to regularize
    _n = min(len(W), max(32, 2 * round(len(W) ** 0.5)))
    if sparse and _n < len(W):
        _sparse_idx = torch.randperm(len(W))[:_n].sort().values
        W = W[_sparse_idx]

    N, E = W.shape
    if N < 2 or E == 0:
        return torch.tensor(0.0, device=W.device, dtype=W.dtype)

    with torch.amp.autocast(W.device.type, enabled=False):
        if not W.norm(2, 1).allclose(torch.ones_like(W[0, 0])):
            with torch.no_grad():
                WN = W.norm(2, 1, True)
            W = W / WN

        tril_idx = torch.tril_indices(*W.shape, offset=-1)
        ztril = cosine_to_zscore(W @ W.T, W.shape[-1])[*tril_idx]

        return 2 * ztril[ztril > 0].sum() / tril_idx.shape[-1]
