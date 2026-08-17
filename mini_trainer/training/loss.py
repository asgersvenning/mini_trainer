import math

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss

from mini_trainer.utils import cosine_to_zscore


class EvenCrossEntropyLoss(CrossEntropyLoss):
    """A minimal wrapper around ``torch.nn.modules.loss.CrossEntropyLoss``
    that ensures that the size of the loss is more or less independent
    from the number of classes.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        max_CE = input.new_full((1,), input.size(1), requires_grad=False).log()
        return super().forward(input=input, target=target) / max_CE


class EMLACrossEntropy(torch.nn.CrossEntropyLoss):
    """Entropy-Modulated Logit Adjusted (EMLA) Cross Entropy for Long-Tail Learning.

    This loss function dynamically applies the Logit Adjustment penalty proposed by
    Menon et al. (2021) based on the model's instance-level confidence (Shannon Entropy).

    Standard Logit Adjustment applies a static penalty to rare classes to ensure
    Fisher consistency for the balanced error. It enforces a large relative margin
    between the logits of rare and dominant labels. However, applying this penalty
    uniformly can disrupt early-stage feature learning or over-penalize genuinely ambiguous samples.

    This method introduces an instance-aware curriculum-learning gate:
    1. Calculates the exact Shannon Entropy of the raw logits using purely numerically
       stable log-space arithmetic via the identity: log(softmax(z)) = z - LSE(z).
    2. Normalizes the entropy to a [0, 1] scale (where 0 is fully certain, 1 is uniform).
    3. Computes a 'confidence' score (1 - normalized_entropy) which is detached from the gradient.
    4. Scales the class prior penalty (tau * log(pi_y)) by this confidence score.
    5. Applies the modulated penalty to the raw logits before native Cross-Entropy normalization.

    Mechanics:
        Unconfident predictions (e.g., early training or noisy samples) yield high entropy,
        suppressing the penalty and allowing standard Empirical Risk Minimization (ERM).
        Conversely, when the model becomes overconfident on a rare-attribute sample,
        the low entropy triggers the full negative logit penalty, driving the softmax
        probability to zero and generating a maximum-strength gradient to correct the boundary.

    References:
        - Menon, A. K., Jain, H., Rawat, A. S., Veit, A., & Kumar, S. (2021).
          Long-tail learning via logit adjustment. arXiv preprint arXiv:2007.07314.
    """

    def __init__(
        self,
        class_frequencies: list[int] | list[float] | np.ndarray | torch.Tensor,
        flatten: float = 0.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,  # Apparently `-100` is used instead of `None` in nn.CrossEntropy
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        device: torch.types.Device = None,
    ) -> None:
        """.

        Args:
            class_frequencies: The raw frequency or count of each class in the training dataset.
            flatten: Adjusts the weights (i.e. the inverse class frequencies, normalized)
                such that they are a mixture of the uniform and the raw distribution with weight `flatten`.
            weight: A manual rescaling weight given to each class.
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss.
            device: Device expected during training can optionally be passed, otherwise it will be inferred dynamically.
        """
        # Initialize the parent nn.CrossEntropyLoss with all standard arguments
        super().__init__(weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
        self._device = device
        if isinstance(self._device, (int, str)):
            self._device = torch.device(self._device)

        # Safely convert to a float tensor whether the input is a list or already a tensor
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

        # Register the base adjustments as a buffer so they move to the correct device
        self.register_buffer("adjustments", log_priors)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """.

        Args:
            logits: Raw, unnormalized outputs of shape (Batch, Classes).
            targets: Ground truth class indices of shape (Batch,).

        Returns:
            The computed loss.
        """
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
            The size of the set will be equal to the square root of the number of classes.
            Will use a random subset of classes each time.

    Returns:
        A scalar tensor representing the regularization loss.
    """
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
