import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Chi2
from torch.nn.modules.loss import CrossEntropyLoss


class EvenCrossEntropyLoss(CrossEntropyLoss):
    """A minimal wrapper around ``torch.nn.modules.loss.CrossEntropyLoss``
    that ensures that the size of the loss is more or less independent
    from the number of classes.
    """
    def forward(self, input : torch.Tensor, target : torch.Tensor):
        max_CE = input.new_full((1, ), input.size(1), requires_grad=False).log()
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
        flatten: float = 0.1, 
        weight: torch.Tensor | None = None,
        ignore_index: int = -100, # Apparently `-100` is used instead of `None` in nn.CrossEntropy
        reduction: str = 'mean',
        label_smoothing: float = 0.0
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
        """
        # Initialize the parent nn.CrossEntropyLoss with all standard arguments
        super().__init__(
            weight=weight, 
            ignore_index=ignore_index, 
            reduction=reduction, 
            label_smoothing=label_smoothing
        )
        
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
        log_priors = (log_counts - log_counts.mean())
        
        # Register the base adjustments as a buffer so they move to the correct device
        self.register_buffer('adjustments', log_priors)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """.

        Args:
            logits: Raw, unnormalized outputs of shape (Batch, Classes).
            targets: Ground truth class indices of shape (Batch,).
            
        Returns:
            The computed loss.
        """
        # Uncertainty gate: 1.0 when confident, 0.0 when uncertain
        with torch.no_grad():
            log_probs = logits.log_softmax(dim=-1)
            entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1, keepdim=True)
            evenness = 1.0 - entropy / math.log(logits.size(-1))
        
        return super().forward(logits + (evenness * self.adjustments.to(logits.device)), targets)



def kl_distill_ema(
        logits : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        ema_logits : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        T : float=1.0
    ):
    """Compute KL-divergence between online model-logits and EMA-model logits.
    """
    if isinstance(logits, (list, tuple)) or isinstance(ema_logits, (list, tuple)):
        return torch.stack([kl_distill_ema(lg, ema_lg, T=T) for lg, ema_lg in zip(logits, ema_logits)]).mean()
    orig_dtype = logits.dtype
    logits = (logits / T).float().log_softmax(-1)
    ema_logits = (ema_logits / T).float().log_softmax(-1).detach()
    return (F.kl_div(logits, ema_logits, log_target=True, reduction="batchmean") * T**2).to(dtype=orig_dtype)


def class_weight_distribution_regularization(
    W: torch.Tensor,
    epsilon: float = 1e-6,
    sparse : bool=True
):
    """Calculates a regularization term based on the pairwise distances of
      normalized class weight vectors, assuming a Chi-squared distribution prior
      for these distances.

    The regularization term is defined as:
        :math:`R = -sum[ log(L_chi2( N_E , tril(WW) )) * 1_{tril(WW) < E(chi2( N_E ))} ] + |tril(WW)|`
    Where N_E is the number of embedding dimensions and :math:`tril(WW)`
      is the lower triangle (not including the diagonal) of:
        :math:`WW = (|| W, W ||^2) / 2`
    Thus R corresponds to the negative log-likelihood of :math:`tril(WW)`
      given a Chi2 distribution with N_E degrees of freedom, 
      divided by the number of "samples".

    For efficiency instead of letting W be the full weight matrix, if `sparse=True`, 
      a random subset of classes corresponding to :math:`~sqrt(N_C)` is used.
    This has the effect that the size of WW is :math:`O(N_C)` instead of :math:`O(N_C^2)`, 
      which is not ideal when you have thousands of classes.

    Args:
        W: Tensor of shape [num_classes, num_embeddings],
            typically the weights of the final linear layer.
        epsilon: Small value for numerical stability.
        sparse: Use a sparse set of classes to compute the regularization over.
            The size of the set will be equal to the square root of the number of classes.
            Will use a random subset of classes each time.

    Returns:
        A scalar tensor representing the regularization loss.
    """
    with torch.no_grad():
        # Calculate embedding statistics on the full weight matrix 
        # (without gradient; the embeddings are assumed to be normalized with batchnorm already)
        muE = W.mean(dim=0, keepdim=True)
        # Add epsilon to std to prevent division by zero if a weight vector's components are all identical
        sigE = W.std(dim=0, unbiased=True, keepdim=True) + epsilon

    # Select a subset of classes to regularize
    _n = min(len(W), max(32, 2 * round(len(W) ** 0.5)))
    if sparse and _n < len(W):
        _sparse_idx = torch.randperm(len(W))[:_n]
        W = W[_sparse_idx]

    N, E = W.shape
    if N < 2 or E == 0:
        return torch.tensor(0.0, device=W.device, dtype=W.dtype)

    # 1. Normalize each embedding to have mean 0 and std 1. 
    # Under the assumption that the embedding dimensions are independent, 
    # each row in the weight matrix can now be considered a sample from a standard multivariate gaussian.
    WN = (W - muE) / sigE

    # 2. Calculate squared Euclidean distances and the Chi2 statistic
    cdm2 = torch.cdist(WN, WN, p=2) ** 2

    # Statistic for Chi2 distribution: D^2 / 2
    chi2 = cdm2 / 2.0
    chi2_tril = chi2[*torch.tril_indices(*chi2.shape, -1)]

    # 3. CDF Transformation
    dof_tensor = torch.tensor(float(E), device=W.device, dtype=W.dtype)
    if dof_tensor <= 0: # Should not happen if num_embeddings > 0
        return torch.tensor(0.0, device=W.device, dtype=W.dtype)      
    chi2_dist = Chi2(dof_tensor)
    
    # Calculate the density of the statistics for the values below the expected value
    # (since we only want to penalize classes which are too close, not too far)
    # and multiply by two to compensate
    log_prob : torch.Tensor = 2 * chi2_dist.log_prob(chi2_tril[chi2_tril < chi2_dist.mean])
    
    # Return the likelihood divided by the number of statistics
    return -log_prob.sum() / torch.tensor((N * (N - 1) / 2), device=W.device, dtype=W.dtype)


def weight_kl_gaussian(
        W: torch.Tensor,
        eps: float = 1e-6,
        sparse: bool = True,
        normalize_rows: bool = True,
    ) -> torch.Tensor:
    """Regularization term that encourages orthogonality between rows in W.
    """
    if W.numel() == 0 or W.ndim != 2 or W.size(0) <= 1 or W.size(1) == 0:
        # return scalar tensor on same device; fp32 is fine
        return torch.zeros((), device=W.device, dtype=torch.float32)

    with torch.amp.autocast(W.device.type, enabled=False):
        comp_dtype = torch.float32 if W.dtype in (torch.float16, torch.bfloat16) else W.dtype
        Wc = W.to(comp_dtype)

        if normalize_rows:
            Wc = F.normalize(Wc, dim=1)
        if sparse:
            _n = min(Wc.size(0), max(32, 2 * int(Wc.size(0) ** 0.5)))
            if _n < Wc.size(0):
                idx = torch.randperm(Wc.size(0), device=Wc.device)[:_n]
                Wc = Wc[idx]

        N = Wc.size(0)
        if N <= 1:
            return torch.zeros((), device=W.device, dtype=torch.float32)

        eps_t = torch.as_tensor(eps, dtype=comp_dtype, device=Wc.device)

        # SVD in fp32 + jitter for stability
        fro = Wc.norm(p="fro")
        scale = (fro / (Wc.numel() ** 0.5)).clamp(min=1.0)
        jitter = eps * scale
        if jitter > 0:
            Wc = Wc + jitter * torch.randn_like(Wc)
        try:
            s2 = torch.linalg.svdvals(Wc) ** 2
        except torch._C._LinAlgError:
            # if SVD still fails, do not destabilize training
            return torch.zeros((), device=W.device, dtype=torch.float32)
        R = s2.numel()

        logdet = (s2 + eps_t).log().sum()
        if N > R:
            logdet = logdet + (N - R) * eps_t.log()

        tr = s2.sum()
        return 0.5 * (tr - logdet - N)


def coherence_hinge_regularization(
    W: torch.Tensor,
    scale: float = 1.5, # >1 makes the threshold looser than Welch bound
    sparse: bool=True,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Penalize only overly-similar class directions:
        :math:`loss = E_{(i,j)} [ ReLU( cos(w_i, w_j) - τ )^2 ]`.
    τ is set from the Welch bound: 
        :math:`τ = scale * sqrt((K-d)/(d(K-1)))` Clipped to [0, 0.5]
    .
    """
    if len(W) <= 1 or W.shape[1] == 0:
        return torch.zeros((1,), dtype=W.dtype, device=W.device)
    if normalize_rows:
        W = F.normalize(W, dim=1)
    _n = min(len(W), max(32, 2 * round(len(W) ** 0.5)))
    if sparse and _n < len(W):
        _sparse_idx = torch.randperm(len(W))[:_n]
        W = W[_sparse_idx]
    N, E = W.shape

    mu_welch = max((N - E) / (E * (N - 1)), 0.0) ** 0.5
    tau = float(min(max(scale * mu_welch, 0.0), 0.5))

    i, j = torch.tril_indices(N, E, offset=-1)

    cos_ij = (W[i] * W[j]).sum(dim=1)
    return F.relu(cos_ij - tau).square().mean()