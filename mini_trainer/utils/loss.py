import torch
from torch.distributions import Chi2, Normal

def class_weight_distribution_regularization(
    classification_weights: torch.Tensor,
    epsilon: float = 1e-6 # For numerical stability
):
    """
    Calculates a regularization term based on the pairwise distances of
    normalized class weight vectors, assuming a Chi-squared distribution prior
    for these distances.

    The regularization term is defined as the average 

    Args:
        classification_weights: Tensor of shape [num_classes, num_embeddings],
                                typically the weights of the final linear layer.
        epsilon: Small value for numerical stability.

    Returns:
        A scalar tensor representing the regularization loss.
    """
    num_classes, num_embeddings = classification_weights.shape

    if num_classes < 2 or num_embeddings == 0:
        return torch.tensor(0.0, device=classification_weights.device, dtype=classification_weights.dtype)

    # Work with higher precision and ensure it's on the same device as weights
    weights_f64 = classification_weights.to(dtype=torch.float64)

    # 1. Normalize each class's weight vector (across embedding dimensions)
    mean_weights = weights_f64.mean(dim=0, keepdim=True)
    # Add epsilon to std to prevent division by zero if a weight vector's components are all identical
    std_weights = weights_f64.std(dim=0, unbiased=True, keepdim=True) + epsilon
    normalized_weights = (weights_f64 - mean_weights) / std_weights

    # 2. Calculate squared Euclidean distances and the Chi2 statistic
    # D_sq_ij = ||w_i_norm - w_j_norm||^2
    # Using p=2 for Euclidean, then squaring
    class_dmat_sq = torch.cdist(normalized_weights, normalized_weights, p=2) ** 2

    # Statistic for Chi2 distribution: D^2 / 2
    # Degrees of freedom = num_embeddings
    chi2_statistic = class_dmat_sq / 2.0
    chi2_statistic_tril = chi2_statistic[*torch.tril_indices(*chi2_statistic.shape, -1)]

    # 3. CDF Transformation
    dof_tensor = torch.tensor(float(num_embeddings), device=weights_f64.device, dtype=torch.float64)
    if dof_tensor <= 0: # Should not happen if num_embeddings > 0
        return torch.tensor(0.0, device=classification_weights.device, dtype=classification_weights.dtype)
        
    chi2_dist = Chi2(dof_tensor)
    chi2_expected = chi2_dist.mean
    # New method return mean negative log-likelihood of the lower triangle
    log_prob : torch.Tensor = chi2_dist.log_prob(chi2_statistic_tril[chi2_statistic_tril <= chi2_expected])
    return -log_prob.sum() + torch.tensor((num_classes * (num_classes - 1) / 2), device=weights_f64.device, dtype=torch.float64).log()