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
    return -log_prob.sum() / (num_classes * (num_classes - 1) / 2)

    # Old method return sum of z-scores
    # class_dmat_cdf = chi2_dist.cdf(chi2_statistic)

    # # We only need the upper (or lower) triangle of the matrix, excluding the diagonal
    # # as distances are symmetric (d_ij = d_ji) and d_ii = 0.
    # indices = torch.triu_indices(num_classes, num_classes, offset=1, device=weights_f64.device)
    # if indices.numel() == 0: # Handles num_classes < 2, though already checked
    #     return torch.tensor(0.0, device=classification_weights.device, dtype=classification_weights.dtype)
        
    # unique_cdf_values = class_dmat_cdf[indices[0], indices[1]]

    # # 4. Quantile-Quantile Transformation: CDF values to Z-scores
    # # Clamp CDF values to (epsilon, 1-epsilon) to avoid -inf/inf from icdf for perfect 0 or 1
    # clamped_cdf_values = torch.clamp(unique_cdf_values, min=epsilon, max=1.0 - epsilon)

    # normal_dist = Normal(
    #     torch.tensor(0.0, device=weights_f64.device, dtype=torch.float64),
    #     torch.tensor(1.0, device=weights_f64.device, dtype=torch.float64)
    # )
    # z_scores = normal_dist.icdf(clamped_cdf_values) # Inverse CDF (probit)

    # # 5. Regularization term: sum of squared Z-scores
    # # This penalizes deviations where the observed distance was very unlikely (tail regions of Chi2).
    # # Normalizing by the number of pairs to make it somewhat independent of num_classes.
    # num_pairs = z_scores.numel()
    # if num_pairs == 0: # Should be caught by num_classes < 2 earlier
    #      return torch.tensor(0.0, device=classification_weights.device, dtype=classification_weights.dtype)

    # reg_term = torch.sum(z_scores ** 2) / num_pairs

    # return reg_term.to(classification_weights.dtype) # Cast back to original dtype