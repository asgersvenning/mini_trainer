import torch
from torch.distributions import Chi2

def class_weight_distribution_regularization(
    classification_weights: torch.Tensor,
    epsilon: float = 1e-6, # For numerical stability
    sparse : bool=True
):
    """
    Calculates a regularization term based on the pairwise distances of
    normalized class weight vectors, assuming a Chi-squared distribution prior
    for these distances.

    The regularization term is defined as:
        R = -sum[ log(L_chi2( N_E , tril(WW) )) * 1_{tril(WW) < E(chi2( N_E ))} ] + |tril(WW)|
    Where N_E is the number of embedding dimensions and tril(WW) is the lower triangle (not including the diagonal) of:
        WW = (|| W, W ||^2) / 2
    Thus R corresponds to the negative log-likelihood of tril(WW) given a Chi2 distribution with N_E degrees of freedom, divided by the number of "samples".

    For efficiency instead of letting W be the full weight matrix, if `sparse=True`, a random subset of classes corresponding to ~sqrt(N_C) is used.
    This has the effect that the size of WW is O(N_C) instead of O(N_C^2), which is not ideal when you have thousands of classes.

    Args:
        classification_weights: Tensor of shape [num_classes, num_embeddings],
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
        mean_weights = classification_weights.mean(dim=0, keepdim=True)
        # Add epsilon to std to prevent division by zero if a weight vector's components are all identical
        std_weights = classification_weights.std(dim=0, unbiased=True, keepdim=True) + epsilon

    # Select a subset of classes to regularize
    if sparse:
        w_size = max(2, round(len(classification_weights) ** (1/2)))
        w = torch.randperm(len(classification_weights), device=classification_weights.device, dtype=torch.long)[:w_size]
        classification_weights = classification_weights[w]

    num_classes, num_embeddings = classification_weights.shape
    if num_classes < 2 or num_embeddings == 0:
        return torch.tensor(0.0, device=classification_weights.device, dtype=classification_weights.dtype)

    # 1. Normalize each embedding to have mean 0 and std 1. 
    # Under the assumption that the embedding dimensions are independent, 
    # each row in the weight matrix can now be considered a sample from a standard multivariate gaussian.
    normalized_weights = (classification_weights - mean_weights) / std_weights

    # 2. Calculate squared Euclidean distances and the Chi2 statistic
    class_dmat_sq = torch.cdist(normalized_weights, normalized_weights, p=2) ** 2

    # Statistic for Chi2 distribution: D^2 / 2
    chi2_statistic = class_dmat_sq / 2.0
    chi2_statistic_tril = chi2_statistic[*torch.tril_indices(*chi2_statistic.shape, -1)]

    # 3. CDF Transformation
    dof_tensor = torch.tensor(float(num_embeddings), device=classification_weights.device, dtype=classification_weights.dtype)
    if dof_tensor <= 0: # Should not happen if num_embeddings > 0
        return torch.tensor(0.0, device=classification_weights.device, dtype=classification_weights.dtype)
        
    chi2_dist = Chi2(dof_tensor)
    chi2_expected = chi2_dist.mean
    
    # Calculate the density of the statistics for the values below the expected value
    # (since we only want to penalize classes which are too close, not too far)
    # and multiply by two to compensate
    log_prob : torch.Tensor = 2 * chi2_dist.log_prob(chi2_statistic_tril[chi2_statistic_tril < chi2_expected])
    
    # Return the likelihood divided by the number of statistics
    return -log_prob.sum() + torch.tensor((num_classes * (num_classes - 1) / 2), device=classification_weights.device, dtype=classification_weights.dtype).log()