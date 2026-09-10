"""Linear spatial views with measured neighbourhood preservation."""

import numpy as np
import torch
from scipy.spatial import cKDTree


def projected_neighbours(coordinates, k):
    """Euclidean neighbours in the displayed plane; exact ties use class order."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=k + 1)
    result = []
    for i, radius in enumerate(distances[:, -1]):
        candidates = np.asarray(tree.query_ball_point(coordinates[i], np.nextafter(radius, np.inf)), dtype=int)
        candidates = candidates[candidates != i]
        squared = ((coordinates[candidates] - coordinates[i]) ** 2).sum(axis=1)
        result.append(candidates[np.lexsort((candidates, squared))[:k]].tolist())
    return result


@torch.no_grad()
def prototype_projections(weight, neighbours):
    """Project the same row directions used in the cosine diagnostic.

    PCA centers unit rows without whitening or feature standardization. It does
    not change prototype weights, repository distances or their neighbour ranks.
    Both displayed axes share one spatial scale. Fit each case independently.
    """
    weight = weight.detach().cpu().float()
    norms = weight.norm(dim=1, keepdim=True)
    if not torch.isfinite(weight).all() or (norms == 0).any():
        raise ValueError("Projection requires finite, nonzero prototype rows")
    directions = weight / norms
    centered = directions - directions.mean(dim=0)
    # Symmetric covariance eigendecomposition avoids an N x N pairwise matrix.
    gram = centered.T @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    total = centered.square().sum().item()
    count = min(4, weight.shape[1])
    axes = eigenvectors[:, -count:].flip(1)
    # Fix the otherwise arbitrary sign; degenerate eigenspaces may still rotate.
    pivots = axes.abs().argmax(dim=0)
    axes *= torch.where(axes[pivots, torch.arange(count)] < 0, -1.0, 1.0)
    coordinates = (centered @ axes).numpy()
    variances = eigenvalues[-count:].flip(0).clamp_min(0).numpy()
    result = {}
    original = [set(row) for row in neighbours]
    k = len(neighbours[0])
    for start in range(0, count - 1, 2):
        points = coordinates[:, start : start + 2]
        near = projected_neighbours(points, k)
        retained = [len(original[i].intersection(row)) / k for i, row in enumerate(near)]
        key = f"PCA {start + 1}–{start + 2}"
        result[key] = {
            "coordinates": points.tolist(),
            "axes": [f"PC {start + 1}", f"PC {start + 2}"],
            "variance_fraction": (variances[start : start + 2] / total).tolist() if total > 0 else [0.0, 0.0],
            "neighbours": near,
            "retained_fraction": retained,
            "mean_retained_fraction": float(np.mean(retained)),
            "method": "Centered PCA of unit prototype directions; float32 covariance eigendecomposition; no whitening",
        }
    return result
