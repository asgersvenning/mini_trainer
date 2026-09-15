"""Projection geometry and distortion indicators have independent references."""

import numpy as np
import torch

from dev.prototype_space.projection import projected_neighbours, prototype_projections


def test_pca_matches_svd_preserves_input_and_reports_actual_neighbours():
    weight = torch.randn(24, 7, generator=torch.Generator().manual_seed(123))
    weight *= torch.linspace(0.2, 3.0, len(weight))[:, None]
    before = weight.clone()
    unit = (weight / weight.norm(dim=1, keepdim=True)).numpy()
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argsort(-similarity, axis=1, kind="stable")[:, :4].tolist()
    result = prototype_projections(weight, neighbours)
    centered = unit - unit.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    for start, plane in zip([0, 2], result.values()):
        actual = np.array(plane["coordinates"])
        expected = centered @ vt[start : start + 2].T
        # Signs are arbitrary; Euclidean geometry must match independent SVD.
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=3e-6, rtol=1e-5)
        np.testing.assert_allclose(plane["variance_fraction"], singular[start : start + 2] ** 2 / (singular**2).sum(), atol=1e-6)
        distances = ((actual[:, None] - actual[None, :]) ** 2).sum(axis=2)
        np.fill_diagonal(distances, np.inf)
        near = np.argsort(distances, axis=1, kind="stable")[:, :4].tolist()
        assert plane["neighbours"] == near
        retained = [len(set(a).intersection(b)) / 4 for a, b in zip(near, neighbours)]
        assert plane["retained_fraction"] == retained
        assert plane["mean_retained_fraction"] == np.mean(retained)
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_degenerate_projection_and_distance_ties_are_explicit_and_finite():
    points = np.zeros((5, 2))
    assert projected_neighbours(points, 2) == [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]]
    result = prototype_projections(torch.ones(2, 4), [[1], [0]])
    for plane in result.values():
        assert plane["variance_fraction"] == [0.0, 0.0]
        assert np.isfinite(plane["coordinates"]).all()
        assert plane["retained_fraction"] == [1.0, 1.0]


def test_angular_projection_uses_repository_geometry_and_reports_fidelity():
    from dev.prototype_space.explore import weight_model
    from dev.prototype_space.projection import angular_distances, angular_tsne
    from mini_trainer.modeling import class_similarity

    weights = torch.randn(36, 8, generator=torch.Generator().manual_seed(81))
    z = class_similarity(weight_model(weights), cdf=False)[0].numpy()
    unit = (weights / weights.norm(dim=1, keepdim=True)).numpy()
    reference = np.arccos(np.clip(unit @ unit.T, -1 + 1e-7, 1 - 1e-7))
    np.fill_diagonal(reference, 0)
    np.testing.assert_allclose(angular_distances(z, 8), reference, atol=5e-7)
    np.fill_diagonal(z, -np.inf)
    neighbours = np.argsort(-z, axis=1, kind="stable")[:, :4].tolist()
    np.fill_diagonal(z, 0)
    result = angular_tsne(z, 8, neighbours, iterations=300)
    assert np.isfinite(result["coordinates"]).all()
    assert result["variance_fraction"] is None
    expected = [len(set(a).intersection(b)) / 4 for a, b in zip(neighbours, result["neighbours"])]
    assert result["retained_fraction"] == expected
    assert result["parameters"]["metric"] == "precomputed angular radians"
