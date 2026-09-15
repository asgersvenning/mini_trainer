"""Log-domain prototype diagnostics retain representable scores, not rounded CDFs."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mini_trainer.modeling import distance
from mini_trainer.visualization.plot import (
    _aggregate_matrix_max,
    _generate_log_heatmap_rgb_array,
    _log_colorbar_ticks,
    plot_heatmap,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_log_tail_matches_erfc_without_cdf_or_autocast_rounding(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not enabled")
    dim = 1280
    targets = torch.tensor([0.0, 5.0, 8.0, 12.0, 20.0, 31.0], device=device)
    cosine = (targets / math.sqrt(dim - 2)).sin()
    weights = torch.zeros(len(targets) + 1, dim, device=device)
    weights[0, 0] = 1
    weights[1:, 0] = cosine
    weights[1:, 1] = (1 - cosine.square()).sqrt()
    weights.requires_grad_()
    original = weights.detach().clone()
    monkeypatch.setattr(distance, "classification_module", lambda model: SimpleNamespace(last_layer_weights=weights))
    z = distance.class_similarity(None, cdf=False)[0]

    def reject_cdf(*args, **kwargs):
        raise AssertionError("Log-domain diagnostic must not materialize a CDF")

    monkeypatch.setattr(torch.distributions.Normal, "cdf", reject_cdf)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        actual = distance.class_log_similarity(None, complement=True)[0]
        log_cdf = distance.class_log_similarity(None)[0]
    expected = [math.log(0.5 * math.erfc(value / math.sqrt(2))) for value in z[0, 1:].tolist()]
    np.testing.assert_allclose(actual[0, 1:].cpu(), expected, rtol=2e-6, atol=1e-6)
    assert actual.dtype == log_cdf.dtype == torch.float32
    assert actual.device == weights.device
    assert not actual.requires_grad and weights.grad is None
    assert torch.isneginf(actual.diag()).all() and (log_cdf.diag() == 0).all()
    assert torch.isfinite(actual[0, 1:]).all()
    assert actual[0, 1:].unique().numel() == len(targets)
    torch.testing.assert_close(actual, actual.T, rtol=0, atol=0)
    torch.testing.assert_close(weights.detach(), original, rtol=0, atol=0)


def test_log_similarity_preserves_multiple_prototype_levels(monkeypatch):
    weights = [torch.eye(3), torch.eye(4)]
    monkeypatch.setattr(distance, "classification_module", lambda model: SimpleNamespace(last_layer_weights=weights))
    results = distance.class_log_similarity(None, complement=True)
    assert [tuple(result.shape) for result in results] == [(3, 3), (4, 4)]
    for result in results:
        off_diagonal = ~torch.eye(len(result), dtype=torch.bool)
        torch.testing.assert_close(result[off_diagonal], torch.full_like(result[off_diagonal], -math.log(2)))


def test_log_heatmap_retains_extreme_tails_and_masks_zero_without_exp(monkeypatch):
    values = np.array([[-np.inf, -50, -100], [-50, -500, -1000]], dtype=np.float32)
    before = values.copy()
    rgb, norm, _, _ = _generate_log_heatmap_rgb_array(values, None, "magma", True)
    assert rgb.shape == (2, 3, 3)
    assert len({tuple(rgb[r, c]) for r, c in [(0, 1), (0, 2), (1, 1), (1, 2)]}) == 4
    assert (rgb[0, 0] == 0).all() and norm is not None
    np.testing.assert_array_equal(values, before)

    def reject_exp(*args, **kwargs):
        raise AssertionError("Log rendering must not exponentiate matrix values")

    monkeypatch.setattr(np, "exp", reject_exp)
    image = plot_heatmap(values, log_input=True)
    assert image.dtype == np.uint8 and image.shape[2] == 3


def test_log_padding_and_ticks_do_not_materialize_tiny_probabilities():
    values = -np.arange(1, 10, dtype=float).reshape(3, 3)
    actual = _aggregate_matrix_max(values, (2, 2), pad_value=-np.inf)
    np.testing.assert_array_equal(actual, [[-1, -3], [-7, -9]])
    _, labels = _log_colorbar_ticks(-1000 * math.log(10), 0, 3, True)
    assert labels == ["10^-998%", "10^-498%", "10^2%"]


def test_colour_clipping_preserves_scores_and_ignores_extreme_outlier():
    values = np.array([[-5.0, -10.0, -15.0, -500.0]], dtype=np.float32)
    before = values.copy()
    image, norm, low, high = _generate_log_heatmap_rgb_array(values, None, "magma", True, log_range=(-20.0, 0.0))
    reference, _, _, _ = _generate_log_heatmap_rgb_array(values[:, :3], None, "magma", True, log_range=(-20.0, 0.0))
    np.testing.assert_array_equal(image[:, :3], reference)
    np.testing.assert_array_equal(values, before)
    assert (low, high) == (-20, 0) and norm.clip
    assert len({tuple(colour) for colour in image[0, :3]}) == 3
    with pytest.raises(ValueError, match="increasing"):
        plot_heatmap(values, log_input=True, log_range=(0.0, -10.0))
