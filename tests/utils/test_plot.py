import numpy as np
import pytest

from mini_trainer.training import raw_confusion_matrix
from mini_trainer.visualization.plot import (
    MIN_DISPLAY_DIM_HEATMAP,
    _aggregate_matrix_max,
    _get_colorbar_ticks_and_labels,
    _get_scaled_matrix_for_display,
)


def test_raw_confusion_matrix():
    cm = raw_confusion_matrix([0, 1, 2, 0], [0, 1, 0, 0], n_classes=3)
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 1, 0], [1, 0, 0]])


def test_aggregate_matrix_max():
    mat = np.arange(1, 17).reshape(4, 4)
    np.testing.assert_array_equal(_aggregate_matrix_max(mat, (2, 2)), [[6, 8], [14, 16]])


def test_get_scaled_matrix_for_display():
    scaled = _get_scaled_matrix_for_display(np.zeros((100, 100)))
    assert min(scaled.shape) >= MIN_DISPLAY_DIM_HEATMAP


@pytest.mark.parametrize("cmap_name", ["magma", "viridis"])
@pytest.mark.parametrize("percent", [True, False])
def test_chunked_heatmap_matches_previous_rgb_pixels(cmap_name, percent):
    import math

    import matplotlib as mpl
    from matplotlib.colors import LogNorm

    from mini_trainer.visualization.plot import _generate_heatmap_rgb_array

    values = np.geomspace(1e-4, 5, 512 * 513).reshape(512, 513)
    values.flat[:4] = [0, np.nan, np.inf, -1]
    original = values.copy()
    masked = np.ma.masked_less_equal(np.ma.masked_invalid(values), 1e-3)
    positive = masked.compressed()
    vmin = 10 ** math.floor(math.log10(min(0.1, positive.min())))
    vmax = 1 if percent else positive.max()
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad((0, 0, 0) if cmap_name == "magma" else (1, 1, 1), alpha=1)
    expected = (cmap(LogNorm(vmin, vmax)(masked))[:, :, :3] * 255).astype(np.uint8)
    actual, _, actual_min, actual_max = _generate_heatmap_rgb_array(values, 1e-3, cmap_name, percent)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, original)
    assert (actual_min, actual_max) == (vmin, vmax)


@pytest.mark.parametrize("limit", [0, 1, 2, 8])
@pytest.mark.parametrize("percent", [False, True])
def test_colorbar_ticks_are_ordered_bounded_and_labeled(limit, percent):
    ticks, labels = _get_colorbar_ticks_and_labels(0.001, 1.0, limit, percent)
    assert ticks == sorted(set(ticks))
    assert len(ticks) == len(labels) == (2 if limit == 0 else limit)
    assert ticks[0] == 0.001
    assert all(0.001 <= tick <= 1.0 for tick in ticks)
    if limit != 1:
        assert ticks[-1] == 1.0
    assert all(label.endswith("%") == percent for label in labels)
