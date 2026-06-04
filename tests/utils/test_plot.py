import importlib.util
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mini_trainer.training import raw_confusion_matrix
from mini_trainer.visualization import plot_probabilistic_dendrogram
from mini_trainer.visualization.plot import (
    _aggregate_matrix_max,
    _get_scaled_matrix_for_display,
)

has_dendrogram_deps = (
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("Bio") is not None
    and importlib.util.find_spec("pycirclize") is not None
)


def test_raw_confusion_matrix():
    # 3 classes: 0, 1, 2
    labels = [0, 1, 2, 0]
    preds = [0, 1, 0, 0]

    # Class 0: 2 instances. Both predicted as 0. Correct: 2. Total: 2.
    # Class 1: 1 instance. Predicted as 1. Correct: 1. Total: 1.
    # Class 2: 1 instance. Predicted as 0. Correct: 0. Total: 1.

    # Confusion matrix (normalized by row/true label)
    # Row 0: 2/2 -> column 0=1.0
    # Row 1: 1/1 -> column 1=1.0
    # Row 2: 1 total. Pred 0. -> column 0=1.0

    cm = raw_confusion_matrix(labels, preds, n_classes=3)

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1.0
    assert cm[1, 1] == 1.0
    assert cm[2, 0] == 1.0
    assert cm[2, 2] == 0.0


def test_aggregate_matrix_max():
    mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # Aggregate 2x2 blocks by max
    # Block (0,0): [[1,2],[5,6]] -> max 6
    # Block (0,1): [[3,4],[7,8]] -> max 8
    # Block (1,0): [[9,10],[13,14]] -> max 14
    # Block (1,1): [[11,12],[15,16]] -> max 16

    agg = _aggregate_matrix_max(mat, (2, 2))
    expected = np.array([[6, 8], [14, 16]])
    np.testing.assert_array_equal(agg, expected)


def test_get_scaled_matrix_for_display():
    # Create large matrix
    mat = np.zeros((100, 100))
    scaled = _get_scaled_matrix_for_display(mat)
    # Should scale up to meet MIN_DISPLAY_DIM_HEATMAP (500)
    # 100 < 500. So upscaled by at least 5x5 blocks.
    assert scaled.shape[0] >= 500
    assert scaled.shape[1] >= 500


@pytest.mark.skipif(not has_dendrogram_deps, reason="Dendrogram dependencies (scipy, biopython, pycirclize) not installed")
def test_plot_probabilistic_dendrogram():
    mock_model = MagicMock()
    mock_model_module = MagicMock()

    # 1. Test flat metadata dictionary
    mock_model_module.metadata = {"idx2cls": {0: "5219173", 1: "2435261", 2: "7429082", 3: "9117798"}}

    dummy = torch.rand(4, 4)
    dummy = dummy + dummy.T  # Make it symmetric
    dummy.fill_diagonal_(0)  # Make diagonal zero

    with patch("mini_trainer.visualization.dendrogram.classification_module", return_value=mock_model_module):
        with patch("mini_trainer.visualization.dendrogram.class_distance", return_value=dummy):
            fig = plot_probabilistic_dendrogram(mock_model)
            assert fig is not None

    # 2. Test hierarchical metadata dictionary (e.g. from JSON)
    mock_model_module.metadata = {
        "cls2idx": {
            "0": {
                "Carnivora - Canidae - dog": 0,
                "Carnivora - Felidae - cat": 1,
                "Rodentia - Muridae - mouse": 2,
                "Perissodactyla - Equidae - horse": 3,
            }
        }
    }

    with patch("mini_trainer.visualization.dendrogram.classification_module", return_value=mock_model_module):
        with patch("mini_trainer.visualization.dendrogram.class_distance", return_value=dummy):
            fig = plot_probabilistic_dendrogram(mock_model)
            assert fig is not None

    # 3. Test pathological hierarchical metadata with unhashable dictionaries directly in idx2cls
    mock_model_module.metadata = {
        "idx2cls": {
            0: {"order": "Carnivora", "species": "dog"},
            1: {"order": "Carnivora", "species": "cat"},
            2: {"order": "Rodentia", "species": "mouse"},
            3: {"order": "Perissodactyla", "species": "horse"},
        }
    }

    with patch("mini_trainer.visualization.dendrogram.classification_module", return_value=mock_model_module):
        with patch("mini_trainer.visualization.dendrogram.class_distance", return_value=dummy):
            fig = plot_probabilistic_dendrogram(mock_model)
            assert fig is not None
