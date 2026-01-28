import numpy as np

from mini_trainer.utils.plot import _aggregate_matrix_max, _get_scaled_matrix_for_display, raw_confusion_matrix


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
    mat = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
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
