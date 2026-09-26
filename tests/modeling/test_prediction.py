import json
from contextlib import nullcontext

import pytest
import torch

from mini_trainer.modeling import Prediction


@pytest.mark.parametrize("mapping", [None, {}, {"cat": 1, "bird": 2, "dog": 0}])
@pytest.mark.parametrize("topk", [1, 2])
def test_prediction_labels_follow_score_indices(mapping, topk, tmp_path):
    logits = torch.tensor([[-1.0, 3.0, 1.0], [2.0, -3.0, 0.0]])
    with pytest.warns(UserWarning, match="experimental") if topk == 2 else nullcontext():
        result = Prediction(logits, cls2idx=mapping, topk=topk)
    labels = [["cat", "bird"], ["dog", "bird"]] if mapping else [["1", "2"], ["0", "2"]]
    assert result.labels == [row[:topk] for row in labels]
    assert result.indices.tolist() == [row[:topk] for row in [[1, 2], [0, 2]]]
    torch.testing.assert_close(result.confidence, logits.softmax(-1).gather(-1, result.indices))
    if topk == 1:
        path = tmp_path / "predictions.json"
        result.save(path)
        saved = json.loads(path.read_text())
        assert [item["label"] for item in saved["results"]] == [row[0] for row in labels]
        assert [item["index"] for item in saved["results"]] == [1, 0]
