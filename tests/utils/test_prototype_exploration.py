"""Offline explorer invariants: checkpoint identity, self-pairs and tied scores."""

import base64
import math

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("scipy")

from dev.prototype_space.explore import analyze, block_extrema, load_prototypes, stable_scores, synthetic_cases, weight_model
from mini_trainer.modeling import Classifier, class_distance, class_similarity


def test_extraction_preserves_magnitudes_bias_and_checkpoint_class_order(tmp_path):
    layer = Classifier._normalize_layer(nn.Linear(4, 3), orthogonal_init=False)
    with torch.no_grad():
        layer.parametrizations.weight.original0.copy_(torch.tensor([[2.0], [0.5], [3.0]]))
        layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    state = {f"classifier.linear.{key}": value for key, value in layer.state_dict().items()}
    state["classifier._extra_state"] = {
        "normalized": True,
        "classifier_class": "mini_trainer.hierarchical.model:HierarchicalClassifier",
        "cls2idx": {"0": {"last": 2, "first": 0, "middle": 1}},
        "labels": {"first": ["first", "parent"]},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(state, path)
    before = path.read_bytes()
    weight, names, groups, meta = load_prototypes(path)
    torch.testing.assert_close(weight, layer.weight, rtol=0, atol=0)
    assert names == ["first", "middle", "last"]
    assert groups[0] == ["first", "parent"]
    assert meta["bias_range"] == [layer.bias.min().item(), layer.bias.max().item()]
    assert path.read_bytes() == before
    state["classifier._extra_state"]["classifier_class"] = "unsupported:MultiPrototypeHead"
    torch.save(state, path)
    with pytest.raises(ValueError, match="supports only"):
        load_prototypes(path)


def test_block_summaries_exclude_diagonal_and_partial_padding():
    matrix = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.6], [0.8, 0.6, 0.0]])
    low, high, block = block_extrema(matrix, np.array([2, 0, 1]), size=2)
    assert block == 2
    np.testing.assert_allclose(low, [[0.8, 0.2], [0.2, np.nan]])
    np.testing.assert_allclose(high, [[0.8, 0.6], [0.6, np.nan]])
    assert np.diag(matrix).tolist() == [0.0, 0.0, 0.0]


def test_algebraic_cases_keep_baseline_ties_and_original_submatrix_values():
    weight = synthetic_cases(1280)["Algebraic edge cases"]
    model = weight_model(weight)
    distance = class_distance(model)[0].numpy()
    z = class_similarity(model, cdf=False)[0].numpy()
    # An actual duplicate and a near duplicate both saturate the CDF, but their
    # pre-CDF scores remain distinct. They must not be presented as equal vectors.
    assert distance[0, 1] == distance[0, 5] == 0
    assert z[0, 1] > z[0, 5] > z[0, 3] > z[0, 2]
    names = [f"class-{i}" for i in range(len(weight))]
    result = analyze(weight, names, [[name] for name in names], neighbours=3)
    assert result["neighbours"][0][:2] == [1, 5]
    for i, ranks in enumerate(result["profile_neighbours"]):
        row = z[i].copy()
        row[i] = -np.inf
        expected = np.argsort(-row, kind="stable")[np.asarray(result["profile_ranks"]) - 1]
        assert ranks == expected.tolist()
    assert all(i not in row for i, row in enumerate(result["neighbours"]))
    assert result["zero_count"][0] == 2
    assert result["stats"]["pair_sample_count"] == 15
    for field, original in [("local_distance", distance), ("local_z", z)]:
        actual = np.frombuffer(base64.b64decode(result[field]), dtype="<f4").reshape(6, 4, 4)
        for i, row in enumerate(result["neighbours"]):
            ids = [i, *row]
            np.testing.assert_array_equal(actual[i], original[np.ix_(ids, ids)])


def test_log_domain_scores_preserve_tails_against_independent_erfc_reference():
    values = [-8.0, -1.0, 0.0, 1.0, 8.0, 11.0, 31.0]
    distance, logtail = stable_scores(torch.tensor(values))
    expected_distance = [
        -math.log(0.5 * math.erfc(-z / math.sqrt(2))) if z <= 0 else -math.log1p(-0.5 * math.erfc(z / math.sqrt(2))) for z in values
    ]
    expected_tail = [math.log10(0.5 * math.erfc(z / math.sqrt(2))) for z in values]
    np.testing.assert_allclose(distance, expected_distance, rtol=2e-12, atol=0)
    np.testing.assert_allclose(logtail, expected_tail, rtol=2e-12, atol=1e-14)
    assert distance.dtype == torch.float64 and (distance > 0).all()
    # Beyond the float64 linear-probability range, log-tail still has meaning.
    extreme_d, extreme_logtail = stable_scores(torch.tensor([56.0]))
    assert extreme_d.item() == 0 and torch.isfinite(extreme_logtail).all()


def test_optional_gbif_snapshot_is_embedded_as_data(tmp_path):
    import json

    from mini_trainer.visualization.prototype_space.explore import render_report

    render_report(tmp_path, "{}")
    assert '<script id="gbif-snapshot" type="application/json">null</script>' in (tmp_path / "explorer.html").read_text()
    snapshot = {"schema": "mini-trainer-gbif-v1", "taxa": {"1": {"key": 1, "canonicalName": "</script><script>unsafe</script>"}}}
    (tmp_path / "gbif-snapshot.json").write_text(json.dumps(snapshot))
    render_report(tmp_path, "{}")
    html = (tmp_path / "explorer.html").read_text()
    assert "</script><script>unsafe" not in html
    assert "\\u003c/script>" in html
    assert "__GBIF_SCRIPT__" not in html
