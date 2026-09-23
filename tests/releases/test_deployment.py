"""Small portable inference contracts, without downloading models or importing ORT."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from deployment.mambo_deploy import Predictor
from deployment.mambo_deploy.bundle import Bundle
from deployment.mambo_deploy.preprocessing import RECIPE, preprocess
from deployment.mambo_deploy.results import Prediction, hierarchy

CLASSES = {"labels": [["a", "b", "c"], ["g0", "g1"], ["f0"]], "parents": [[0, 0, 1], [0, 0]]}


@pytest.fixture
def bundle(tmp_path):
    payloads = {
        "classes.json": json.dumps(CLASSES),
        "preprocessing.json": json.dumps(RECIPE),
        "presets.json": json.dumps({"europe": {"path": "europe.classes", "count": 2}}),
        "europe.classes": "a\nb\n",
    }
    manifest = {
        "schema": "mambo-release-v1",
        "model_id": "fixture",
        "score_semantics": "hierarchical-leaf-logits-logsumexp-v1",
        "files": {},
    }
    for name, value in payloads.items():
        data = value.encode()
        (tmp_path / name).write_bytes(data)
        manifest["files"][name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (tmp_path / "release.json").write_text(json.dumps(manifest))
    return tmp_path


def test_mask_recomputes_parent_scores_and_normalization():
    leaf = np.log(np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
    raw, labels, indices = hierarchy(leaf, [0, 2], CLASSES)
    prediction = Prediction(raw, labels, indices)
    assert prediction[0].label == ("c", "g1", "f0")
    np.testing.assert_allclose(prediction[0].confidence, [0.875, 0.875, 1], rtol=1e-6)
    assert prediction[0].index == (1, 1, 0)
    assert prediction.global_indices.tolist() == [[[2, 1, 0]]]


def test_tiny_legacy_top1_fixture():
    import tomllib

    fixture = tomllib.loads((Path(__file__).parents[2] / "dev/releases/mambo_v3/compatibility.toml").read_text())["top1"]
    raw = [np.array([row], dtype=np.float32) for row in fixture["logits"]]
    prediction = Prediction(raw, fixture["classes"], [np.arange(len(names)) for names in fixture["classes"]])
    assert prediction[0].label == tuple(fixture["labels"])
    assert prediction[0].index == tuple(fixture["indices"])
    np.testing.assert_allclose(prediction[0].confidence, fixture["confidence"], rtol=1e-6)
    assert prediction.indices.shape == tuple(fixture["array_shape"])


def test_custom_list_replaces_preset_and_predictors_are_isolated(bundle):
    first = Predictor(bundle, class_list=["c", "c", "a"])
    second = Predictor(bundle)
    assert first.class_list == ["a", "c"]
    assert second.class_list == ["a", "b"]
    first._apply_class_mask(-1)
    assert first.class_list == ["a", "b", "c"]
    assert second.class_list == ["a", "b"]
    with pytest.raises(ValueError, match="Unknown species"):
        Predictor(bundle, class_list=["unknown"])
    with pytest.raises(ValueError, match="empty"):
        Predictor(bundle, class_list=[])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Predictor(bundle, class_list=["a"], class_mask=[0])


def test_hash_failure_and_escape_are_rejected(bundle):
    loaded = Bundle(bundle)
    with pytest.raises(ValueError, match="escapes"):
        loaded.file("../outside")
    (bundle / "europe.classes").write_text("c\nb\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        Predictor(bundle)


def test_uint8_and_float_inputs_align_and_reject_hwc():
    image = np.arange(3 * 21 * 30, dtype=np.uint8).reshape(3, 21, 30)
    np.testing.assert_array_equal(preprocess(image), preprocess(image.astype(np.float32) / 255))
    assert preprocess(image[:1]).shape == (3, 384, 384)
    with pytest.raises(ValueError, match="CHW"):
        preprocess(image.transpose(1, 2, 0))
    with pytest.raises(ValueError, match="finite"):
        preprocess(np.full_like(image, np.nan, dtype=np.float32))


def test_batches_embeddings_and_masked_output_contract(bundle, monkeypatch):
    predictor = Predictor(bundle, batch_size=2, class_list=["c"])
    seen = []

    def fake_backend(images, embeddings):
        seen.append(len(images))
        return np.tile([1.0, 2.0, 3.0], (len(images), 1)), np.ones((len(images), 1280), dtype=np.float32) if embeddings else None

    monkeypatch.setattr(predictor, "_onnx", fake_backend)
    images = [np.zeros((3, 4, 5), dtype=np.uint8)] * 5
    result, vectors = predictor.predict_with_embeddings(images)
    assert seen == [2, 2, 1]
    assert len(result) == 5 and vectors.shape == (5, 1280)
    assert result[0].label == ("c", "g1", "f0")
    assert result[0].confidence == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="No images"):
        predictor.predict([])


def test_topk_serialization_handles_nested_items(tmp_path):
    classes = {"labels": [["a", "b"], ["g0", "g1"], ["f0", "f1"]], "parents": [[0, 1], [0, 1]]}
    raw, labels, indices = hierarchy(np.array([[1.0, 2.0]], dtype=np.float32), [0, 1], classes)
    result = Prediction(raw, labels, indices, topk=2)
    result.save(tmp_path / "result.json")
    assert json.loads((tmp_path / "result.json").read_text())["results"][0][0]["label"] == ["b", "g1", "f1"]
    with pytest.raises(ValueError, match="smallest retained rank"):
        Prediction(raw, labels, indices, topk=3)


def test_native_facade_preserves_container_and_shared_confidence(bundle, monkeypatch):
    import torch

    from mini_trainer import deploy
    from mini_trainer.hierarchical.model import HierarchicalPrediction

    monkeypatch.setattr(deploy, "_runtime", lambda: Predictor)
    facade = deploy.Predictor(device="cpu", bundle=bundle)
    # Nonnegative logits summing to one must still be treated as logits.
    raw = [np.array([[0.2, 0.8]], dtype=np.float32), np.array([[1.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32)]
    result = Prediction(raw, [["a", "b"], ["g0"], ["f0"]], [np.arange(2), np.arange(1), np.arange(1)])
    monkeypatch.setattr(facade._predictor, "predict", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        facade._predictor, "predict_with_embeddings", lambda *args, **kwargs: (result, np.ones((1, 1280), dtype=np.float32))
    )
    native = facade("unused")
    assert isinstance(native, HierarchicalPrediction)
    assert native[0].label == result[0].label
    np.testing.assert_array_equal(native.confidence.numpy(), result.confidence)
    assert isinstance(native.indices, torch.Tensor)
    assert isinstance(facade.predict_with_embeddings("unused")[1], torch.Tensor)
