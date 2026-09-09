import json
import random

import numpy as np
import pytest
import torch
from PIL import Image

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.benchmarks.inference.prepare_inputs import ordered_classes, prepare, repository_preprocess, seeded, select_records


def identity_factory(metadata):
    return lambda images: images.float() / 255


@pytest.fixture
def example(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    records = []
    for i, split in enumerate(["train"] * 4 + ["val"] * 3 + ["test"] * 2):
        path = root / f"{i}.png"
        Image.fromarray(np.full((3, 4, 3), i * 20, dtype=np.uint8)).save(path)
        records.append({"path": path.name, "sha256": file_hash(path), "split": split, "targets": [i % 2, i % 2]})
    classes = {"a": 0, "b": 1}
    dataset = {"schema_version": 1, "class_spec": {"cls2idx": {"0": classes, "1": classes}}, "records": records}
    export = {
        "schema_version": 1,
        "source": {"checkpoint_sha256": "fixture"},
        "input": {"name": "images", "dtype": "float32", "shape": ["batch", 3, 2, 2]},
        "outputs": [{"name": "scores"}],
        "classifiers": [{"module": "head", "metadata": {"cls2idx": classes, "resize_size": 2, "backbone_class": "unused"}}],
        "preprocessing": {"in_graph": False},
    }
    dataset_path, export_path = tmp_path / "dataset.json", tmp_path / "export.json"
    dataset_path.write_text(json.dumps(dataset))
    export_path.write_text(json.dumps(export))
    return dataset_path, root, export_path, dataset, export


def test_preparation_repeats_exact_bytes_and_publishes_both_contracts(example, tmp_path):
    dataset, root, export, _, _ = example
    hashes = []
    for index in range(2):
        out = tmp_path / f"train-{index}"
        report = prepare(
            dataset, root, export, out, "train", count=3, batch_size=2, preprocess_factory=identity_factory, score_semantics="logits"
        )
        assert report["status"] == "prepared"
        assert report["settings"]["workers"] == 0
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["split"] == "train"
        hashes.append([b["sha256"] for b in manifest["batches"]])
        assert [b["sample_ids"] for b in manifest["batches"]] == [["0", "1"], ["2"]]
    assert hashes[0] == hashes[1]
    result = prepare(
        dataset, root, export, tmp_path / "val", "val", batch_size=2, preprocess_factory=identity_factory, score_semantics="logits"
    )
    assert result["selection"]["method"] == "manifest order"
    with np.load(tmp_path / "val/batch-00000.npz") as arrays:
        np.testing.assert_array_equal(arrays["images"][:, 0, 0, 0], np.array([80, 100], dtype=np.float32) / 255)
    with pytest.raises(FileExistsError):
        prepare(dataset, root, export, tmp_path / "val", "val", preprocess_factory=identity_factory, score_semantics="logits")


def test_source_hash_failure_is_retained_before_any_batches(example, tmp_path):
    dataset, root, export, _, _ = example
    (root / "4.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source image hash mismatch"):
        prepare(dataset, root, export, tmp_path / "failed", "val", preprocess_factory=identity_factory, score_semantics="logits")
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed" and not report["batches"]
    assert not (tmp_path / "failed/manifest.json").exists()


def test_cross_split_duplicate_bytes_are_rejected(example):
    dataset = example[3]
    dataset["records"][4]["sha256"] = dataset["records"][0]["sha256"]
    with pytest.raises(ValueError, match="across splits"):
        select_records(dataset, "train", None, 42)


def test_class_order_mismatch_and_implicit_semantics_are_rejected(example, tmp_path):
    dataset, root, export, _, metadata = example
    with pytest.raises(ValueError, match="semantics explicitly"):
        prepare(dataset, root, export, tmp_path / "implicit", "val")
    metadata["classifiers"][0]["metadata"]["cls2idx"] = {"b": 0, "a": 1}
    export.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="class ordering"):
        prepare(dataset, root, export, tmp_path / "mismatch", "val", score_semantics="logits")


@pytest.mark.parametrize("mapping", [{"a": 1}, {"a": 0, "b": 0}, {"a": True}, {"1": {"a": 0}}])
def test_invalid_class_indices(mapping):
    with pytest.raises(ValueError):
        ordered_classes(mapping)


def test_factory_uses_existing_loader_without_building_trained_head(monkeypatch):
    from mini_trainer.modeling.architectures import load

    calls = []
    sentinel = object()

    def get_model(name, **kwargs):
        calls.append((name, kwargs))
        return object(), "head", sentinel, 123, 224

    monkeypatch.setattr(load, "get_model", get_model)
    result = repository_preprocess(
        {"backbone_class": "efficientnet_v2_s", "resize_size": 128, "preprocess_dtype": "float32", "out_features": 1000000}
    )
    assert result is sentinel
    assert calls == [
        (
            "efficientnet_v2_s",
            {"model_args": {"pretrained": False, "local_files_only": True, "resize_size": 128}, "preprocess_dtype": torch.float32},
        )
    ]


def test_rng_context_is_repeatable_and_restores_callers():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    values = []
    for _ in range(2):
        with seeded(42):
            values.append((random.random(), np.random.rand(), torch.rand(()).item()))
    assert values[0] == values[1]
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_historical_preprocessing_factory_remains_importable():
    from dev.benchmarks.prepare_inputs import repository_preprocess as historical_factory

    assert historical_factory is repository_preprocess
