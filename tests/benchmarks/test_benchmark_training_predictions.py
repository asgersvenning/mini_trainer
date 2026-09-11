import json

import numpy as np
import pytest

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.benchmarks.inference.quality_compare import read_manifest, read_predictions
from dev.benchmarks.training.training_predictions import prepare


def make_run(path, *, hierarchical=False, synthetic=False, reverse=False):
    path.mkdir()
    levels = 3 if hierarchical else 1
    records = [
        {"path": "b.jpg", "split": "test", "label": 0, "targets": [0] * levels, "sha256": "a" * 64},
        {"path": "a.jpg", "split": "test", "label": 1, "targets": [1] * levels, "sha256": "b" * 64},
    ]
    if reverse:
        records.reverse()
    inventory_path = path / ("data/manifest.json" if synthetic else "dataset_manifest.json")
    inventory_path.parent.mkdir(exist_ok=True)
    inventory_path.write_text(json.dumps({"records": records}))
    mapping = {"001": 0, "1": 1}
    report = {
        "schema_version": 1,
        "status": "failed" if synthetic else "completed",
        "coverage": {"training": True, "checkpoint_reload": True, "inference": True},
        "test_used_for_training_or_selection": False,
        "score_semantics": "model_eval_forward",
        "head": "hierarchical" if hierarchical else "flat",
        "dataset": "synthetic" if synthetic else "blair",
        "class_mapping": {str(i): mapping for i in range(levels)} if hierarchical else mapping,
        "dataset_manifest_sha256": file_hash(inventory_path),
        "checkpoint_sha256": "c" * 64,
    }
    (path / "report.json").write_text(json.dumps(report))
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    scores = np.eye(2, dtype=np.float32)[labels] * 1000 - 500
    arrays = {"paths": np.array([r["path"] for r in records]), "scores": scores, "labels": labels}
    for i in range(levels):
        arrays[f"scores_{i}"] = scores
        arrays[f"labels_{i}"] = labels
    np.savez(path / "predictions.npz", **arrays)
    return path


@pytest.mark.parametrize("hierarchical,synthetic", [(False, False), (True, False), (False, True)])
def test_adapter_canonicalizes_identity_and_preserves_literal_labels(tmp_path, hierarchical, synthetic):
    a = make_run(tmp_path / "a", hierarchical=hierarchical, synthetic=synthetic)
    b = make_run(tmp_path / "b", hierarchical=hierarchical, synthetic=synthetic, reverse=True)
    before = {p: file_hash(p) for folder in (a, b) for p in folder.rglob("*") if p.is_file()}
    manifest = prepare(a, b, tmp_path / "out")
    metadata, _ = read_manifest(manifest)
    assert [s["filename"] for s in metadata["samples"]] == ["a.jpg", "b.jpg"]
    assert len(metadata["levels"]) == (3 if hierarchical else 1)
    tables = [read_predictions(manifest.parent / metadata[m]["path"], metadata[m], metadata)[0] for m in ("baseline", "candidate")]
    assert tables[0] == tables[1]
    assert set(tables[0]["prediction"]) == {"001", "1"}
    assert tables[0]["confidence"] == [1.0] * len(tables[0]["confidence"])
    assert {p: file_hash(p) for p in before} == before
    # Results bundles can be transported without checkpoints or source images.
    assert not (a / "training").exists()
    with pytest.raises(FileExistsError):
        prepare(a, b, tmp_path / "out")


@pytest.mark.parametrize(
    "corruption",
    [
        "labels",
        "nan",
        "aliases",
        "paths",
        "extra_level",
        "class_indices",
        "class_order",
        "checkpoint",
        "manifest_hash",
        "incomplete",
        "image_hash",
        "targets",
        "identity",
        "target_identity",
    ],
)
def test_adapter_rejects_inconsistent_evidence_before_writing(tmp_path, corruption):
    a, b = make_run(tmp_path / "a"), make_run(tmp_path / "b")
    report_path = b / "report.json"
    report = json.loads(report_path.read_text())
    with np.load(b / "predictions.npz") as archive:
        arrays = dict(archive)
    if corruption == "labels":
        arrays["labels"] = arrays["labels_0"] = np.array([1, 0])
    elif corruption == "nan":
        arrays["scores"] = arrays["scores_0"] = np.full((2, 2), np.nan)
    elif corruption == "aliases":
        arrays["scores"] = -arrays["scores_0"]
    elif corruption == "paths":
        arrays["paths"] = arrays["paths"][::-1]
    elif corruption == "extra_level":
        arrays["scores_1"] = arrays["scores"]
    elif corruption == "class_indices":
        report["class_mapping"] = {"001": 1, "1": 1}
    elif corruption == "class_order":
        report["class_mapping"] = {"001": 1, "1": 0}
    elif corruption == "checkpoint":
        del report["checkpoint_sha256"]
    elif corruption == "manifest_hash":
        report["dataset_manifest_sha256"] = "0" * 64
    elif corruption == "incomplete":
        report["coverage"]["inference"] = False
    else:
        source = b / "dataset_manifest.json"
        inventory = json.loads(source.read_text())
        if corruption == "image_hash":
            inventory["records"][0]["sha256"] = "d" * 64
        elif corruption == "identity":
            inventory["records"][0]["path"] = "z.jpg"
            arrays["paths"] = np.array(["z.jpg", "a.jpg"])
        elif corruption == "target_identity":
            inventory["records"][0]["label"] = 1
            arrays["labels"] = arrays["labels_0"] = np.array([1, 1])
        else:
            inventory["records"][0]["label"] = 2
        source.write_text(json.dumps(inventory))
        report["dataset_manifest_sha256"] = file_hash(source)
    np.savez(b / "predictions.npz", **arrays)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        prepare(a, b, tmp_path / "out")
    assert not (tmp_path / "out").exists()
