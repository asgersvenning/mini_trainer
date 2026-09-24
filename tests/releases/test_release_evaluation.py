"""Release identity, excluded-truth and metric serialization contracts."""

import csv
import json

import numpy as np
import pytest

from deployment.mambo_deploy.results import Prediction, hierarchy
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, canonical_rows, load_records, prepare_flemming
from dev.releases.mambo_v3.metrics import finite_json


def fixture_data(tmp_path):
    root = tmp_path / "images"
    (root / "outside").mkdir(parents=True)
    (root / "outside/photo.jpg").write_bytes(b"image fixture")
    reference = tmp_path / "truth.csv"
    with reference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for rank, truth in enumerate(("outside", "genus", "family")):
            writer.writerow([0, "/old/root/outside/photo.jpg", rank, truth, "unused", 0.5, 0, 0, 1, -1])
    return root, reference


def test_archive_truth_join_preserves_unknown_species_and_hashes(tmp_path):
    root, reference = fixture_data(tmp_path)
    output = tmp_path / "manifest.json"
    data = prepare_flemming(root, reference, output)
    assert data["records"][0]["labels"] == ["outside", "genus", "family"]
    assert len(data["records"][0]["sha256"]) == 64
    _, records = load_records(output, root, 1)
    assert records == data["records"]
    (root / "outside/extra.jpg").write_bytes(b"extra")
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_flemming(root, reference, tmp_path / "other.json")


def test_manifest_escape_is_rejected(tmp_path):
    root, reference = fixture_data(tmp_path)
    output = tmp_path / "manifest.json"
    data = prepare_flemming(root, reference, output)
    data["records"][0]["path"] = "../truth.csv"
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsafe image"):
        load_records(output, root)


def test_canonical_rows_retain_excluded_truth_with_known_parent():
    classes = {"labels": [["inside"], ["genus"], ["family"]], "parents": [[0], [0]]}
    result = Prediction(*hierarchy(np.array([[1.0]], dtype=np.float32), [0], classes))
    records = [{"path": "outside/photo.jpg", "labels": ["outside", "genus", "family"]}]
    rows = [dict(zip(CSV_COLUMNS, row, strict=True)) for row in canonical_rows(records, result, 13)]
    assert len(rows) == 3 and rows[0]["instance_id"] == 13
    assert rows[0]["label"] == "outside" and rows[0]["known_label"] == 0 and rows[0]["correct"] == -1
    assert rows[1]["known_label"] == 1 and rows[1]["correct"] == 1
    assert all(row["prediction_made"] == 1 and row["threshold"] == 0 for row in rows)
    with pytest.raises(ValueError, match="count mismatch"):
        list(canonical_rows([], result))


def test_metric_json_keeps_undefined_values_explicit():
    assert finite_json({0: np.nan, 1: (np.float64(0.5), np.inf)}) == {"0": None, "1": [0.5, None]}


def test_comparison_rejects_changed_truth(tmp_path):
    from dev.releases.mambo_v3.compare_quality import compare

    for name, truth in (("left", "a"), ("right", "b")):
        directory = tmp_path / name
        (directory / "full").mkdir(parents=True)
        (directory / "report.json").write_text(json.dumps({"status": "complete", "sample_ids_sha256": "same"}))
        with (directory / "full/mini_metric.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            writer.writerow([0, "image.jpg", 0, truth, "a", 0.5, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="Ground truth"):
        compare(tmp_path / "left", tmp_path / "right", ["full"])


def test_ucloud_recovery_checks_original_split_and_taxonomy(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dev.releases.mambo_v3 import prepare_ucloud
    from dev.releases.mambo_v3.evaluation_data import write_json

    metadata = tmp_path / "metadata.parquet"
    pq.write_table(
        pa.table(
            {
                "filename": ["test.jpg", "train.jpg"],
                "set": ["0", "1"],
                "speciesKey": ["a", "b"],
                "genusKey": ["g", "g"],
                "familyKey": ["f", "f"],
            }
        ),
        metadata,
    )
    monkeypatch.setattr(prepare_ucloud, "HERE", tmp_path)
    (tmp_path / "construction.toml").write_text(f'[source]\nsha256 = "{prepare_ucloud.file_hash(metadata)}"\n')
    staging, reference = tmp_path / "staging.json", tmp_path / "truth.csv"
    write_json(staging, {"files": [{"source": "/work/global_lepi/images/a/test.jpg", "staged": "/staged/0/1.jpg"}]})
    with reference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "level", "label"])
        writer.writerows([["/staged/0/1.jpg", i, label] for i, label in enumerate(["a", "g", "f"])])
    records = prepare_ucloud.recover(metadata, staging, reference)
    assert records == [{"path": "images/a/test.jpg", "labels": ["a", "g", "f"], "split": "test"}]
    with pytest.raises(ValueError, match="membership differs"):
        prepare_ucloud.recover(metadata, staging, reference, "1")


def test_threaded_preprocessing_is_byte_identical_and_ordered(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    from dev.releases.mambo_v3.evaluate import prepare_batch

    paths = []
    for i in range(5):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (11 + i, 13 + i), (i * 30, i * 10, 255 - i * 20)).save(path)
        paths.append(path)
    serial = prepare_batch(paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        threaded = prepare_batch(paths, pool)
    np.testing.assert_array_equal(serial, threaded)
    assert not np.array_equal(threaded[0], threaded[-1])


def test_loading_observer_preserves_classmethod_and_restores_it():
    import torch

    from dev.releases.mambo_v3.benchmark import observe_loading
    from mini_trainer.modeling.classifier import Classifier

    original = Classifier.init_spherical_repulsion.__func__
    values = {}
    with observe_loading("torch", values):
        layer = torch.nn.Linear(2, 3)
        assert Classifier.init_spherical_repulsion(layer, iterations=1) is layer
    assert Classifier.init_spherical_repulsion.__func__ is original
    assert values["spherical_initialization_within_model_build"] >= 0


def test_pinned_metrics_distinguish_micro_macro_and_known_truth(tmp_path):
    import os
    import subprocess

    executable = os.environ.get("MAMBO_METRICS_PYTHON")
    if not executable:
        pytest.skip("Set MAMBO_METRICS_PYTHON to the pinned metric environment")
    source = tmp_path / "mini_metric.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for i, truth in enumerate(("a", "a", "a", "outside")):
            for rank in range(3):
                writer.writerow([i, f"{i}.jpg", rank, truth, "a", 0.8, 0, int(truth == "a"), 1, 1 if truth == "a" else -1])
    output = subprocess.check_output(
        [executable, "-m", "dev.releases.mambo_v3.metrics", "--source", str(source), "--output", str(tmp_path / "metrics.json")],
        text=True,
    )
    ranks = json.loads(output)
    for row in ranks.values():
        assert row["micro_accuracy_all"] == pytest.approx(0.75)
        assert row["macro_accuracy_all"] == pytest.approx(0.5)
        assert row["micro_accuracy_known"] == pytest.approx(1.0)
        assert row["abstention_coverage"] == 1
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["all"]["f1"]["0"] == pytest.approx(3 / 7)
    assert metrics["known"]["f1"]["0"] == pytest.approx(1)


def test_family_audit_export_keeps_predicted_only_groups_without_recall(tmp_path):
    from dev.releases.mambo_v3.family_precision_report import export

    # mini_metrics emits no recall group for a family that occurs only in predictions.
    row = {
        "threshold": 0.95,
        "truth_counts": {"present": 2},
        "predicted_counts": {"present": 1, "absent": 1},
        "groups": {
            "precision": {"present": [1, 1], "absent": [0, 1]},
            "recall": {"present": [0.5, 1]},
            "f1": {"present": [2 / 3, 1], "absent": [0, 1]},
        },
    }
    data = {"taxonomy": {"names": {"absent": "Absent family"}}, "models": {"v3": {"optimized": row}}}
    export(data, tmp_path)
    with (tmp_path / "mambo-family-precision.csv").open() as stream:
        rows = {r["family_id"]: r for r in csv.DictReader(stream)}
    assert rows["absent"]["truth_images"] == "0"
    assert rows["absent"]["accepted_predictions"] == "1"
    assert rows["absent"]["precision"] == "0"
    assert rows["absent"]["recall"] == ""
    assert rows["absent"]["recall_weight"] == "0"
    assert rows["absent"]["f1_weight"] == "1"
