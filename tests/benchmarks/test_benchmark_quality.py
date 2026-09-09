import csv
import json

import pytest

from dev.benchmarks.inference.quality_compare import COLUMNS, METRICS, compare, read_manifest, read_predictions


@pytest.fixture
def example(tmp_path):
    classes = [["001", "1"], ["alpha", "beta"]]
    labels = [["001", "alpha"], ["001", "alpha"], ["1", "beta"], ["1", "beta"]]
    samples = [{"instance_id": i, "filename": f"image-{i}.jpg", "labels": labs} for i, labs in enumerate(labels)]
    metadata = {
        "schema_version": 1,
        "split": "val",
        "provenance": {"dataset": "oracle fixture"},
        "levels": [{"name": name, "classes": cs} for name, cs in zip(("leaf", "parent"), classes, strict=True)],
        "samples": samples,
    }
    for mode in ("baseline", "candidate"):
        path = tmp_path / f"{mode}.csv"
        metadata[mode] = {"path": path.name, "classes": classes, "provenance": {"model": mode}}
        rows = []
        for level in range(2):
            for sample in samples:
                prediction = sample["labels"][level]
                if mode == "candidate" and level == 0 and sample["instance_id"] == 1:
                    prediction = "1"
                rows.append(
                    dict(
                        instance_id=sample["instance_id"],
                        filename=sample["filename"],
                        level=level,
                        label=sample["labels"][level],
                        prediction=prediction,
                        confidence=0.8,
                        threshold=0,
                    )
                )
        if mode == "candidate":
            rows.reverse()
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(metadata))
    return manifest, metadata


@pytest.mark.parametrize("kind", ["train", "mapping", "duplicate"])
def test_manifest_rejects_invalid_evaluation_contract(example, kind):
    path, metadata = example
    if kind == "train":
        metadata["split"] = "train"
    elif kind == "mapping":
        metadata["candidate"]["classes"] = [["1", "001"], ["alpha", "beta"]]
    else:
        metadata["samples"].append(metadata["samples"][0])
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        read_manifest(path)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "label", "filename", "class", "threshold", "nonfinite", "hash"])
def test_predictions_reject_mismatched_samples_and_policy(example, kind):
    path, metadata = example
    artifact = metadata["candidate"]
    source = path.parent / artifact["path"]
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[0])
    elif kind in ("label", "filename"):
        rows[0][kind] = "changed"
    elif kind == "class":
        rows[0]["prediction"] = "not a class"
    elif kind == "threshold":
        rows[0]["threshold"] = "0.5"
    elif kind == "nonfinite":
        rows[0]["confidence"] = "nan"
    else:
        artifact["sha256"] = "wrong"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_predictions(source, artifact, metadata)


def test_real_metrics_preserve_literal_labels_and_pair_reordered_rows(example, tmp_path, monkeypatch):
    metrics = pytest.importorskip("mini_metrics.metrics")
    path, _ = example
    original = metrics.evaluate_file

    def evaluate(source, **kwargs):
        assert kwargs["opt_crit"] is metrics.MacroF1
        assert kwargs["optimal"] is False and kwargs["threshold"] == 0
        return original(source, **kwargs)

    monkeypatch.setattr(metrics, "evaluate_file", evaluate)
    report = compare(path, tmp_path / "result")
    assert report["status"] == "evaluated" and not report["undefined_metrics"]
    assert report == json.loads((tmp_path / "result/report.json").read_text())
    baseline = report["models"]["baseline"]["metrics"]
    candidate = report["models"]["candidate"]["metrics"]
    assert set(baseline) == set(METRICS)
    assert all(value == pytest.approx(1) for levels in baseline.values() for value in levels.values())
    assert candidate["f1"]["0"] == pytest.approx(11 / 15)
    assert candidate["recall"]["0"] == pytest.approx(0.75)
    assert candidate["precision"]["0"] == pytest.approx(5 / 6)
    assert candidate["coverage"]["0"] == 1
    assert candidate["theilU"]["0"] == pytest.approx(0.31127812445913283)
    assert report["levels"][0]["prediction_changes"] == 1
    assert report["levels"][1]["prediction_changes"] == 0
    with pytest.raises(FileExistsError):
        compare(path, tmp_path / "result")


def test_invalid_candidate_retains_baseline_evaluation(example, tmp_path):
    pytest.importorskip("mini_metrics")
    path, metadata = example
    metadata["candidate"]["sha256"] = "changed"
    path.write_text(json.dumps(metadata))
    output = tmp_path / "failure"
    with pytest.raises(ValueError, match="hash mismatch"):
        compare(path, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["models"]["baseline"]["metrics"]


def test_undefined_theil_u_is_explicit_null_not_invalid_json(example, tmp_path):
    pytest.importorskip("mini_metrics")
    path, metadata = example
    metadata["samples"] = metadata["samples"][:1]
    path.write_text(json.dumps(metadata))
    for mode in ("baseline", "candidate"):
        source = path.parent / metadata[mode]["path"]
        with source.open() as stream:
            rows = [r for r in csv.DictReader(stream) if r["instance_id"] == "0"]
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    report = compare(path, tmp_path / "undefined")
    assert report["models"]["baseline"]["metrics"]["theilU"]["0"] is None
    assert report["levels"][0]["candidate_minus_baseline"]["theilU"] is None
    assert len(report["undefined_metrics"]) == 4
