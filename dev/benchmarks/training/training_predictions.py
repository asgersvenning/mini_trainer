"""Prepare paired training-run predictions for the five-metric quality evaluator."""

import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.benchmarks.inference.quality_compare import COLUMNS, read_manifest, read_predictions, validate_dataset


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _read_run(directory):
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    report = json.loads(report_path.read_bytes())
    if (
        report.get("schema_version") != 1
        or report.get("status") not in ("passed", "completed", "failed")
        or not all(report.get("coverage", {}).get(key) is True for key in ("training", "checkpoint_reload", "inference"))
        or report.get("test_used_for_training_or_selection") is not False
        or report.get("score_semantics") != "model_eval_forward"
        or not _is_sha256(report.get("checkpoint_sha256"))
    ):
        raise ValueError("Require a completed training/reload/inference report with unused held-out test predictions")
    mapping = report["class_mapping"]
    if report.get("head") == "flat":
        mappings = [mapping]
    elif report.get("head") == "hierarchical" and set(mapping) == {str(i) for i in range(len(mapping))}:
        mappings = [mapping[str(i)] for i in range(len(mapping))]
    else:
        raise ValueError("Require flat or contiguous hierarchical class mappings")
    classes = []
    for level in mappings:
        if not level or any(type(i) is not int for i in level.values()) or set(level.values()) != set(range(len(level))):
            raise ValueError("Class indices must be unique contiguous integers starting at zero")
        classes.append([name for name, _ in sorted(level.items(), key=lambda item: item[1])])
    inventory_path = directory / ("data/manifest.json" if report.get("dataset") == "synthetic" else "dataset_manifest.json")
    if file_hash(inventory_path) != report.get("dataset_manifest_sha256"):
        raise ValueError("Dataset manifest hash does not match the training report")
    inventory = json.loads(inventory_path.read_bytes())
    records = [record for record in inventory["records"] if record["split"] == "test"]
    paths = [record["path"] for record in records]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Require nonempty, unique held-out image paths")
    samples = []
    image_hashes = {}
    for identifier, record in enumerate(sorted(records, key=lambda record: record["path"])):
        targets = [record["label"]] if len(classes) == 1 else record.get("targets", [])
        if len(targets) != len(classes) or any(type(i) is not int or not 0 <= i < len(c) for i, c in zip(targets, classes, strict=True)):
            raise ValueError("Held-out targets do not match the class mappings")
        digest = record.get("sha256", "")
        if not _is_sha256(digest):
            raise ValueError("Require held-out image SHA256 identifiers")
        image_hashes[record["path"]] = digest
        samples.append(
            {"instance_id": identifier, "filename": record["path"], "labels": [c[i] for c, i in zip(classes, targets, strict=True)]}
        )
    metadata = {
        "schema_version": 1,
        "split": "test",
        "levels": [{"name": "leaf" if i == 0 else f"level_{i}", "classes": c} for i, c in enumerate(classes)],
        "samples": samples,
        "provenance": {"dataset_manifest_sha256": file_hash(inventory_path), "image_sha256": image_hashes},
    }
    validate_dataset(metadata)
    predictions_path = directory / "predictions.npz"
    tables = []
    with np.load(predictions_path, allow_pickle=False) as data:
        if not np.array_equal(data["paths"], paths):
            raise ValueError("Prediction paths differ from the held-out manifest order")
        expected_keys = {f"{key}_{level}" for level in range(len(classes)) for key in ("scores", "labels")}
        if set(data.files) != expected_keys | {"scores", "labels", "paths"}:
            raise ValueError("Prediction archive has missing or unexpected levels/arrays")
        for key in ("scores", "labels"):
            if not np.array_equal(data[key], data[f"{key}_0"]):
                raise ValueError("Leaf prediction aliases disagree")
        indices = {path: i for i, path in enumerate(paths)}
        for level, names in enumerate(classes):
            scores, labels = data[f"scores_{level}"], data[f"labels_{level}"]
            targets = [r["label"] if len(classes) == 1 else r["targets"][level] for r in records]
            if labels.dtype.kind not in "iu" or not np.array_equal(labels, targets):
                raise ValueError("Prediction labels differ from the held-out manifest")
            if scores.dtype.kind != "f" or scores.shape != (len(paths), len(names)):
                raise ValueError("Require finite floating scores with one row per image and column per class")
            # This is a fixed argmax comparison, not a confidence calibration
            # claim. Softmax supplies a bounded confidence for threshold zero.
            for sample in samples:
                row = indices[sample["filename"]]
                values = scores[row].astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Require finite floating scores")
                prediction = int(values.argmax())
                confidence = float(1 / np.exp(values - values[prediction]).sum())
                tables.append(
                    (
                        sample["instance_id"],
                        sample["filename"],
                        level,
                        names[labels[row]],
                        names[prediction],
                        confidence,
                        0.0,
                    )
                )
    provenance = {
        "report_path": str(report_path),
        "report_sha256": file_hash(report_path),
        "predictions_sha256": file_hash(predictions_path),
        "training_report": report,
        "checkpoint_verification": "Checkpoint identifier is taken from the training report; no checkpoint is loaded.",
    }
    return metadata, tables, provenance


def prepare(baseline, candidate, output):
    """Validate both inputs before creating a portable quality-evaluation bundle."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    runs = {mode: _read_run(path) for mode, path in (("baseline", baseline), ("candidate", candidate))}
    metadata = runs["baseline"][0]
    other = runs["candidate"][0]
    if metadata["levels"] != other["levels"] or metadata["samples"] != other["samples"]:
        raise ValueError("Paired runs must have identical held-out identities, labels and ordered class mappings")
    if metadata["provenance"]["image_sha256"] != other["provenance"]["image_sha256"]:
        raise ValueError("Paired held-out image hashes differ")
    metadata["provenance"].update(
        adapter_sha256=file_hash(__file__),
        policy="Fixed argmax, softmax confidence, threshold zero; no threshold tuning or abstention.",
        scope="Saved test predictions only; does not rerun inference, verify source images or establish performance/acceptance.",
    )
    output.mkdir(parents=True)
    for mode, (_, rows, provenance) in runs.items():
        path = output / f"{mode}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        metadata[mode] = {
            "path": path.name,
            "sha256": file_hash(path),
            "classes": [s["classes"] for s in metadata["levels"]],
            "provenance": provenance,
        }
        read_predictions(path, metadata[mode], metadata)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    read_manifest(manifest)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Completed dataset benchmark directory")
    parser.add_argument("--candidate", type=Path, required=True, help="Paired dataset benchmark directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory for CSVs and quality manifest")
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
