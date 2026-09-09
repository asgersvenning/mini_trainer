"""Compare paired prediction tables with explicit held-out identity and mini_metrics."""

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
from argparse import ArgumentParser
from pathlib import Path

from .onnx_inference import file_hash

METRICS = ("f1", "recall", "precision", "coverage", "theilU")
COLUMNS = ("instance_id", "filename", "level", "label", "prediction", "confidence", "threshold")


def read_manifest(path):
    payload = Path(path).read_bytes()
    manifest = json.loads(payload)
    validate_dataset(manifest)
    for mode in ("baseline", "candidate"):
        artifact = manifest.get(mode, {})
        if not isinstance(artifact.get("path"), str) or not artifact["path"] or not artifact.get("provenance"):
            raise ValueError(f"Declare prediction path and model/preprocessing provenance for {mode}")
        if artifact.get("classes") != [level["classes"] for level in manifest["levels"]]:
            raise ValueError(f"{mode} class mappings must match the ordered level mappings")
    return manifest, hashlib.sha256(payload).hexdigest()


def validate_dataset(manifest):
    """Validate the shared held-out identity contract before inference or evaluation."""
    if manifest.get("schema_version") != 1 or manifest.get("split") not in ("val", "test"):
        raise ValueError("Require schema_version=1 and a declared val/test split")
    if not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]:
        raise ValueError("Record dataset and preprocessing provenance")
    levels = manifest.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Supply ordered level names and class mappings")
    names = set()
    for level in levels:
        name, classes = level.get("name"), level.get("classes")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Level names must be nonempty and unique")
        names.add(name)
        if (
            not isinstance(classes, list)
            or not classes
            or any(not isinstance(c, str) or not c for c in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("Each level requires unique nonempty class names in score-column order")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Declare the complete held-out sample set")
    ids = set()
    class_sets = [set(level["classes"]) for level in levels]
    for sample in samples:
        identifier = sample.get("instance_id")
        if type(identifier) is not int or not 0 <= identifier < 2**63 or identifier in ids:
            raise ValueError("Sample IDs must be unique nonnegative INT64 integers")
        ids.add(identifier)
        if not isinstance(sample.get("filename"), str) or not sample["filename"]:
            raise ValueError("Each sample needs its filename/identity")
        labels = sample.get("labels")
        if not isinstance(labels, list) or len(labels) != len(levels):
            raise ValueError("Each sample needs one label per level")
        if any(label not in classes for label, classes in zip(labels, class_sets, strict=True)):
            raise ValueError("Sample labels must belong to the declared classes")


def read_predictions(path, artifact, manifest):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if artifact.get("sha256") is not None and artifact["sha256"] != digest:
        raise ValueError(f"Prediction hash mismatch: {path}")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS) or len(reader.fieldnames) != len(COLUMNS):
        raise ValueError("Prediction CSV must contain exactly the documented seven columns")
    samples = {sample["instance_id"]: sample for sample in manifest["samples"]}
    classes = [set(level["classes"]) for level in manifest["levels"]]
    rows = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed prediction CSV row")
        identifier, level = int(row["instance_id"]), int(row["level"])
        key = identifier, level
        if identifier not in samples or not 0 <= level < len(classes) or key in rows:
            raise ValueError("Unknown or duplicate sample/level in predictions")
        sample = samples[identifier]
        if row["filename"] != sample["filename"] or row["label"] != sample["labels"][level]:
            raise ValueError("Prediction filename/label does not match held-out manifest")
        if row["prediction"] not in classes[level]:
            raise ValueError("Prediction is outside declared classes")
        confidence, threshold = float(row["confidence"]), float(row["threshold"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1 or threshold != 0:
            raise ValueError("Require confidence in [0,1] and threshold zero; this comparison does not tune or abstain")
        rows[key] = {**row, "instance_id": identifier, "level": level, "confidence": confidence, "threshold": 0.0}
    if len(rows) != len(samples) * len(classes):
        raise ValueError("Predictions do not cover every held-out sample at every level")
    # Canonicalize rows so CSV order cannot silently change the comparison.
    ordered = [rows[(sample["instance_id"], level)] for level in range(len(classes)) for sample in manifest["samples"]]
    return {column: [row[column] for row in ordered] for column in COLUMNS}, digest


def compare(manifest, output):
    metadata, digest = read_manifest(manifest)
    try:
        import mini_metrics
        from mini_metrics.data import MetricDF
        from mini_metrics.metrics import MacroF1, evaluate_file
    except ImportError as error:
        raise ImportError("Prepare mini_metrics explicitly in the evaluation environment; this command installs nothing") from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "status": "running",
        "runner_sha256": file_hash(__file__),
        "environment": {
            "python": platform.python_version(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "manifest": {"path": str(manifest), "sha256": digest, "contents": metadata},
        "policy": {
            "threshold": 0,
            "optimal": False,
            "known_only": False,
            "per_class": False,
            "criterion": "MacroF1",
            "hierarchical_metrics": False,
        },
        "scope": (
            "Fixed predictions evaluated independently at each declared level; "
            "not threshold tuning, score parity, inference performance or production acceptance."
        ),
        "models": {},
        "undefined_metrics": [],
    }
    try:
        root = Path(mini_metrics.__file__).parent
        try:
            version = importlib.metadata.version("mini_metrics")
        except importlib.metadata.PackageNotFoundError:
            version = None
        report["mini_metrics"] = {
            "installed_distribution_version": version,
            "imported_package": str(root),
            "source_hashes": {str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))},
        }
        tables = {}
        for mode in ("baseline", "candidate"):
            artifact = metadata[mode]
            path = Path(manifest).parent / artifact["path"]
            table, source_hash = read_predictions(path, artifact, metadata)
            tables[mode] = table
            info = {"source_sha256": source_hash, "source_path": str(path)}
            report["models"][mode] = info
            normalized = output / f"{mode}.csv"
            with normalized.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(dict(zip(COLUMNS, row, strict=True)) for row in zip(*(table[c] for c in COLUMNS), strict=True))
            info["normalized_csv_sha256"] = file_hash(normalized)
            # Construct directly to preserve literal string labels such as "001".
            result = evaluate_file(
                MetricDF(table),
                optimal=False,
                threshold=0,
                known_only=False,
                per_class=False,
                simple=True,
                hierarchical=False,
                pattern=r"^(f1|recall|precision|coverage|theilU)$",
                opt_crit=MacroF1,
                verbose=0,
            )
            if set(result) != set(METRICS):
                raise ValueError("mini_metrics did not return exactly the requested five metrics")
            info["metrics"] = {}
            for metric in METRICS:
                values = {int(level): float(value) for level, value in result[metric].items()}
                if set(values) != set(range(len(metadata["levels"]))):
                    raise ValueError("mini_metrics returned unexpected levels")
                info["metrics"][metric] = {str(level): value if math.isfinite(value) else None for level, value in values.items()}
                report["undefined_metrics"].extend(
                    {"model": mode, "metric": metric, "level": level} for level, value in values.items() if not math.isfinite(value)
                )
        report["levels"] = []
        for level, spec in enumerate(metadata["levels"]):
            delta = {}
            for metric in METRICS:
                a, b = [report["models"][mode]["metrics"][metric][str(level)] for mode in ("baseline", "candidate")]
                delta[metric] = None if a is None or b is None else b - a
            predictions = [
                [pred for pred, lvl in zip(tables[mode]["prediction"], tables[mode]["level"], strict=True) if lvl == level]
                for mode in ("baseline", "candidate")
            ]
            report["levels"].append(
                {
                    "name": spec["name"],
                    "samples": len(metadata["samples"]),
                    "candidate_minus_baseline": delta,
                    "prediction_changes": sum(a != b for a, b in zip(*predictions, strict=True)),
                }
            )
        report["status"] = "evaluated"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for canonical CSVs and metric report")
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
