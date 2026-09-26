"""Release evaluation identities and canonical rows; no runtime or taxonomy downloads."""

import csv
import json
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.benchmarks.inference.prepare_inputs import select_records
from dev.benchmarks.inference.quality_compare import COLUMNS

CSV_COLUMNS = (*COLUMNS, "known_label", "prediction_made", "correct")
PRESETS = ("full", "europe", "north_europe", "europe_v3", "north_europe_v3")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_flemming(root, reference, output):
    """Join archived ground truth by relative species/image path, then hash local bytes."""
    records = {}
    with Path(reference).open(newline="") as stream:
        for row in csv.DictReader(stream):
            relative = Path(*Path(row["filename"]).parts[-2:]).as_posix()
            record = records.setdefault(relative, {"path": relative, "labels": [None] * 3, "split": "test"})
            rank = int(row["level"])
            if rank not in range(3) or record["labels"][rank] is not None:
                raise ValueError("Duplicate or invalid truth rank")
            record["labels"][rank] = row["label"]
    observed = {p.relative_to(root).as_posix() for p in Path(root).rglob("*.jpg")}
    if observed != set(records):
        raise ValueError(f"Image identity mismatch: missing {len(set(records) - observed)}, extra {len(observed - set(records))}")
    for record in records.values():
        if any(label is None for label in record["labels"]) or record["labels"][0] != Path(record["path"]).parent.name:
            raise ValueError("Incomplete or conflicting ground truth")
        record["sha256"] = file_hash(Path(root) / record["path"])
    result = {
        "schema_version": 1,
        "dataset": "flemming",
        "records": sorted(records.values(), key=lambda r: r["path"]),
        "provenance": {"truth_csv_sha256": file_hash(reference), "split": "original expert set; no resplitting"},
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, result)
    return result


def load_records(path, root, count=None, seed=20260923):
    manifest = json.loads(Path(path).read_text())
    records = select_records(manifest, "test", count, seed)
    for record in records:
        candidate = (Path(root) / record["path"]).resolve()
        if not candidate.is_relative_to(Path(root).resolve()) or not candidate.is_file():
            raise ValueError(f"Missing or unsafe image: {record['path']}")
        if len(record.get("labels", [])) != 3 or not all(isinstance(label, str) and label for label in record["labels"]):
            raise ValueError("Require explicit species/genus/family ground truth")
    return manifest, records


def canonical_rows(records, prediction, offset=0):
    if len(records) != len(prediction):
        raise ValueError("Prediction/sample count mismatch")
    for i, (record, item) in enumerate(zip(records, prediction, strict=True)):
        for rank in range(3):
            truth, predicted = record["labels"][rank], item.label[rank]
            yield (
                offset + i,
                record["path"],
                rank,
                truth,
                predicted,
                item.confidence[rank],
                0,
                int(truth in prediction.cls2idx[str(rank)]),
                1,
                1 if truth == predicted else -1,
            )
