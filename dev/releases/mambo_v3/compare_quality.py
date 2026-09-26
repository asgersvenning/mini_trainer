"""Compare completed prediction runs by original image/rank identity, without score tolerances."""

import argparse
import csv
import json
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import PRESETS, write_json


def read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = {}
        for row in csv.DictReader(stream):
            key = (row["filename"], int(row["level"]))
            if key in rows:
                raise ValueError("Duplicate image/rank")
            rows[key] = row
    return rows


def compare(left, right, presets=PRESETS):
    reports = [json.loads((path / "report.json").read_text()) for path in (left, right)]
    if any(r["status"] != "complete" for r in reports) or reports[0]["sample_ids_sha256"] != reports[1]["sample_ids_sha256"]:
        raise ValueError("Require completed runs with identical ordered sample hashes")
    result = {
        "left": str(left),
        "right": str(right),
        "reports_sha256": [file_hash(path / "report.json") for path in (left, right)],
        "presets": {},
    }
    for preset in presets:
        a, b = [read_rows(path / preset / "mini_metric.csv") for path in (left, right)]
        if set(a) != set(b):
            raise ValueError("Sample identity mismatch")
        for key in a:
            if (a[key]["label"], a[key]["known_label"]) != (b[key]["label"], b[key]["known_label"]):
                raise ValueError("Ground truth or class-list coverage mismatch")
        result["presets"][preset] = {}
        for rank in range(3):
            keys = [key for key in a if key[1] == rank]
            changes = sum(a[key]["prediction"] != b[key]["prediction"] for key in keys)
            result["presets"][preset][str(rank)] = {
                "images": len(keys),
                "changed": changes,
                "agreement": 1 - changes / len(keys),
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, compare(args.left, args.right))
