"""Recover original UCloud test identities; verify-only works without the image dataset."""

import argparse
import csv
import json
import tomllib
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.audit import HERE
from dev.releases.mambo_v3.evaluation_data import write_json


def recover(metadata, staging, reference, test_set="0"):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    expected = tomllib.loads((HERE / "construction.toml").read_text())["source"]
    if file_hash(metadata) != expected["sha256"]:
        raise ValueError("Original metadata snapshot hash mismatch")
    table = pq.read_table(metadata, columns=["filename", "set", "speciesKey", "genusKey", "familyKey"])
    table = table.filter(pc.equal(table["set"], test_set))
    truth = {}
    for row in table.to_pylist():
        key = f"images/{row['speciesKey']}/{row['filename']}"
        if key in truth:
            raise ValueError("Duplicate original image identity")
        truth[key] = [row[k] for k in ("speciesKey", "genusKey", "familyKey")]
    files = json.loads(Path(staging).read_text())["files"]
    mapping = {}
    original = set()
    for item in files:
        source = Path(item["source"])
        # Preserve the archived suffix from the original global_lepi mount.
        relative = source.relative_to("/work/global_lepi").as_posix()
        if item["staged"] in mapping or relative in original:
            raise ValueError("Duplicate staging identity")
        mapping[item["staged"]] = relative
        original.add(relative)
    if original != set(truth):
        raise ValueError("Staging membership differs from the original supplied split")
    seen = set()
    with Path(reference).open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = mapping[row["filename"]]
            rank = int(row["level"])
            if (name, rank) in seen or not 0 <= rank < 3 or row["label"] != truth[name][rank]:
                raise ValueError("Archived truth differs from original taxonomy or has duplicate ranks")
            seen.add((name, rank))
    if len(seen) != len(truth) * 3:
        raise ValueError("Incomplete archived predictions")
    return [{"path": name, "labels": labels, "split": "test"} for name, labels in sorted(truth.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata", "staging", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--test-set", default="0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = recover(args.metadata, args.staging, args.reference, args.test_set)
    if len(records) != 632913:
        raise ValueError("Expected 632,913 original test images")
    provenance = {key: file_hash(getattr(args, key)) for key in ("metadata", "staging", "reference")}
    provenance["test_set"] = args.test_set
    if args.verify_only:
        write_json(
            args.output,
            {
                "status": "verified-identities-only",
                "images": len(records),
                "provenance": provenance,
                "limitation": "Image existence and bytes not verified locally",
            },
        )
    else:
        if args.root is None:
            parser.error("--root is required unless --verify-only")
        for i, record in enumerate(records):
            path = (args.root / record["path"]).resolve()
            if not path.is_relative_to(args.root.resolve()):
                raise ValueError("Unsafe original path")
            record["sha256"] = file_hash(path)
            if i % 10000 == 0:
                print(f"Hashed {i}/{len(records)}", flush=True)
        write_json(args.output, {"schema_version": 1, "dataset": "global-lepi-test", "provenance": provenance, "records": records})


if __name__ == "__main__":
    main()
