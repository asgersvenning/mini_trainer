"""Recover original test identities from archived staging and truth for UCloud setup."""

import csv
import json
import tomllib
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.audit import HERE


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
