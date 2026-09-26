"""Recover display-only taxon names from the pinned preset metadata; never change IDs."""

import argparse
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dev.releases.mambo_v3.prepare_candidate import HERE, ROOT, digest


def prepare(metadata, output):
    expected = tomllib.loads((HERE / "construction.toml").read_text())["source"]
    if digest(metadata) != expected["sha256"]:
        raise ValueError("Metadata differs from pinned preset source")
    descriptor = json.loads((ROOT / "deployment/mambo_deploy/default_bundle.json").read_text())
    classes = json.loads(descriptor["metadata"]["classes.json"])["labels"]
    eligible = set().union(*map(set, classes))
    counts = defaultdict(Counter)
    columns = ["speciesKey", "genusKey", "familyKey", "genus", "specificEpithet", "family"]
    for batch in pq.ParquetFile(metadata).iter_batches(batch_size=262144, columns=columns):
        grouped = pa.Table.from_batches([batch]).group_by(columns).aggregate([("speciesKey", "count")]).to_pylist()
        for row in grouped:
            species = " ".join(str(row[key]) for key in ("genus", "specificEpithet") if row[key])
            for key, name in (("speciesKey", species), ("genusKey", row["genus"]), ("familyKey", row["family"])):
                label = str(row[key])
                if label in eligible and name:
                    counts[label][str(name)] += row["speciesKey_count"]
    names = {label: sorted(options, key=lambda name: (-options[name], name))[0] for label, options in sorted(counts.items())}
    output.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n")
    print(f"Display names: {len(names)}/{len(eligible)}; sha256={digest(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.metadata, args.output)
