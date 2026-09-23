"""Reconstruct frozen regional membership from the pinned local Parquet metadata."""

import argparse
import json
import tomllib
from pathlib import Path

from dev.releases.mambo_v3.audit import HERE, sha256


def region_counts(table, column, values):
    import pyarrow as pa
    import pyarrow.compute as pc

    selected = table.filter(pc.is_in(table[column], value_set=pa.array(values)))
    if selected["speciesKey"].null_count:
        raise ValueError("Selected metadata contains null species IDs")
    counts = selected["speciesKey"].value_counts().to_pylist()
    return {row["values"]: row["counts"] for row in counts}


def membership(counts, threshold):
    return {species for species, count in counts.items() if count > threshold}


def reconstruct(metadata):
    import pyarrow.parquet as pq

    config = tomllib.loads((HERE / "construction.toml").read_text())
    source = config["source"]
    if metadata.stat().st_size != source["size"] or sha256(metadata) != source["sha256"]:
        raise ValueError("Source Parquet differs from pinned metadata")
    table = pq.read_table(metadata, columns=["speciesKey", "countryCode", "continent"])
    if table.num_rows != source["rows"]:
        raise ValueError("Unexpected metadata row count")
    report = {"source_sha256": source["sha256"], "rows": table.num_rows, "regions": {}}
    for name in ("europe", "north_europe"):
        rule = config[name]
        counts = region_counts(table, rule["column"], rule["values"])
        actual = membership(counts, rule["exclusive_minimum_rows"])
        expected = set((HERE / "presets" / f"{name}.classes").read_text().splitlines())
        if actual != expected:
            raise ValueError(f"{name} differs: missing={len(expected - actual)}, extra={len(actual - expected)}")
        stats = {
            "selected_rows": sum(counts.values()),
            "species_before_threshold": len(counts),
            "species_after_threshold": len(actual),
        }
        if any(value != rule[key] for key, value in stats.items()):
            raise ValueError(f"{name} count totals differ from reconstruction record")
        report["regions"][name] = stats
    northern = config["north_europe"]
    base = set((HERE / "presets/north_europe.classes").read_text().splitlines())
    for label, extra in (("neutral_additions", northern["membership_neutral_additions"]), ("with_GB", ["GB"])):
        counts = region_counts(table, "countryCode", northern["values"] + extra)
        actual = membership(counts, northern["exclusive_minimum_rows"])
        report["regions"]["north_europe"][label] = {"extra": len(actual - base), "missing": len(base - actual)}
        if label == "neutral_additions" and actual != base:
            raise ValueError("Documented neutral countries change membership")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.metadata), indent=2))
