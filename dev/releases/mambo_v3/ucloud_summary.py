"""Validate completed UCloud evidence and export environment-labelled quality/speed tables."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import METRIC_SCHEMA, REVISION
from dev.releases.mambo_v3.ucloud_release import VARIANTS, bank_identity, validated_report


def summarize(root, output):
    plans = {phase: json.loads((root / phase / "plan.json").read_text()) for phase in ("full", "benchmark")}
    if any(p["status"] != "complete" for p in plans.values()):
        raise ValueError("Both collection and benchmark phases must be complete")
    if plans["full"]["fingerprint"] != plans["benchmark"]["fingerprint"]:
        raise ValueError("Quality and timing campaign identities differ")
    data = {
        "environment_id": plans["full"]["environment_id"],
        "fingerprint": plans["full"]["fingerprint"],
        "quality": [],
        "speed": [],
        "streaming_speed": [],
        "runtime_reports": {},
        "metric_revision": REVISION,
        "policy": "Original full test split; no threshold selection on test; all and known truth; timings isolated from collection",
    }
    identities, banks = set(), set()
    for phase, plan in plans.items():
        if set(plan["completed"]) != {j["name"] for j in plan["jobs"]}:
            raise ValueError("Incomplete job list")
        if {j["variant"] for j in plan["jobs"]} != set(VARIANTS):
            raise ValueError("Missing release variant")
        for job in plan["jobs"]:
            directory = root / phase / job["name"]
            r = validated_report(directory)
            if file_hash(directory / "report.json") != plan["reports_sha256"][job["name"]]:
                raise ValueError("Changed job report")
            data["runtime_reports"][f"{phase}/{job['name']}"] = {k: v for k, v in r.items() if k not in ("samples", "cells")}
            if phase == "full":
                if r["samples"] != 632913:
                    raise ValueError("Not the complete original test set")
                identities.add(r["sample_ids_sha256"])
                for preset, digest in r["csv_sha256"].items():
                    m = json.loads((directory / preset / "metrics.json").read_text())
                    if m["source_sha256"] != digest or m["metric_schema"] != METRIC_SCHEMA or m["mini_metrics_revision"] != REVISION:
                        raise ValueError("Stale or incompatible mini_metrics output")
                    for scope in ("all", "known"):
                        for level, rank in enumerate(("species", "genus", "family")):
                            data["quality"].append(
                                {
                                    "environment_id": data["environment_id"],
                                    "variant": job["variant"],
                                    "preset": preset,
                                    "scope": scope,
                                    "rank": rank,
                                    **m["ranks"][rank],
                                    **{k: v[str(level)] for k, v in m[scope].items()},
                                }
                            )
            else:
                banks.add(bank_identity(directory, r))
                for c in r["cells"]:
                    if "streaming" in c:
                        stream = c["streaming"]
                        if len(stream["seconds"]) != 3 or any(v <= 0 for v in stream["seconds"]):
                            raise ValueError("Require three positive streaming observations")
                        data["streaming_speed"].append(
                            {
                                "environment_id": data["environment_id"],
                                "variant": job["variant"],
                                "device": job["device"],
                                "trial": job["name"],
                                "preset": c["preset"],
                                "batch_size": c["batch_size"],
                                "images": stream["images"],
                                "images_per_second": stream["images"] / statistics.median(stream["seconds"]),
                                "seconds": stream["seconds"],
                            }
                        )
                    seconds = c["end_to_end"]["seconds"]
                    if len(seconds) != 7 or any(v <= 0 for v in seconds):
                        raise ValueError("Require seven positive completed observations")
                    data["speed"].append(
                        {
                            "environment_id": data["environment_id"],
                            "variant": job["variant"],
                            "device": job["device"],
                            "trial": job["name"],
                            "preset": c["preset"],
                            "batch_size": c["batch_size"],
                            "images_per_second": c["batch_size"] / statistics.median(seconds),
                            "seconds": seconds,
                            "peak_host_rss_mib": r["peak_rss_kib_linux"] / 1024,
                        }
                    )
    if len(identities) != 1 or len(banks) != 1:
        raise ValueError("Mismatched quality populations or timing image banks")
    data["sample_ids_sha256"] = next(iter(identities))
    data["timing_bank_sha256"] = next(iter(banks))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "ucloud-summary.json", data)
    for name in ("quality", "speed", "streaming_speed"):
        if not data[name]:
            continue
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[name][0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data[name])
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
