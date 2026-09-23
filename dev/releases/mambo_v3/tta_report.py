"""Compute pinned mini_metrics for completed TTA qualification and compact the results."""

import argparse
import json
from pathlib import Path

from deployment.mambo_deploy.augmentation import resolve_tta
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import METRIC_SCHEMA, REVISION, measure
from dev.releases.mambo_v3.tta_candidates import CANDIDATES, candidate_policy


def run(args):
    result = {"purpose": "fixed-subset qualification; not full-data TTA efficacy", "revision": REVISION, "variants": []}
    sample_hash = None
    seen = set()
    for root_parent in (args.root, *args.extra_root):
        for backend in args.backends:
            root = root_parent / f"mambo-tta-{backend}"
            report = json.loads((root / "report.json").read_text())
            if report["status"] != "complete" or file_hash(root / "samples.json") != report["sample_ids_sha256"]:
                raise ValueError("Incomplete or changed TTA qualification")
            if sample_hash is not None and sample_hash != report["sample_ids_sha256"]:
                raise ValueError("Sample populations differ")
            sample_hash = report["sample_ids_sha256"]
            result["sample_ids_sha256"] = sample_hash
            result["samples"] = report["samples"]
            for profile in report["profiles"]:
                if (backend, profile) in seen:
                    raise ValueError("Duplicate backend/profile evidence")
                seen.add((backend, profile))
                tta = candidate_policy(profile) if profile in CANDIDATES else resolve_tta(profile)
                row = {"backend": backend, "profile": profile, "views": len(tta.transforms) if tta else 1, "presets": {}}
                for preset, digest in report["profiles"][profile]["csv_sha256"].items():
                    source = root / profile / preset / "mini_metric.csv"
                    if file_hash(source) != digest:
                        raise ValueError("Changed prediction CSV")
                    path = source.with_name("metrics.json")
                    metric = json.loads(path.read_text()) if path.exists() else measure(source)
                    if (
                        metric["source_sha256"] != digest
                        or metric["mini_metrics_revision"] != REVISION
                        or metric["metric_schema"] != METRIC_SCHEMA
                    ):
                        raise ValueError("Stale metric file")
                    if not path.exists():
                        write_json(path, metric)
                    row["presets"][preset] = {key: metric[key] for key in ("ranks", "all", "known", "source_sha256")}
                result["variants"].append(row)
                print(backend, profile, row["presets"]["north_europe"]["all"]["accuracy"]["0"], flush=True)
    write_json(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    parser.add_argument("--backends", nargs="+", default=["torch", "onnx"], choices=["torch", "onnx"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    run(args)
