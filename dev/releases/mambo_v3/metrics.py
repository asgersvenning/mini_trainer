"""Evaluate canonical release CSVs at the campaign's pinned mini_metrics revision."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json

REVISION = "70cc69adc05362863439277048e06386c1f885e1"
METRIC_SCHEMA = "mini-metrics-quality-v2"


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def require_pinned_metrics():
    """Reject metric environments without the campaign's recorded Git revision."""
    distribution = importlib.metadata.distribution("mini_metrics")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError(f"Require mini_metrics git revision {REVISION} in a separate environment")


def measure(source):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import evaluate_file

    require_pinned_metrics()
    data = MetricDF.from_source(source)
    if np.any(data.threshold != 0) or not np.isfinite(data.confidence).all():
        raise ValueError("Evaluation requires finite, unthresholded predictions")
    result = {
        "metric_schema": METRIC_SCHEMA,
        "source_sha256": file_hash(source),
        "mini_metrics_revision": REVISION,
        "policy": "threshold=0; no optimization; undefined metrics are null",
        "ranks": {},
    }
    for level, rank in enumerate(("species", "genus", "family")):
        selected = np.asarray(data.level) == level
        known = selected & np.asarray(data.known_label)
        result["ranks"][rank] = {
            "images": int(selected.sum()),
            "known_images": int(known.sum()),
            "list_coverage": float(known.sum() / selected.sum()),
            "truth_species_or_taxa": len(set(data.label[selected])),
        }
    for scope, known_only, per_class in (("all", False, False), ("known", True, False), ("per_class", False, True)):
        result[scope] = finite_json(
            evaluate_file(
                data,
                threshold=0,
                optimal=False,
                known_only=known_only,
                per_class=per_class,
                simple=True,
                hierarchical=False,
                pattern=r"^(micro_accuracy|accuracy|f1|recall|precision|coverage|theilU)$",
                verbose=0,
            )
        )
    for level, rank in enumerate(("species", "genus", "family")):
        row = result["ranks"][rank]
        row["micro_accuracy_all"] = result["all"]["micro_accuracy"][str(level)]
        row["micro_accuracy_known"] = result["known"]["micro_accuracy"][str(level)] if row["known_images"] else None
        row["macro_accuracy_all"] = result["all"]["accuracy"][str(level)]
        row["macro_accuracy_known"] = result["known"]["accuracy"][str(level)] if row["known_images"] else None
        row["abstention_coverage"] = result["all"]["coverage"][str(level)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path)
    sources.add_argument("--collection", type=Path, help="Completed run_local full directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.collection:
        plan = json.loads((args.collection / "plan.json").read_text())
        if plan["status"] != "complete":
            raise ValueError("Collection phase is not complete")
        for variant in plan["completed"]:
            root = args.collection / variant
            report = json.loads((root / "report.json").read_text())
            if report["status"] != "complete":
                raise ValueError("Incomplete variant")
            for name, digest in report["csv_sha256"].items():
                source = root / name / "mini_metric.csv"
                if file_hash(source) != digest:
                    raise ValueError("Prediction CSV changed since collection")
                output = source.with_name("metrics.json")
                if output.exists():
                    prior = json.loads(output.read_text())
                    if prior["source_sha256"] != digest or prior["mini_metrics_revision"] != REVISION:
                        raise ValueError("Existing metrics do not match this input or pinned revision")
                    if prior.get("metric_schema") == METRIC_SCHEMA:
                        continue
                    raise ValueError("Existing metrics use an older extractor; archive them before recomputing")
                write_json(output, measure(source))
                print(variant, name, flush=True)
    else:
        if args.output is None:
            parser.error("--output is required with --source")
        if args.output.exists():
            raise FileExistsError(args.output)
        report = measure(args.source)
        write_json(args.output, report)
        print(json.dumps(report["ranks"], indent=2))


if __name__ == "__main__":
    main()
