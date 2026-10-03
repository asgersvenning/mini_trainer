"""Apply the Flemming calibration and support policy to the UCloud test predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.defaults_report import SERIES
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics
from dev.releases.mambo_v3.tail_charts import render_paired
from dev.releases.mambo_v3.tail_report import collect as collect_tails
from dev.releases.mambo_v3.threshold_report import identity
from dev.releases.mambo_v3.ucloud_release import validated_report


def collect(root, output):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold, evaluate_file

    require_pinned_metrics()
    plan = json.loads((root / "full/plan.json").read_text())
    if plan["status"] != "complete" or set(plan["completed"]) != {m for m, _, _ in SERIES}:
        raise ValueError("Require five completed models")
    output.mkdir(parents=True, exist_ok=True)
    study = {
        "revision": REVISION,
        "plan_sha256": file_hash(root / "full/plan.json"),
        "preset": "full",
        "tta": "rotation30_pad25_3",
        "split": "MetricDF.split((0.9, 0.1), strata=('label',), seed=42); reporting/calibration; grouped by instance_id",
        "policy": "Per-rank MacroF1; eps=0.01; use_quantiles=True; n_bootstraps=0; all truth",
        "models": {},
    }
    expected = None
    for model, _, _ in SERIES:
        folder = root / "full" / model
        report = validated_report(folder)
        if file_hash(folder / "report.json") != plan["reports_sha256"][model] or report["samples"] != 632913:
            raise ValueError("Changed or incomplete report")
        source = folder / "full/mini_metric.csv"
        data = MetricDF.from_source(source)
        if np.any(data.threshold != 0) or not np.isfinite(data.confidence).all():
            raise ValueError("Require unthresholded finite predictions")
        reporting, calibration = data.split((0.9, 0.1), strata=("label",), seed=42)
        identities = {"full": identity(data), "report": identity(reporting), "calibration": identity(calibration)}
        if expected is not None and expected != identities:
            raise ValueError("Models differ in reporting/calibration populations")
        expected = identities
        if set(reporting.instance_id) & set(calibration.instance_id):
            raise ValueError("Calibration leakage")
        selected = OptimalConfidenceThreshold(crit=MacroF1, eps=0.01, use_quantiles=True, n_bootstraps=0)(calibration, verbose=0)
        thresholds = [float(selected[k]) for k in range(3)]
        row = {
            "source": str(source),
            "source_sha256": file_hash(source),
            "identities": identities,
            "report_images": len(set(reporting.instance_id)),
            "calibration_images": len(set(calibration.instance_id)),
            "thresholds": thresholds,
        }
        for scope, vector in (("zero", 0), ("optimized", thresholds)):
            row[f"report_{scope}"] = finite_json(
                evaluate_file(
                    reporting,
                    threshold=vector,
                    optimal=False,
                    known_only=False,
                    simple=True,
                    hierarchical=False,
                    pattern=r"^(accuracy|micro_accuracy|precision|recall|f1|coverage|theilU)$",
                    verbose=0,
                )
            )
        study["models"][model] = row
        write_json(output / "mambo-indomain-thresholds.json", study)
        print(model, thresholds, row["report_images"], row["calibration_images"], flush=True)
    tails = collect_tails(study)
    first = next(iter(study["models"].values()))
    tails.update(
        tta=study["tta"],
        report_images=first["report_images"],
        calibration_images=first["calibration_images"],
        dataset_title="In-domain global-lepi test · global vocabulary",
        independent_recipe=True,
    )
    write_json(output / "mambo-indomain-tail.json", tails)
    rows = [{**{k: v for k, v in row.items() if k not in ("metrics", "classes")}, **row["metrics"]} for row in tails["rows"]]
    with (output / "mambo-indomain-tail.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    publish(output)


def publish(output):
    tails = json.loads((output / "mambo-indomain-tail.json").read_text())
    compact = {k: v for k, v in tails.items() if k != "rows"}
    compact["rows"] = [
        {k: v for k, v in r.items() if k != "classes"}
        for r in tails["rows"]
        if r["cutoff"] == -1 or (r["cutoff"] == 5 and r["domain"] == "common")
    ]
    compact["shared_class_domains"] = {
        f"{r['scope']}/{r['rank']}": r["classes"]
        for r in tails["rows"]
        if r["model"] == "torch" and r["cutoff"] == 5 and r["domain"] == "common"
    }
    write_json(output / "mambo-indomain-support.json", compact)
    render_paired(tails, output)
    for suffix in ("svg", "png"):
        (output / f"mambo-threshold-tail.{suffix}").rename(output / f"mambo-indomain-quality.{suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.render_only:
        publish(args.output)
    elif args.root is None:
        parser.error("--root is required for collection")
    else:
        collect(args.root, args.output)
