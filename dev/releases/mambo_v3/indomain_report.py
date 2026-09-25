"""Apply the Flemming calibration and support policy to the UCloud test predictions."""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.defaults_report import SERIES
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json
from dev.releases.mambo_v3.tail_charts import render_paired
from dev.releases.mambo_v3.tail_report import collect as collect_tails
from dev.releases.mambo_v3.threshold_report import identity
from dev.releases.mambo_v3.ucloud_release import validated_report


def collect(root, output):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold, evaluate_file

    provenance = json.loads(importlib.metadata.distribution("mini_metrics").read_text("direct_url.json"))
    if provenance.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError("Require pinned mini_metrics")
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
    study = json.loads((output / "mambo-indomain-thresholds.json").read_text())
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

    lookup = {(r["model"], r["scope"], r["rank"], r["cutoff"], r["domain"]): r for r in tails["rows"]}
    text = "# In-domain deployment evidence\n\n"
    text += (
        f"Global vocabulary; {tails['report_images']:,} reporting images and {tails['calibration_images']:,} "
        "separate calibration images from the original 632,913-image test split. "
        "All truth is in vocabulary. Shared label-stratified 90/10 split, seed 42, grouped by image ID; "
        "per-rank mini_metrics Macro-F1 calibration, eps=0.01, quantiles, no bootstraps.\n\n"
        "The recipe was selected on Flemming. These general photographs complement the more deployment-relevant "
        "Flemming monitoring crops; their different TTA response does not invalidate that deployment evidence.\n\n"
        "Full support / >5 values use the same reporting rows. >5 requires truth and accepted-prediction support "
        "in every pipeline, separately per confidence setting. No rows are removed; per-class FP/FN remain intact.\n\n"
    )
    for level, rank in enumerate(("species", "genus", "family")):
        text += (
            f"## {rank.title()}\n\n| Pipeline | Confidence | Threshold | "
            "Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |\n|---|---|---:|---:|---:|---:|\n"
        )
        for model, label, _ in SERIES:
            for scope, name in (("zero", "None"), ("optimized", "Calibrated")):
                a = lookup[model, scope, rank, -1, "per_model"]
                b = lookup[model, scope, rank, 5, "common"]
                t = study["models"][model]["thresholds"][level] if scope == "optimized" else 0
                text += (
                    f"| {label} | {name} | {t:.4f} | {a['metrics']['accuracy']:.2%} / {b['metrics']['accuracy']:.2%} | "
                    f"{a['metrics']['f1']:.4f} / {b['metrics']['f1']:.4f} | {a['overall_coverage']:.2%} |\n"
                )
        text += "\n"
    text += (
        "## Support outside the truncated average\n\n"
        "| Confidence | Rank | Shared classes | Truth images outside | Proportion |\n|---|---|---:|---:|---:|\n"
    )
    for scope in ("zero", "optimized"):
        for rank in ("species", "genus", "family"):
            row = lookup["torch", scope, rank, 5, "common"]
            n = row["report_images"] - row["truth_images_in_retained_classes"]
            text += f"| {scope} | {rank} | {row['class_count']} | {n:,} | {n / row['report_images']:.2%} |\n"
    text += (
        "\nMachine-readable [metrics](assets/mambo-indomain-tail.csv), "
        "[thresholds and split identities](assets/mambo-indomain-thresholds.json), and "
        "[class domains](assets/mambo-indomain-support.json) retain provenance and supplementary metrics. "
        "Thresholds are dataset-specific evidence, not new deployment defaults.\n"
    )
    text += (
        "\n## Historical HPC timing boundaries\n\n"
        "The latest V3 B200 timings are in the [current HPC evidence](mambo-hpc-evidence.md). "
        "The observations below predate the pipeline improvements.\n\n"
        "The EPYC 9655/B200 campaign retains 3 fresh-process trials per variant/device, 7 request observations "
        "per cell and 3 streaming observations per cell. Global and northern-Europe timing presets are available. "
        "CPU runtime threads: 4; streaming preparation workers: 48; readers: 256. "
        "Request, streaming and prepared-input diagnostics have different boundaries; do not pool them. "
        "The short streaming bank contains 1,024 images and includes pipeline startup. "
        "Prepared-input diagnostics exclude decoding/hierarchy reduction but include transfers, and remain supplementary.\n\n"
        "[Request observations](assets/mambo-indomain-speed.csv) and "
        "[streaming observations](assets/mambo-indomain-streaming-speed.csv) retain all trials. "
        "[Campaign provenance](assets/mambo-indomain-campaign.json) identifies source hashes and runtime environments. "
        "Peak host memory spans each complete benchmark process and its tested batch sizes; it is not per-cell model memory. "
        "V2 was tested through batch 32 on GPU, V3 through batch 256.\n\n"
        "## Reproduce\n\n"
        "Use the pinned mini_metrics environment described in the [UCloud workflow](../dev/releases/mambo_v3/ucloud-release.md). "
        "From the repository root, with the extracted archive beneath `local-evidence/ucloud-2026-09-25/`:\n\n"
        "```sh\npython -m dev.releases.mambo_v3.indomain_report \\\n"
        "  --root local-evidence/ucloud-2026-09-25/mambo-results/runs-transfers \\\n"
        "  --output local-evidence/ucloud-2026-09-25/presentation\n"
        "python -m dev.releases.mambo_v3.indomain_speed \\\n"
        "  --source local-evidence/ucloud-2026-09-25/mambo-results/summary-transfers \\\n"
        "  --output local-evidence/ucloud-2026-09-25/presentation\n```\n"
    )
    (output / "mambo-indomain-evidence.md").write_text(text)


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
