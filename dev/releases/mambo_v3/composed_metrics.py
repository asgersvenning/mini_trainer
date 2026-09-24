"""Full composed-TTA calibration, matched-coverage diagnostics and support truncation."""

import argparse
import csv
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.composed_full import SHORTLIST
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json
from dev.releases.mambo_v3.tail_report import eligible_classes
from dev.releases.mambo_v3.threshold_report import identity


def matched_thresholds(data, coverage):
    """Label-free reporting-score operating points; never used as deployment calibration."""
    return [float(np.quantile(data.confidence[data.level == level], 1 - coverage, method="higher")) for level in range(3)]


def collect(args):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroAccuracy, MacroF1, MacroPrecision, MacroRecall, OptimalConfidenceThreshold, evaluate_file

    provenance = json.loads(importlib.metadata.distribution("mini_metrics").read_text("direct_url.json"))
    if provenance.get("vcs_info", {}).get("commit_id") != REVISION:
        raise ValueError("Require pinned mini_metrics")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline.read_text())
    sources = {model: row["source"] for model, row in baseline["models"].items()}
    for backend in args.backends:
        folder = args.root / backend
        report = json.loads((folder / "report.json").read_text())
        if report["status"] != "complete" or report["samples"] != 58640:
            raise ValueError(f"Incomplete full run: {folder}")
        for recipe in SHORTLIST:
            source = folder / recipe / "mini_metric.csv"
            if file_hash(source) != report["csv_sha256"][recipe]:
                raise ValueError("Changed prediction source")
            sources[f"{backend}:{recipe}"] = str(source)
    class_metrics = {"accuracy": MacroAccuracy(), "precision": MacroPrecision(), "recall": MacroRecall(), "f1": MacroF1()}
    groups, result = (
        {},
        {
            "revision": REVISION,
            "calibration": "MacroF1 eps=0.01 use_quantiles=True n_bootstraps=0; shared stratified seed42 90/10 report/calibration split",
            "matched_coverage": (
                "Label-free quantiles of reporting confidences; realized coverage computed by mini_metrics, ties retained. "
                "Diagnostic only, not deployable calibration."
            ),
            "support": (
                "Full per-model class domain; >5 common truth and accepted-prediction support "
                "across all compared models separately per operating point"
            ),
            "models": {},
        },
    )
    expected = baseline["models"]["torch"]["identities"]
    for model, source in sources.items():
        digest = file_hash(source)
        if model in baseline["models"] and digest != baseline["models"][model]["source_sha256"]:
            raise ValueError("Changed baseline predictions")
        data = MetricDF.from_source(source)
        reporting, calibration = data.split((0.9, 0.1), strata=("label",), seed=42)
        identities = {"full": identity(data), "report": identity(reporting), "calibration": identity(calibration)}
        if identities != expected:
            raise ValueError(f"Changed evaluation populations: {model}")
        if model in baseline["models"]:
            thresholds = baseline["models"][model]["thresholds"]
        else:
            selected = OptimalConfidenceThreshold(crit=MacroF1, eps=0.01, use_quantiles=True, n_bootstraps=0)(calibration, verbose=0)
            thresholds = [float(selected[k]) for k in range(3)]
        row = {
            "source": source,
            "source_sha256": digest,
            "identities": identities,
            "report_images": len(set(reporting.instance_id)),
            "calibration_images": len(set(calibration.instance_id)),
            "thresholds": thresholds,
            "operating_points": {},
        }
        points = {"zero": [0.0] * 3, "optimized": thresholds}
        points.update({f"coverage_{int(c * 100)}": matched_thresholds(reporting, c) for c in (0.7, 0.8, 0.9)})
        groups[model] = {}
        for scope, vector in points.items():
            df = reporting.with_threshold(vector)
            score = evaluate_file(
                reporting,
                threshold=vector,
                simple=True,
                hierarchical=False,
                pattern=r"^(accuracy|micro_accuracy|precision|recall|f1|coverage|theilU)$",
                verbose=0,
            )
            if model in baseline["models"] and scope in ("zero", "optimized"):
                previous = baseline["models"][model][f"report_{scope}"]
                for metric, levels in score.items():
                    for level, value in levels.items():
                        if not np.isclose(value, previous[metric][str(level)], atol=1e-12, rtol=0, equal_nan=True):
                            raise ValueError(f"Baseline changed: {model}/{scope}/{metric}/{level}")
            row["operating_points"][scope] = {"thresholds": vector, "full": finite_json(score), "tail": {}}
            per_class = {name: metric(df, aggregate=False, verbose=0) for name, metric in class_metrics.items()}
            groups[model][scope] = {}
            for level in range(3):
                part = df[df.level == level]
                groups[model][scope][level] = {
                    "truth": Counter(map(str, part.label)),
                    "predicted": Counter(map(str, part.prediction[part.prediction_made])),
                    "metrics": {name: values[level] for name, values in per_class.items()},
                }
        if model.startswith("onnx:"):
            native = model.replace("onnx:", "torch:", 1)
            if native in result["models"]:
                vector = result["models"][native]["thresholds"]
                row["native_threshold_alignment"] = {
                    "thresholds": vector,
                    "scores": finite_json(
                        evaluate_file(
                            reporting,
                            threshold=vector,
                            simple=True,
                            hierarchical=False,
                            pattern=r"^(accuracy|micro_accuracy|precision|recall|f1|coverage|theilU)$",
                            verbose=0,
                        )
                    ),
                }
        result["models"][model] = row
        write_json(args.output / f"{model.replace(':', '-')}.json", row)
        print(model, thresholds, flush=True)
    for scope in ("zero", "optimized", "coverage_70", "coverage_80", "coverage_90"):
        for level in range(3):
            domains = [eligible_classes(g[scope][level]["truth"], g[scope][level]["predicted"], 5) for g in groups.values()]
            selected = set.intersection(*domains)
            for model, g in groups.items():
                item = g[scope][level]
                result["models"][model]["operating_points"][scope]["tail"][str(level)] = {
                    "classes": sorted(selected),
                    "class_count": len(selected),
                    "truth_images": sum(item["truth"][k] for k in selected),
                    "accepted_predictions": sum(item["predicted"][k] for k in selected),
                    "metrics": {
                        name: float(metric._aggregate_groups({k: v for k, v in item["metrics"][name].items() if str(k) in selected}))
                        if selected
                        else None
                        for name, metric in class_metrics.items()
                    },
                }
    result = finite_json(result)
    write_json(args.output / "composed-comparison.json", result)
    export_csv(result, args.output / "composed-comparison.csv")
    return result


def export_csv(data, path):
    rows = []
    for model, result in data["models"].items():
        for scope, point in result["operating_points"].items():
            for k, rank in enumerate(("species", "genus", "family")):
                level = str(k)
                tail = point["tail"][level]
                rows.append(
                    {
                        "model": model,
                        "scope": scope,
                        "rank": rank,
                        "threshold": point["thresholds"][k],
                        "report_images": result["report_images"],
                        "coverage": point["full"]["coverage"][level],
                        **{f"full_{name}": levels[level] for name, levels in point["full"].items() if name != "coverage"},
                        **{f"tail_{name}": value for name, value in tail["metrics"].items()},
                        "tail_classes": tail["class_count"],
                        "tail_truth_images": tail["truth_images"],
                        "tail_accepted_predictions": tail["accepted_predictions"],
                    }
                )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("docs/assets/mambo-threshold-comparison.json"))
    parser.add_argument("--backends", nargs="+", choices=("torch", "onnx"), default=["torch", "onnx"])
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())
