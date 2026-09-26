"""Supplementary support-truncated macro metrics from pinned mini_metrics class outputs."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics
from dev.releases.mambo_v3.threshold_report import identity


def eligible_classes(truth, accepted_predictions, cutoff):
    """Use strict support cutoffs in both domains, without dropping evaluation rows."""
    return {
        label
        for label in truth.keys() | accepted_predictions.keys()
        if truth.get(label, 0) > cutoff and accepted_predictions.get(label, 0) > cutoff
    }


def collect(study):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroAccuracy, MacroF1, MacroPrecision, MacroRecall

    require_pinned_metrics()
    metrics = {"accuracy": MacroAccuracy(), "precision": MacroPrecision(), "recall": MacroRecall(), "f1": MacroF1()}
    work = {}
    identities = {}
    for model, source in study["models"].items():
        if file_hash(source["source"]) != source["source_sha256"]:
            raise ValueError("Changed predictions")
        data = MetricDF.from_source(source["source"])
        data, _ = data.split((0.9, 0.1), strata=("label",), seed=42)
        identities[model] = identity(data)
        if identity(data) != source["identities"]["report"]:
            raise ValueError("Changed reporting partition")
        work[model] = {}
        scopes = (("zero", 0), ("optimized", source["thresholds"]))
        for scope, threshold in scopes:
            df = data.with_threshold(threshold)
            groups = {name: metric(df, aggregate=False, verbose=0) for name, metric in metrics.items()}
            work[model][scope] = {}
            for level in range(3):
                rows = df[df.level == level]
                work[model][scope][level] = {
                    "truth": Counter(map(str, rows.label)),
                    "predictions": Counter(map(str, rows.prediction[rows.prediction_made])),
                    "groups": {m: g[level] for m, g in groups.items()},
                }
    if len(set(identities.values())) != 1:
        raise ValueError("Model evaluation populations differ")
    result = {
        "population": "reporting_partition",
        "identities": identities,
        "sources_sha256": {model: source["source_sha256"] for model, source in study["models"].items()},
        "revision": REVISION,
        "policy": "Strictly > cutoff in both truth and accepted predictions; preserve all per-class FP/FN; macro reaggregation only",
        "rows": [],
    }
    for scope in ("zero", "optimized"):
        for level, rank in enumerate(("species", "genus", "family")):
            for cutoff in (-1, 5, 10, 20):
                eligible = {
                    model: eligible_classes(scopes[scope][level]["truth"], scopes[scope][level]["predictions"], cutoff)
                    for model, scopes in work.items()
                }
                shared = set.intersection(*eligible.values())
                for model, scopes in work.items():
                    row = scopes[scope][level]
                    domains = [("per_model", eligible[model])]
                    if cutoff >= 0:
                        domains.append(("common", shared))
                    for domain, selected in domains:
                        reference = study["models"][model]["report_zero" if scope == "zero" else "report_optimized"]
                        result["rows"].append(
                            {
                                "model": model,
                                "scope": scope,
                                "rank": rank,
                                "cutoff": cutoff,
                                "domain": domain,
                                "classes": sorted(selected),
                                "class_count": len(selected),
                                "truth_images_in_retained_classes": sum(row["truth"][k] for k in selected),
                                "accepted_predictions_in_retained_classes": sum(row["predictions"][k] for k in selected),
                                "report_images": sum(row["truth"].values()),
                                "overall_coverage": reference["coverage"][str(level)],
                                "metrics": {
                                    name: float(
                                        metric._aggregate_groups({k: v for k, v in row["groups"][name].items() if str(k) in selected})
                                    )
                                    if selected
                                    else None
                                    for name, metric in metrics.items()
                                },
                            }
                        )
    return finite_json(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("docs/assets/mambo-threshold-comparison.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(json.loads(args.study.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "mambo-tail-metrics.json", result)
    with (args.output / "mambo-tail-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "scope",
                "rank",
                "cutoff",
                "domain",
                "class_count",
                "truth_images_in_retained_classes",
                "accepted_predictions_in_retained_classes",
                "report_images",
                "overall_coverage",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**{k: v for k, v in row.items() if k not in ("classes", "metrics")}, **row["metrics"]})
