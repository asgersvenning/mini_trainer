"""Calibrate and compare northern-Europe rejection thresholds using pinned mini_metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.acceleration_report import METRICS
from dev.releases.mambo_v3.defaults_report import SERIES
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics

from .figure_export import save_figure

SOURCES = {
    "v2": "mambo-release-comparison-quality/v2-full",
    "torch": "mambo-accelerated-quality/torch-cuda-0-prediction",
    "onnx": "mambo-accelerated-quality/onnx-cuda-0-prediction",
    "torch-tta": "mambo-default-tta-quality/torch-cuda-0-prediction",
    "onnx-tta": "mambo-default-tta-quality/onnx-cuda-0-prediction",
}
RANKS = ("species", "genus", "family")


def identity(data):
    """Hash ordered image/rank/truth identities, independently of predictions."""
    rows = zip(data.instance_id.tolist(), data.level.tolist(), data.label.tolist(), strict=True)
    return hashlib.sha256(json.dumps(list(rows), separators=(",", ":")).encode()).hexdigest()


def collect(root, output):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold, evaluate_file

    require_pinned_metrics()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "revision": REVISION,
        "preset": "north_europe",
        "policy": "MacroF1; eps=0.01; use_quantiles=True; n_bootstraps=0; per-rank thresholds; all truth",
        "split": "MetricDF.split((0.9, 0.1), strata=('label',), seed=42); report/calibration; grouped by instance_id",
        "curve": "51 uniform confidence values plus 21 calibration quantiles per rank, and selected thresholds; reporting data",
        "models": {},
    }
    baseline = json.loads(Path("docs/assets/mambo-defaults-comparison.json").read_text())
    reference = {r["model"]: r["scores"]["all"] for r in baseline["quality"] if r["preset"] == "north_europe"}
    identities = None
    pattern = "^(" + "|".join(METRICS) + ")$"

    def measure(data, thresholds, known=False, curve=False):
        return finite_json(
            evaluate_file(
                data,
                threshold=thresholds,
                optimal=False,
                known_only=known,
                simple=True,
                hierarchical=False,
                pattern=r"^(accuracy|micro_accuracy|precision|recall|f1|coverage)$" if curve else pattern,
                verbose=0,
            )
        )

    for model, relative in SOURCES.items():
        source = root / relative / "north_europe/mini_metric.csv"
        digest = file_hash(source)
        prior = json.loads(source.with_name("metrics.json").read_text())
        if prior["source_sha256"] != digest or prior["mini_metrics_revision"] != REVISION:
            raise ValueError(f"Prediction provenance mismatch: {source}")
        data = MetricDF.from_source(source)
        if np.any(data.threshold != 0) or not np.isfinite(data.confidence).all():
            raise ValueError("Require finite unthresholded predictions")
        if np.any((data.confidence < 0) | (data.confidence > 1)):
            raise ValueError("Confidence outside [0, 1]")
        reporting, calibration = data.split((0.9, 0.1), strata=("label",), seed=42)
        current = {"full": identity(data), "report": identity(reporting), "calibration": identity(calibration)}
        if identities is not None and current != identities:
            raise ValueError("Models do not share identical image/truth partitions")
        identities = current
        if set(reporting.instance_id) & set(calibration.instance_id):
            raise ValueError("Calibration leakage across image IDs")
        full = measure(data, 0)
        for metric in METRICS:
            for level in range(3):
                if not np.isclose(full[metric][str(level)], reference[model][metric][str(level)], atol=1e-12, rtol=0):
                    raise ValueError(f"Threshold-zero baseline changed: {model}/{metric}/{level}")
        selected = OptimalConfidenceThreshold(crit=MacroF1, eps=0.01, use_quantiles=True, n_bootstraps=0)(calibration, verbose=0)
        thresholds = [float(selected[level]) for level in range(3)]
        row = {
            "source": str(source),
            "source_sha256": digest,
            "identities": current,
            "report_images": len(set(reporting.instance_id)),
            "calibration_images": len(set(calibration.instance_id)),
            "thresholds": thresholds,
            "full_zero": full,
            "report_zero": measure(reporting, 0),
            "report_optimized": measure(reporting, thresholds),
            "known_zero": measure(reporting, 0, known=True),
            "known_optimized": measure(reporting, thresholds, known=True),
            "calibration_zero": measure(calibration, 0),
            "calibration_optimized": measure(calibration, thresholds),
            "curve": [],
        }
        # Public optimal=True must reproduce the explicit shared split and optimizer.
        public = finite_json(
            evaluate_file(
                data,
                optimal=True,
                seed=42,
                opt_crit=MacroF1,
                eps=0.01,
                use_quantiles=True,
                simple=True,
                hierarchical=False,
                pattern=pattern,
                verbose=0,
            )
        )
        for metric in METRICS:
            if public[metric] != row["report_optimized"][metric]:
                raise ValueError(f"Public optimizer mismatch: {model}/{metric}")
        grids = [
            np.unique(
                np.r_[
                    np.linspace(0, 1, 51),
                    np.quantile(calibration.confidence[calibration.level == level], np.linspace(0, 1, 21)),
                    thresholds[level],
                ]
            )
            for level in range(3)
        ]
        # One vector call handles independent operating points at all three ranks.
        for i in range(max(map(len, grids))):
            vector = [float(grid[min(i, len(grid) - 1)]) for grid in grids]
            row["curve"].append({"thresholds": vector, "scores": measure(reporting, vector, curve=True)})
        result["models"][model] = row
        write_json(output / f"{model}.json", row)
        print(model, "thresholds", thresholds, "report/calibration", row["report_images"], row["calibration_images"], flush=True)
    result["tta_threshold_sensitivity"] = tta_threshold_sensitivity(result)
    write_json(output / "mambo-threshold-comparison.json", result)
    return result


def tta_threshold_sensitivity(data):
    """Separate backend differences from selection of near-optimal operating points."""
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroF1, evaluate_file

    result = {}
    for model in ("torch-tta", "onnx-tta"):
        row = data["models"][model]
        if file_hash(row["source"]) != row["source_sha256"]:
            raise ValueError("Changed sensitivity input")
        reporting, calibration = MetricDF.from_source(row["source"]).split((0.9, 0.1), strata=("label",), seed=42)
        calibration = calibration[calibration.level == 0]
        reporting = reporting[reporting.level == 0]
        curve = MacroF1().compute_threshold_curve(calibration)
        result[model] = {"calibration_max_f1": float(np.max(curve.values)), "operating_points": []}
        for selected_by in ("torch-tta", "onnx-tta"):
            threshold = data["models"][selected_by]["thresholds"][0]
            point = {"selected_by": selected_by, "threshold": threshold}
            for name, partition in (("calibration", calibration), ("report", reporting)):
                point[name] = finite_json(
                    evaluate_file(
                        partition,
                        threshold=threshold,
                        simple=True,
                        hierarchical=False,
                        pattern=r"^(precision|recall|f1|coverage)$",
                        verbose=0,
                    )
                )
            result[model]["operating_points"].append(point)
    return result


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"svg.fonttype": "none", "svg.hashsalt": "mambo-threshold-v1", "axes.spines.top": False, "axes.spines.right": False}
    )

    def save(fig, name):
        fig.tight_layout(rect=(0, 0.10, 1, 0.89))
        fig.text(
            0.02,
            0.015,
            "Legacy northern Europe; same reporting images for all pipelines; all truth, including unknown taxa.\n"
            "mini_metrics: Macro-F1 calibration on separate 10%; seed 42. Coverage = fraction of images accepted at each rank.\n"
            "Calibration/report split is image-level; TTA was previously selected using a subset of Flemming.",
            fontsize=9,
        )
        save_figure(fig, output, name, dpi=140)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for level, rank in enumerate(RANKS):
        k = str(level)
        for model, label, color in SERIES:
            row = data["models"][model]
            points = row["curve"]
            axes[level, 0].plot(
                [p["scores"]["recall"][k] for p in points],
                [p["scores"]["precision"][k] for p in points],
                color=color,
                label=label,
                linestyle="--" if model.startswith("onnx") else "-",
            )
            axes[level, 1].plot(
                [p["scores"]["coverage"][k] for p in points],
                [p["scores"]["micro_accuracy"][k] for p in points],
                color=color,
                linestyle="--" if model.startswith("onnx") else "-",
            )
            for name, marker in (("report_zero", "o"), ("report_optimized", "*")):
                r = row[name]
                axes[level, 0].scatter(r["recall"][k], r["precision"][k], color=color, marker=marker, s=65 if marker == "*" else 20)
                axes[level, 1].scatter(r["coverage"][k], r["micro_accuracy"][k], color=color, marker=marker, s=65 if marker == "*" else 20)
        axes[level, 0].set(
            title=rank.title(), xlabel="Macro recall (all truth retained)", ylabel="Macro precision", xlim=(0, 1), ylim=(0, 1.03)
        )
        axes[level, 1].set(
            title=rank.title(), xlabel="Image coverage", ylabel="Micro accuracy among accepted", xlim=(0, 1.02), ylim=(0, 1.03)
        )
        for ax in axes[level]:
            ax.grid(alpha=0.15)
    fig.suptitle("Confidence threshold trade-offs · circles: threshold 0 · stars: calibrated", fontsize=14)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)
    save(fig, "mambo-threshold-curves")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    labels = ["V2", "V3\nPyTorch", "V3\nONNX", "PyTorch\n+ TTA", "ONNX\n+ TTA"]
    for level, rank in enumerate(RANKS):
        for col, metric in enumerate(("f1", "coverage")):
            ax = axes[level, col]
            for offset, scope, label, color in (
                (-0.18, "report_zero", "Threshold zero", "#b8c5d0"),
                (0.18, "report_optimized", "Calibrated threshold", "#098e92"),
            ):
                values = [data["models"][model][scope][metric][str(level)] for model, _, _ in SERIES]
                bars = ax.bar(np.arange(5) + offset, values, 0.36, label=label, color=color)
                ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
            ax.set(
                title=f"{rank.title()} · {'Macro-F1' if metric == 'f1' else 'Image coverage'}",
                xticks=range(5),
                xticklabels=labels,
                ylim=(0, 1.12 if metric == "coverage" else 1),
            )
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
    fig.suptitle("Threshold optimization · same reporting partition before and after", fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    save(fig, "mambo-threshold-comparison")
    with (output / "mambo-threshold-metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["model", "rank", "scope", "threshold", *METRICS])
        for model, row in data["models"].items():
            for scope in ("report_zero", "report_optimized", "known_zero", "known_optimized", "calibration_zero", "calibration_optimized"):
                for level, rank in enumerate(RANKS):
                    writer.writerow(
                        [
                            model,
                            rank,
                            scope,
                            row["thresholds"][level] if scope.endswith("optimized") else 0,
                            *[row[scope][m][str(level)] for m in METRICS],
                        ]
                    )
    write_json(output / "mambo-threshold-comparison.json", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evidence", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence:
        collect(args.evidence, args.output)
    else:
        render(json.loads(args.data.read_text()), args.output)
