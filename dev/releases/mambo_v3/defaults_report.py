"""Historical padded-scale evidence: regional effect, complete metrics and provenance."""

import argparse
import csv
import json
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.acceleration_report import METRICS, aggregate
from dev.releases.mambo_v3.comparison_charts import completed
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION

from .figure_export import save_figure

SERIES = (
    ("v2", "MAMBO v2", "#8064a2"),
    ("torch", "V3 PyTorch", "#098e92"),
    ("onnx", "V3 ONNX", "#e8872e"),
    ("torch-tta", "V3 PyTorch + TTA", "#125351"),
    ("onnx-tta", "V3 ONNX + TTA", "#98440b"),
)


def collect(args):
    baseline = json.loads(args.baseline.read_text())
    reference = json.loads(args.reference.read_text())
    if baseline["reference"] != reference:
        raise ValueError("Baseline uses a different V2/FP32 reference")
    for backend in ("torch", "onnx"):
        report = completed(args.quality / f"{backend}-cuda-0-prediction" / "report.json")
        if report["arguments"].get("tta") != "padded_scale" or report["samples"] != 58640:
            raise ValueError("Expected full-data padded-scale TTA")
    for name in completed(args.performance / "plan.json")["completed"]:
        report = completed(args.performance / name / "report.json")
        if report["settings"].get("tta") != "padded_scale":
            raise ValueError("Expected padded-scale TTA timings")
    augmented = aggregate(args)
    for key in ("bundle_sha256", "manifest_sha256"):
        if baseline[key] != augmented[key]:
            raise ValueError("Baseline and TTA artifacts differ")
    data = {
        "tta": "padded_scale: original + 8% / 15% edge padding; FP32 mean leaf logits",
        "selection": "Chosen for highest subset macro accuracy and lower cost than D4/padded rotations; not optimal for every metric.",
        "metric_revision": REVISION,
        "policy": "threshold=0; optimal=False; simple=True; hierarchical=False",
        "quality": [r for r in reference["quality"] if r["model"] == "v2"] + baseline["quality"],
        "speed": [
            {**r, "trial_min_ips": r["batch"] * 1000 / r["trial_max_ms"], "trial_max_ips": r["batch"] * 1000 / r["trial_min_ms"]}
            for r in reference["speed"]
            if r["model"] == "v2"
        ]
        + baseline["speed"],
        "resources": baseline["resources"].copy(),
        "sources_sha256": {
            str(args.baseline): file_hash(args.baseline),
            **augmented["sources_sha256"],
        },
        "bundle_sha256": baseline["bundle_sha256"],
        "manifest_sha256": baseline["manifest_sha256"],
        "timing_bank_sha256": reference["timing_bank_sha256"],
    }
    for row in reference["resources"]:
        if row["model"] == "v2":
            data["resources"].append(
                {
                    "model": "v2",
                    "device": row["device"],
                    "rss_mib": row["rss_mib"]["median"],
                    "load_first_seconds": row["load_first_seconds"]["median"],
                    "allocated_mib": row["cuda_allocated_mib"]["median"] if row["device"] != "cpu" else None,
                }
            )
    for section in ("quality", "speed", "resources"):
        data[section].extend({**r, "model": r["model"] + "-tta"} for r in augmented[section])
    return data


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-defaults-v1", "axes.spines.top": False, "axes.spines.right": False})

    scores = {(r["model"], r["preset"]): r["scores"]["all"] for r in data["quality"]}
    rank_metrics = (("accuracy", "Macro accuracy (%)", 100), ("f1", "Macro-F1", 1))
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    steps = (("full", "europe"), ("europe", "north_europe"))
    for level, rank in enumerate(("species", "genus", "family")):
        for col, (metric, _, factor) in enumerate(rank_metrics):
            ax = axes[level, col]
            for step, (source, target) in enumerate(steps):
                changes = np.array(
                    [
                        (scores[model, target][metric][str(level)] - scores[model, source][metric][str(level)]) * factor
                        for model, _, _ in SERIES
                    ]
                )
                median = float(np.median(changes))
                ax.hlines(step, changes.min(), changes.max(), color="0.55", linewidth=3)
                for i, (_, label, color) in enumerate(SERIES):
                    ax.scatter(changes[i], step + (i - 2) * 0.055, color=color, s=35, label=label if step == 0 else None)
                ax.scatter(median, step, marker="|", s=200, color="black", zorder=5)
                ax.annotate(
                    f"median {median:+.2f}" if factor == 100 else f"median {median:+.4f}",
                    (median, step),
                    xytext=(0, 22),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
            ax.axvline(0, color="0.7", linewidth=1, linestyle=":")
            ax.set(
                title=rank.title(),
                yticks=(0, 1),
                yticklabels=("Global → Europe", "Europe → N. Europe"),
                ylim=(1.35, -0.5),
                xlabel="Macro accuracy change (percentage points)" if factor == 100 else "Macro-F1 change",
            )
            ax.margins(x=0.2)
            ax.grid(axis="x", alpha=0.15)
            ax.set_axisbelow(True)
    fig.suptitle("Effect of regional filtering · paired changes within each pipeline", fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)
    fig.text(
        0.02,
        0.015,
        "Black mark: median of five pipeline changes; grey line: min–max; coloured dots: individual pipelines.\n"
        "Descriptive spread, not confidence intervals or independent replicates. "
        "Legacy lists; same 58,640 images, all truth, threshold 0.\n"
        "Differences summarize retained mini_metrics outputs; ranks and metric scales are never pooled.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.89))
    save_figure(fig, output, "mambo-defaults-regional-effect")

    with (output / "mambo-defaults-metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["variant", "preset", "scope", "rank", "images", *METRICS])
        for row in data["quality"]:
            for scope in ("all", "known"):
                for level, rank in enumerate(("species", "genus", "family")):
                    writer.writerow(
                        [
                            row["model"],
                            row["preset"],
                            scope,
                            rank,
                            row["ranks"][rank]["images" if scope == "all" else "known_images"],
                            *[row["scores"][scope][key][str(level)] for key in METRICS],
                        ]
                    )
    write_json(output / "mambo-defaults-comparison.json", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    for name in ("baseline", "reference", "quality", "performance"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.data and any(getattr(args, key) is None for key in ("baseline", "reference", "quality", "performance")):
        parser.error("Supply --data or all four evidence inputs")
    render(json.loads(args.data.read_text()) if args.data else collect(args), args.output)
