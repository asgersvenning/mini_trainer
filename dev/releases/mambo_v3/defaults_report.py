"""Release overview: V2, automatic V3 and the enabled-TTA default on full Flemming."""

import argparse
import csv
import json
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.acceleration_report import METRICS, aggregate
from dev.releases.mambo_v3.comparison_charts import REGION_LABELS, REGIONS, completed
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION

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

    def save(fig, name):
        svg = output / f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    for level, rank in enumerate(("species", "genus", "family")):
        for scope in ("all", "known"):
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            for ax, (metric, title) in zip(
                axes.flat,
                (
                    ("accuracy", "Macro accuracy (%)"),
                    ("f1", "Macro-F1"),
                    ("precision", "Macro precision"),
                    ("micro_accuracy", "Micro accuracy (%)"),
                ),
                strict=True,
            ):
                factor = 100 if "accuracy" in metric else 1
                for i, (model, label, color) in enumerate(SERIES):
                    values = [
                        next(r for r in data["quality"] if (r["model"], r["preset"]) == (model, preset))["scores"][scope][metric][
                            str(level)
                        ]
                        * factor
                        for preset in REGIONS
                    ]
                    bars = ax.bar(np.arange(3) + (i - 2) * 0.16, values, 0.16, label=label, color=color)
                    ax.bar_label(bars, fmt="%.1f" if factor == 100 else "%.3f", rotation=60, fontsize=8, padding=3)
                upper = 105 if factor == 100 else min(1, max(bar.get_height() for bar in ax.patches) * 1.35)
                ax.set(title=title, xticks=np.arange(3), xticklabels=REGION_LABELS, ylim=(0, upper))
                ax.grid(axis="y", alpha=0.15)
                ax.set_axisbelow(True)
            fig.suptitle(f"MAMBO release comparison · {rank} metrics · {scope} truth", fontsize=16)
            fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
            fig.text(
                0.02,
                0.015,
                "All: 58,640 images. Known: species 50,598; genus 58,639–58,640 by preset; family 58,640.\n"
                "TTA: original + two padded views; selected on a Flemming subset, not independently validated.\n"
                "Same legacy vocabularies across releases; tables also retain updated European lists and both truth populations.",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.10, 1, 0.88))
            suffix = "" if rank == "species" else f"-{rank}"
            save(fig, f"mambo-defaults-quality{suffix}-{scope}")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for level, rank in enumerate(("species", "genus", "family")):
        for col, (metric, title, factor) in enumerate((("accuracy", "Macro accuracy (%)", 100), ("f1", "Macro-F1", 1))):
            ax = axes[level, col]
            for i, (model, label, color) in enumerate(SERIES):
                values = [
                    next(r for r in data["quality"] if (r["model"], r["preset"]) == (model, preset))["scores"]["all"][metric][str(level)]
                    * factor
                    for preset in REGIONS
                ]
                bars = ax.bar(np.arange(3) + (i - 2) * 0.16, values, 0.16, label=label, color=color)
                ax.bar_label(bars, fmt="%.1f" if factor == 100 else "%.3f", rotation=60, fontsize=8, padding=3)
            upper = 105 if factor == 100 else min(1, max(bar.get_height() for bar in ax.patches) * 1.35)
            ax.set(title=f"{rank.title()} · {title}", xticks=np.arange(3), xticklabels=REGION_LABELS, ylim=(0, upper))
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
    fig.suptitle("MAMBO release comparison · species, genus and family", fontsize=16)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.text(
        0.02,
        0.015,
        "All truth: 58,640 images; 522 species / 322 genera / 23 families. Pinned mini_metrics; threshold 0.\n"
        "Macro accuracy weights truth taxa equally; macro-F1 also includes predicted-only taxa. V3 uses automatic GPU precision.\n"
        "TTA: original + two padded views; recipe selected on a subset of Flemming, not independently validated.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.925))
    save(fig, "mambo-defaults-ranks-all")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, device, batches in zip(axes, ("cpu", "cuda:0"), ((1, 8), (1, 8, 32)), strict=True):
        for model, label, color in SERIES:
            rows = [
                next(r for r in data["speed"] if (r["model"], r["device"], r["preset"], r["batch"]) == (model, device, "north_europe", b))
                for b in batches
            ]
            values = np.array([r["images_per_second"] for r in rows])
            lo = np.array([r["trial_min_ips"] for r in rows])
            hi = np.array([r["trial_max_ips"] for r in rows])
            ax.errorbar(
                range(len(batches)),
                values,
                yerr=[values - lo, hi - values],
                marker="o",
                capsize=3,
                color=color,
                label=label,
                linestyle="--" if model.endswith("tta") else "-",
            )
        ax.set(
            title="CPU · FP32" if device == "cpu" else "GPU · automatic precision",
            xlabel="Images per batch",
            ylabel="End-to-end images / second",
            xticks=range(len(batches)),
            xticklabels=batches,
            ylim=(0, None),
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Complete-pipeline throughput · northern Europe", fontsize=16)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.text(
        0.02,
        0.015,
        "i7-12800H / RTX 3080 Ti Laptop, WSL2; four CPU/preparation threads; same image bank.\n"
        "Three fresh processes × seven observations; bars show trial-median range. Decode through CPU results included.\n"
        "V2 and unaugmented V3 reuse recorded measurements; laptop conditions vary between campaigns. V2 CPU uses input cast.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.85))
    save(fig, "mambo-defaults-speed")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, device in zip(axes, ("cpu", "cuda:0"), strict=True):
        values = [next(r for r in data["resources"] if (r["model"], r["device"]) == (m, device))["rss_mib"] for m, _, _ in SERIES]
        bars = ax.bar(range(5), values, color=[color for _, _, color in SERIES])
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set(
            title="CPU execution" if device == "cpu" else "GPU execution",
            ylabel="Peak host RSS (MiB)",
            xticks=range(5),
            xticklabels=["V2", "V3\ntorch", "V3\nONNX", "torch\n+ TTA", "ONNX\n+ TTA"],
            ylim=(0, max(values) * 1.15),
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Process memory · median of three fresh processes", fontsize=15)
    fig.text(
        0.02,
        0.02,
        "Host memory, not GPU VRAM; includes loading and complete CPU 1/8 or GPU 1/8/32 batch sweep.\n"
        "TTA retains original decoded images for each batch; memory depends on source dimensions.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, "mambo-defaults-memory")

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
