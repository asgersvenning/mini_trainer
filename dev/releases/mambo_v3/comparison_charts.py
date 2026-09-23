"""Build readable release-comparison SVGs and their compact, auditable source data."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import METRIC_SCHEMA

REGIONS = ("north_europe", "europe", "full")
REGION_LABELS = ("Northern Europe", "Europe", "Global")
MODELS = ("v2", "v3-torch", "v3-onnx")
MODEL_LABELS = ("MAMBO v2 · PyTorch", "MAMBO v3 · PyTorch", "MAMBO v3 · ONNX")
COLORS = ("#7b629c", "#168b89", "#df8739")


def completed(path):
    report = json.loads(path.read_text())
    if report["status"] != "complete":
        raise ValueError(f"Incomplete evidence: {path}")
    return report


def aggregate(args):
    sources = {}
    quality = []
    paired_path = args.v3_quality / "comparison.json"
    paired = json.loads(paired_path.read_text())
    reports = [args.v3_quality / f"{backend}-cuda-0-prediction/report.json" for backend in ("torch", "onnx")]
    if paired["reports_sha256"] != [file_hash(path) for path in reports]:
        raise ValueError("V3 backend agreement evidence is stale")
    if any(rank["changed"] for preset in paired["presets"].values() for rank in preset.values()):
        raise ValueError("V3 backend labels differ; plot their quality separately")
    sources[str(paired_path)] = file_hash(paired_path)
    for model, directory, presets in (
        ("v2", args.v2_quality / "v2-full", REGIONS),
        ("v3", args.v3_quality / "torch-cuda-0-prediction", (*REGIONS, "north_europe_v3", "europe_v3")),
    ):
        report = completed(directory / "report.json")
        sources[str(directory / "report.json")] = file_hash(directory / "report.json")
        for preset in presets:
            path = directory / preset / "metrics.json"
            metrics = json.loads(path.read_text())
            if file_hash(directory / preset / "mini_metric.csv") != metrics["source_sha256"]:
                raise ValueError("Metric source has changed")
            if metrics.get("metric_schema") != METRIC_SCHEMA:
                raise ValueError("Recompute quality using the mini_metrics-only extractor")
            sources[str(path)] = file_hash(path)
            quality.append(
                {
                    "model": model,
                    "preset": preset,
                    "ranks": metrics["ranks"],
                    "macro_f1_all": metrics["all"]["f1"]["0"],
                    "scores": {scope: metrics[scope] for scope in ("all", "known")},
                    "metric_revision": metrics["mini_metrics_revision"],
                    "sample_ids_sha256": report["sample_ids_sha256"],
                    "list_sha256": report["lists"][preset]["sha256"],
                }
            )
    if len({r["sample_ids_sha256"] for r in quality}) != 1 or len({r["metric_revision"] for r in quality}) != 1:
        raise ValueError("Quality populations or metric revisions differ")
    for preset in REGIONS:
        selected = [r for r in quality if r["preset"] == preset]
        if len({r["list_sha256"] for r in selected}) != 1:
            raise ValueError(f"Primary comparison lists differ: {preset}")
    grouped = defaultdict(list)
    resources = defaultdict(list)
    components = defaultdict(list)
    bank = None
    for directory, added in ((args.v3_performance, False), (args.added_performance, True)):
        plan = completed(directory / "plan.json")
        for variant in plan["completed"]:
            path = directory / variant / "report.json"
            report = completed(path)
            settings = report["settings"]
            if settings.get("embeddings", False) or settings["threads"] != 4:
                continue
            old = report.get("release") == "MAMBO_v2"
            model = "v2" if old else f"v3-{settings['backend']}"
            device = settings["device"]
            if old and device == "cpu" and not report.get("cpu_input_cast"):
                raise ValueError("V2 CPU evidence must explicitly record the input adapter")
            records = json.loads((path.parent / "samples.json").read_text()) if old else report["samples"]
            if bank is None:
                bank = records
            if records != bank:
                raise ValueError("Timing image banks differ")
            if metrics.get("metric_schema") != METRIC_SCHEMA:
                raise ValueError("Recompute quality using the mini_metrics-only extractor")
            sources[str(path)] = file_hash(path)
            for cell in report["cells"]:
                grouped[(model, device, cell["preset"], cell["batch_size"])].append(cell["end_to_end"])
                if not old and device == "cuda:0" and cell["preset"] == "north_europe":
                    for stage in ("preprocessing", "prepared", "end_to_end"):
                        components[(model, cell["batch_size"], stage)].extend(cell[stage]["seconds"])
            # Use the full-list-containing sweep for comparable whole-process memory.
            if old or not added:
                resources[(model, device)].append(
                    {
                        "load_first_seconds": report["load_and_first_image_seconds"]
                        if old
                        else report["constructor_seconds"] + report["cold_first_image_seconds"],
                        "rss_mib": report["peak_rss_kib_linux"] / 1024,
                        "cuda_allocated_mib": report.get("torch_peak_allocated_bytes", 0) / 2**20 if model != "v3-onnx" else None,
                        "cuda_reserved_mib": report.get("torch_peak_reserved_bytes", 0) / 2**20 if model != "v3-onnx" else None,
                    }
                )
    speed = []
    for (model, device, preset, batch), runs in sorted(grouped.items()):
        if len(runs) != 3:
            raise ValueError(f"Expected three timing trials: {(model, device, preset, batch)}")
        values = [value for run in runs for value in run["seconds"]]
        if len(values) != 21:
            raise ValueError("Expected 21 timing observations")
        medians = [run["median_seconds"] for run in runs]
        speed.append(
            {
                "model": model,
                "device": device,
                "preset": preset,
                "batch": batch,
                "median_ms": statistics.median(values) * 1000,
                "p95_ms": float(np.percentile(values, 95)) * 1000,
                "trial_min_ms": min(medians) * 1000,
                "trial_max_ms": max(medians) * 1000,
                "images_per_second": batch / statistics.median(values),
            }
        )
    memory = []
    for (model, device), runs in sorted(resources.items()):
        if len(runs) != 3:
            raise ValueError("Expected three resource trials")
        row = {"model": model, "device": device}
        for key in runs[0]:
            values = [r[key] for r in runs]
            row[key] = None if values[0] is None else {"median": statistics.median(values), "min": min(values), "max": max(values)}
        memory.append(row)
    return {
        "quality": quality,
        "speed": speed,
        "resources": memory,
        "v3_gpu_components": [
            {"model": model, "batch": batch, "stage": stage, "observations": len(values), "median_ms": statistics.median(values) * 1000}
            for (model, batch, stage), values in sorted(components.items())
        ],
        "quality_policy": {
            "accuracy": "Lead: mini_metrics accuracy (macro); micro_accuracy retained; all and known scopes at each rank",
            "macro_f1": "mini_metrics all.f1.0; equal weight over union of truth and predicted species",
            "threshold": 0,
            "optimal": False,
            "known_only": False,
            "known_accuracy": "mini_metrics known.micro_accuracy; known_only=True; truth in active preset vocabulary",
        },
        "sources_sha256": sources,
        "timing_bank_sha256": hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest(),
        "protocol": "Same laptop and image bank; original v2 preprocessing/CUDA autocast; "
        "v2 CPU requires float32 input cast; v3 FP32; four threads; three trials; predictions only.",
    }


def charts(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "mambo-release-comparison-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name, note):
        fig.text(0.02, 0.025, note, fontsize=9, color="#555555")
        svg = output / f"{name}.svg"
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        fig.savefig(output / f"{name}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.ravel()
    x = np.arange(3)
    for m, (model, label, color) in enumerate(zip(("v2", "v3"), ("MAMBO v2", "MAMBO v3 (both backends)"), COLORS, strict=False)):
        rows = [next(r for r in data["quality"] if r["model"] == model and r["preset"] == region) for region in REGIONS]
        values_by_rank = (
            [100 * r["ranks"]["species"]["macro_accuracy_all"] for r in rows],
            [r["macro_f1_all"] for r in rows],
            [100 * r["ranks"]["genus"]["macro_accuracy_all"] for r in rows],
            [100 * r["ranks"]["family"]["macro_accuracy_all"] for r in rows],
        )
        for index, (ax, values) in enumerate(zip(axes, values_by_rank, strict=True)):
            bars = ax.bar(x + (m - 0.5) * 0.34, values, 0.34, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f" if index == 1 else "%.2f", padding=3, fontsize=9)
            ax.set_xticks(x, REGION_LABELS)
            ax.grid(axis="y", alpha=0.16)
            ax.set_axisbelow(True)
    axes[0].set(title="Species macro accuracy · all truth classes", ylabel="Mean class accuracy (%)", ylim=(0, 100))
    axes[1].set(
        title="Species macro-F1 · truth ∪ predicted classes",
        ylabel="Pinned mini_metrics macro-F1",
        ylim=(0, max(r["macro_f1_all"] for r in data["quality"]) * 1.3),
    )
    axes[2].set(title="Genus macro accuracy · all truth classes", ylabel="Mean class accuracy (%)", ylim=(0, 100))
    axes[3].set(title="Family macro accuracy · all truth classes", ylabel="Mean class accuracy (%)", ylim=(0, 100))
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Flemming: MAMBO v2 versus v3", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save(
        fig,
        "mambo-release-quality",
        "58,640 images · threshold 0 (no abstention or optimization) · known_only=False · unknown truth included\n"
        "V2: original CUDA autocast / BioCLIP recipe. V3: FP32 / release recipe. Real-world comparison of the two release pipelines.",
    )

    metric_panels = (
        ("accuracy", "Macro accuracy"),
        ("precision", "Macro precision"),
        ("recall", "Macro recall"),
        ("f1", "Macro-F1"),
        ("micro_accuracy", "Micro accuracy"),
        ("theilU", "Theil U"),
    )
    for scope in ("all", "known"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        for ax, (metric, title) in zip(axes.ravel(), metric_panels, strict=True):
            for m, (model, label, color) in enumerate(zip(("v2", "v3"), ("MAMBO v2", "MAMBO v3 (both backends)"), COLORS, strict=False)):
                rows = [next(r for r in data["quality"] if r["model"] == model and r["preset"] == region) for region in REGIONS]
                values = [r["scores"][scope][metric]["0"] for r in rows]
                bars = ax.bar(x + (m - 0.5) * 0.34, values, 0.34, label=label, color=color)
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.set(title=title, xticks=x, xticklabels=REGION_LABELS, ylim=(0, 1.1))
            ax.grid(axis="y", alpha=0.16)
            ax.set_axisbelow(True)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
        fig.suptitle("Species baseline · " + ("all Flemming truth" if scope == "all" else "truth within active vocabulary"), fontsize=16)
        fig.tight_layout(rect=(0, 0.1, 1, 0.89))
        save(
            fig,
            f"mambo-release-species-{scope}",
            "Pinned mini_metrics · threshold 0 · no optimization · " + ("58,640 images" if scope == "all" else "50,598 images") + "\n"
            "Macro accuracy/recall: truth classes. Precision: predicted classes. F1: their union. "
            "Coverage is 1.0; full rank tables supplied.",
        )
    with (output / "mambo-release-metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "model",
                "preset",
                "scope",
                "rank",
                "images",
                "macro_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "micro_accuracy",
                "theilU",
                "coverage",
            ]
        )
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
                            *[
                                row["scores"][scope][key][str(level)]
                                for key in ("accuracy", "precision", "recall", "f1", "micro_accuracy", "theilU", "coverage")
                            ],
                        ]
                    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    axes = axes.ravel()
    for ax, device, batch in zip(axes, ("cpu", "cuda:0", "cuda:0", "cuda:0"), (1, 1, 8, 32), strict=True):
        for m, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, COLORS, strict=True)):
            rows = [
                next(r for r in data["speed"] if (r["model"], r["device"], r["preset"], r["batch"]) == (model, device, region, batch))
                for region in REGIONS
            ]
            values = [r["images_per_second"] for r in rows]
            low = [batch * 1000 / r["trial_max_ms"] for r in rows]
            high = [batch * 1000 / r["trial_min_ms"] for r in rows]
            positions = x + (m - 1) * 0.25
            ax.bar(positions, values, 0.25, label=label, color=color)
            ax.errorbar(
                positions, (np.array(low) + high) / 2, yerr=(np.array(high) - low) / 2, fmt="none", ecolor="#333333", capsize=3, linewidth=1
            )
            for position, value, upper in zip(positions, values, high, strict=True):
                ax.annotate(
                    f"{value:.1f}",
                    (position, max(value, upper)),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x, REGION_LABELS)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        ax.margins(y=0.25)
    axes[0].set(title="CPU · one image · v2 with input adapter", ylabel="End-to-end images / second")
    axes[1].set(title="GPU · one image · higher is better", ylabel="End-to-end images / second")
    axes[2].set(title="GPU · batch 8 · higher is better", ylabel="End-to-end images / second")
    axes[3].set(title="GPU · batch 32 · higher is better", ylabel="End-to-end images / second")
    gpu_limit = max(r["batch"] * 1000 / r["trial_min_ms"] for r in data["speed"] if r["device"] == "cuda:0") * 1.2
    for ax in axes[1:]:
        ax.set_ylim(0, gpu_limit)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.suptitle("Laptop inference throughput · higher is better", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.90))
    save(
        fig,
        "mambo-release-speed",
        "i7-12800H / RTX 3080 Ti Laptop · four CPU threads · three trials · decode, preprocessing and CPU results included\n"
        "Whiskers: trial-median range. V2 CPU requires a float32 input cast; GPU uses published autocast. V3 uses FP32.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, device in enumerate(("cpu", "cuda:0")):
        for ax, key in zip(axes[:, col], ("rss_mib", "load_first_seconds"), strict=True):
            rows = [next(r for r in data["resources"] if (r["model"], r["device"]) == (model, device))[key] for model in MODELS]
            values = [r["median"] for r in rows]
            bars = ax.bar(
                np.arange(3),
                values,
                color=COLORS,
                yerr=[[v - r["min"] for v, r in zip(values, rows, strict=True)], [r["max"] - v for v, r in zip(values, rows, strict=True)]],
                capsize=4,
            )
            ax.bar_label(bars, fmt="%.2f" if key == "load_first_seconds" else "%.0f", padding=6)
            ax.set_xticks(np.arange(3), ("v2 + input cast" if device == "cpu" else "v2 PyTorch", "v3 PyTorch", "v3 ONNX"))
            ax.grid(axis="y", alpha=0.16)
            ax.set_axisbelow(True)
            ax.margins(y=0.3)
            if key == "load_first_seconds":
                ax.set_yscale("log")
                ax.set_ylabel("Seconds · logarithmic scale")
            else:
                ax.set_ylabel("Process peak RSS (MiB)")
        axes[0, col].set_title(f"{'CPU' if device == 'cpu' else 'GPU'} execution · host memory")
        axes[1, col].set_title("Local load + first completed prediction")
    fig.suptitle("Memory and startup costs", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    save(
        fig,
        "mambo-release-resources",
        "RSS includes loading and the batch sweep; host memory is not GPU VRAM. Whiskers: three-trial range.\n"
        "Startup uses cached local files, excludes process/bootstrap setup and downloads; includes classifier initialization.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, field, title in zip(
        axes, ("macro_accuracy_all", "micro_accuracy_all"), ("Macro species accuracy", "Micro species accuracy"), strict=True
    ):
        for i, region in enumerate(REGIONS[:2]):
            rows = [next(r for r in data["quality"] if r["model"] == "v3" and r["preset"] == region + suffix) for suffix in ("", "_v3")]
            old, new = [100 * r["ranks"]["species"][field] for r in rows]
            delta = new - old
            ax.barh(i, delta, height=0.5, color=COLORS[1])
            ax.text(0.02, i, f"{delta:+.2f} pp", va="center", fontsize=10)
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set(
            yticks=[0, 1], yticklabels=REGION_LABELS[:2], xlim=(-0.9, 0.25), xlabel="Updated minus legacy (percentage points)", title=title
        )
        ax.invert_yaxis()
    fig.tight_layout(rect=(0, 0.18, 1, 0.93))
    fig.suptitle("V3 updated lists: accuracy trade-offs on Flemming", fontweight="bold")
    save(
        fig,
        "mambo-release-preset-delta",
        "Northern Europe adds 222 candidate species; Europe adds 72. No removals. Choose lists by geographic scope.\n"
        "All 58,640 images; threshold 0, no optimization. Both backends agree; filters were not tuned to Flemming.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Render previously aggregated chart data without private raw predictions")
    for name in ("v2-quality", "v3-quality", "v3-performance", "added-performance"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data:
        data = json.loads(args.data.read_text())
    else:
        if any(getattr(args, name) is None for name in ("v2_quality", "v3_quality", "v3_performance", "added_performance")):
            parser.error("Supply all four evidence directories or --data")
        data = aggregate(args)
    charts(data, args.output)
    write_json(args.output / "mambo-release-comparison.json", data)


if __name__ == "__main__":
    main()
