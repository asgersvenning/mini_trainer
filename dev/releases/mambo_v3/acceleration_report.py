"""Macro-led quality and complete-pipeline speed for accelerated release defaults."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.comparison_charts import COLORS, REGION_LABELS, REGIONS, completed
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import METRIC_SCHEMA

from .figure_export import save_figure

METRICS = ("accuracy", "precision", "recall", "f1", "micro_accuracy", "theilU", "coverage")


def aggregate(args):
    reference = json.loads(args.reference.read_text())
    completed(args.quality / "plan.json")
    data = {
        "reference": reference,
        "quality": [],
        "speed": [],
        "resources": [],
        "sources_sha256": {str(args.reference): file_hash(args.reference)},
    }
    for backend in ("torch", "onnx"):
        folder = args.quality / f"{backend}-cuda-0-prediction"
        report = completed(folder / "report.json")
        for key in ("bundle_sha256", "manifest_sha256"):
            if data.setdefault(key, report[key]) != report[key]:
                raise ValueError("Different quality artifacts or manifest")
        expected = "fp16" if backend == "torch" else "tf32"
        if report["effective_precision"] != expected:
            raise ValueError("Unexpected automatic precision")
        for preset in (*REGIONS, "north_europe_v3", "europe_v3"):
            path = folder / preset / "metrics.json"
            metric = json.loads(path.read_text())
            old = next(r for r in reference["quality"] if r["model"] == "v3" and r["preset"] == preset)
            if (
                metric["metric_schema"] != METRIC_SCHEMA
                or metric["mini_metrics_revision"] != old["metric_revision"]
                or file_hash(path.with_name("mini_metric.csv")) != metric["source_sha256"]
                or metric["source_sha256"] != report["csv_sha256"][preset]
            ):
                raise ValueError("Stale metrics")
            if old["sample_ids_sha256"] != report["sample_ids_sha256"] or old["list_sha256"] != report["lists"][preset]["sha256"]:
                raise ValueError("Changed evaluation population or list")
            data["quality"].append(
                {
                    "model": backend,
                    "precision": expected,
                    "preset": preset,
                    "ranks": metric["ranks"],
                    "scores": {scope: metric[scope] for scope in ("all", "known")},
                }
            )
            data["sources_sha256"][str(path)] = file_hash(path)
        data["sources_sha256"][str(folder / "report.json")] = file_hash(folder / "report.json")
    grouped, resources = defaultdict(list), defaultdict(list)
    for name in completed(args.performance / "plan.json")["completed"]:
        path = args.performance / name / "report.json"
        report = completed(path)
        bank = hashlib.sha256(json.dumps(report["samples"], sort_keys=True).encode()).hexdigest()
        if bank != reference["timing_bank_sha256"]:
            raise ValueError("Different timing image bank")
        if any(report[key] != data[key] for key in ("bundle_sha256", "manifest_sha256")):
            raise ValueError("Timing and quality artifacts differ")
        settings = report["settings"]
        if settings["threads"] != 4 or settings["embeddings"]:
            raise ValueError("Unexpected benchmark configuration")
        backend, device = settings["backend"], settings["device"]
        expected = "fp32" if device == "cpu" else ("fp16" if backend == "torch" else "tf32")
        if settings["precision"] != "auto" or report["effective_precision"] != expected:
            raise ValueError("Unexpected timing precision")
        for cell in report["cells"]:
            grouped[backend, device, cell["preset"], cell["batch_size"]].append(cell["end_to_end"])
        resources[backend, device].append(
            {
                "rss_mib": report["peak_rss_kib_linux"] / 1024,
                "load_first_seconds": report["constructor_seconds"] + report["cold_first_image_seconds"],
                "allocated_mib": report.get("torch_peak_allocated_bytes", 0) / 2**20 if backend == "torch" and device != "cpu" else None,
            }
        )
        data["sources_sha256"][str(path)] = file_hash(path)
    expected_cells = {
        (backend, device, preset, batch)
        for backend in ("torch", "onnx")
        for device in ("cpu", "cuda:0")
        for preset in REGIONS
        for batch in ((1, 8) if device == "cpu" else (1, 8, 32))
    }
    if set(grouped) != expected_cells:
        raise ValueError("Incomplete timing matrix")
    for (backend, device, preset, batch), runs in sorted(grouped.items()):
        values = [v for run in runs for v in run["seconds"]]
        if len(runs) != 3 or len(values) != 21:
            raise ValueError("Require three complete seven-observation trials")
        data["speed"].append(
            {
                "model": backend,
                "device": device,
                "preset": preset,
                "batch": batch,
                "images_per_second": batch / statistics.median(values),
                "trial_min_ips": batch / max(r["median_seconds"] for r in runs),
                "trial_max_ips": batch / min(r["median_seconds"] for r in runs),
            }
        )
    for (backend, device), rows in resources.items():
        if len(rows) != 3:
            raise ValueError("Require three resource observations")
        data["resources"].append(
            {
                "model": backend,
                "device": device,
                **{key: statistics.median(r[key] for r in rows) if rows[0][key] is not None else None for key in rows[0]},
            }
        )
    return data


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "mambo-acceleration-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

    def save(fig, name, note):
        fig.text(0.02, 0.02, note, fontsize=9, color="#555555")
        save_figure(fig, output, name)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, backend in enumerate(("torch", "onnx")):
        for row, device in enumerate(("cpu", "cuda:0")):
            ax = axes[row, col]
            batches = (1, 8) if device == "cpu" else (1, 8, 32)
            for name, label, color in (
                ("v2", "MAMBO v2", COLORS[0]),
                (f"v3-{backend}", "Previous v3 FP32", "#888888"),
                (backend, "Updated v3 auto", COLORS[col + 1]),
            ):
                source = data["speed"] if name == backend else data["reference"]["speed"]
                rows = [
                    next(r for r in source if (r["model"], r["device"], r["preset"], r["batch"]) == (name, device, "north_europe", batch))
                    for batch in batches
                ]
                values = [r["images_per_second"] for r in rows]
                low = [r.get("trial_min_ips", r["batch"] * 1000 / r["trial_max_ms"] if "trial_max_ms" in r else 0) for r in rows]
                high = [r.get("trial_max_ips", r["batch"] * 1000 / r["trial_min_ms"] if "trial_min_ms" in r else 0) for r in rows]
                ax.plot(range(len(batches)), values, marker="o", color=color, label=label)
                ax.vlines(range(len(batches)), low, high, colors=color, alpha=0.5)
                ax.annotate(
                    f"{values[-1]:.1f}",
                    (len(values) - 1, values[-1]),
                    xytext=(4, -14 if name.startswith("v3-") else 5),
                    textcoords="offset points",
                    fontsize=9,
                )
            ax.set(
                xticks=range(len(batches)),
                xticklabels=batches,
                xlabel="Batch size",
                ylabel="End-to-end images / second",
                title=f"{backend.title()} · {'CPU' if device == 'cpu' else 'GPU'}",
                ylim=(0, None),
            )
            ax.margins(y=0.25)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8)
    for row in range(2):
        maximum = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(0, maximum)
    fig.suptitle("Northern Europe: complete-pipeline batch throughput", fontsize=16)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    save(
        fig,
        "mambo-accelerated-speed",
        "Three fresh-process trials; four CPU threads; decode, preparation and CPU results included. Whiskers: trial-median range.\n"
        "Updated CUDA: FP16 backbone for PyTorch, TF32 for standard ONNX. CPU: FP32. V2 CPU requires its documented input cast.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ["V2", "V3 torch\nFP32", "V3 torch\nauto", "V3 ONNX\nFP32", "V3 ONNX\nauto"]
    for ax, device in zip(axes, ("cpu", "cuda:0"), strict=True):
        values = []
        for name in ("v2", "v3-torch", "torch", "v3-onnx", "onnx"):
            updated = name in ("torch", "onnx")
            source = data["resources"] if updated else data["reference"]["resources"]
            value = next(r for r in source if r["model"] == name and r["device"] == device)["rss_mib"]
            values.append(value if updated else value["median"])
        bars = ax.bar(labels, values, color=[COLORS[0], "#888888", COLORS[1], "#888888", COLORS[2]])
        ax.bar_label(bars, fmt="%.0f", padding=3)
        ax.set(title="CPU execution" if device == "cpu" else "GPU execution", ylabel="Peak host RSS (MiB)", ylim=(0, 4800))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Process memory across the complete batch sweep", fontsize=15)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    save(
        fig,
        "mambo-accelerated-memory",
        "Median of three fresh processes; includes loading and batch sweep. Host memory, not GPU VRAM.\n"
        "CPU batches 1/8; GPU batches 1/8/32. V2 CPU uses the documented input cast.",
    )

    series = [
        ("v2", "MAMBO v2", COLORS[0], data["reference"]["quality"]),
        ("v3", "V3 FP32 reference", "#888888", data["reference"]["quality"]),
        ("torch", "V3 PyTorch FP16", COLORS[1], data["quality"]),
        ("onnx", "V3 ONNX TF32", COLORS[2], data["quality"]),
    ]
    for scope in ("all", "known"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        for ax, (metric, title) in zip(
            axes.ravel(),
            (("accuracy", "Macro accuracy"), ("f1", "Macro-F1"), ("precision", "Macro precision"), ("micro_accuracy", "Micro accuracy")),
            strict=True,
        ):
            for i, (model, label, color, source) in enumerate(series):
                values = [
                    next(r for r in source if r["model"] == model and r["preset"] == preset)["scores"][scope][metric]["0"]
                    for preset in REGIONS
                ]
                bars = ax.bar(np.arange(3) + (i - 1.5) * 0.2, values, 0.2, color=color, label=label)
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, rotation=45)
            ax.set(title=title, xticks=np.arange(3), xticklabels=REGION_LABELS, ylim=(0, 1.05))
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, fontsize=9, frameon=False
        )
        fig.suptitle(f"Species quality · {scope} truth · automatic GPU settings", fontsize=16)
        fig.tight_layout(rect=(0, 0.08, 1, 0.88))
        save(
            fig,
            f"mambo-accelerated-quality-{scope}",
            "Pinned mini_metrics; threshold 0, no optimization. All: 58,640 images; known: 50,598 images.\n"
            "Full tables retain macro recall, Theil U, coverage and every taxonomic rank, including updated European lists.",
        )
    with (output / "mambo-accelerated-metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["variant", "preset", "scope", "rank", "images", *METRICS])
        for model, label, color, source in series:
            for entry in (r for r in source if r["model"] == model):
                for scope in ("all", "known"):
                    for level, rank in enumerate(("species", "genus", "family")):
                        writer.writerow(
                            [
                                label,
                                entry["preset"],
                                scope,
                                rank,
                                entry["ranks"][rank]["images" if scope == "all" else "known_images"],
                                *[entry["scores"][scope][key][str(level)] for key in METRICS],
                            ]
                        )
    write_json(output / "mambo-accelerated-comparison.json", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    for name in ("reference", "quality", "performance"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.data and any(getattr(args, key) is None for key in ("reference", "quality", "performance")):
        parser.error("Supply --data or all three evidence inputs")
    render(json.loads(args.data.read_text()) if args.data else aggregate(args), args.output)
