"""Render concise local quality and performance tables from completed release evidence."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from dev.releases.mambo_v3.evaluation_data import PRESETS, write_json


def summarize(quality, benchmarks, output):
    output.mkdir(parents=True, exist_ok=False)
    quality_rows = []
    for backend in ("torch", "onnx"):
        variant = quality / f"{backend}-cuda-0-prediction"
        report = json.loads((variant / "report.json").read_text())
        if report["status"] != "complete":
            raise ValueError("Incomplete quality run")
        for preset in PRESETS:
            metrics = json.loads((variant / preset / "metrics.json").read_text())
            quality_rows.append(
                {
                    "backend": backend,
                    "preset": preset,
                    "samples": report["samples"],
                    "metrics": metrics,
                    "bundle_sha256": report["bundle_sha256"],
                    "list_sha256": report["lists"][preset]["sha256"],
                    "samples_sha256": report["sample_ids_sha256"],
                }
            )
    benchmark_plan = json.loads((benchmarks / "plan.json").read_text())
    if benchmark_plan["status"] != "complete":
        raise ValueError("Benchmark phase is incomplete")
    grouped = defaultdict(list)
    resources = []
    for variant in benchmark_plan["completed"]:
        path = benchmarks / variant / "report.json"
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError(f"Incomplete benchmark: {path}")
        settings = report["settings"]
        resources.append(
            {
                "variant": path.parent.name,
                "backend": settings["backend"],
                "device": settings["device"],
                "embeddings": settings["embeddings"],
                "threads": settings["threads"],
                "peak_rss_mib": report["peak_rss_kib_linux"] / 1024,
                "cold_first_image_seconds": report["cold_first_image_seconds"],
                "load_components_seconds": report["load_components_seconds"],
                "torch_imported": report["torch_imported"],
                "torch_peak_allocated_bytes": report.get("torch_peak_allocated_bytes"),
                "before": report["before"],
                "after": report["after"],
            }
        )
        for cell in report["cells"]:
            key = (settings["backend"], settings["device"], settings["embeddings"], settings["threads"], cell["preset"], cell["batch_size"])
            grouped[key].append(cell)
    performance = []
    for key, cells in sorted(grouped.items()):
        if len(cells) != 3:
            raise ValueError(f"Expected three trials: {key}")
        backend, device, embeddings, threads, preset, batch = key
        values = [v for cell in cells for v in cell["end_to_end"]["seconds"]]
        medians = [cell["end_to_end"]["median_seconds"] for cell in cells]
        prepared = [cell["prepared"]["median_seconds"] for cell in cells]
        performance.append(
            {
                "backend": backend,
                "device": device,
                "embeddings": embeddings,
                "threads": threads,
                "preset": preset,
                "batch": batch,
                "median_batch_ms": 1000 * statistics.median(values),
                "p95_batch_ms": 1000 * float(np.percentile(values, 95)),
                "images_per_second": batch / statistics.median(values),
                "trial_median_min_ms": 1000 * min(medians),
                "trial_median_max_ms": 1000 * max(medians),
                "prepared_batch_ms": 1000 * statistics.median(prepared),
                "observations": len(values),
            }
        )
    write_json(output / "summary.json", {"quality": quality_rows, "performance": performance, "resources": resources})
    report_dataset = json.loads((quality / "torch-cuda-0-prediction/report.json").read_text())["dataset"]
    report_samples = quality_rows[0]["samples"]
    report_species = quality_rows[0]["metrics"]["ranks"]["species"]["truth_species_or_taxa"]
    lines = [
        "# Local MAMBO release qualification",
        "",
        f"Full {report_dataset} evaluation: {report_samples:,} images / {report_species} truth species. "
        "All predictions are unthresholded. Both backends use FP32 and the same release image recipe.",
        "",
        "## Full-dataset species results",
        "",
        "| Backend | Preset | Accuracy, all | Accuracy, known | Macro-F1, all | Truth coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in quality_rows:
        metric = row["metrics"]
        rank = metric["ranks"]["species"]
        lines.append(
            f"| {row['backend']} | {row['preset']} | {rank['micro_accuracy_all']:.2%} | "
            f"{rank['micro_accuracy_known']:.2%} | {metric['all']['f1']['0']:.4f} | {rank['list_coverage']:.2%} |"
        )
    lines += [
        "",
        "## End-to-end latency and throughput",
        "",
        "Laptop measurements; updated Europe, four CPU threads, three alternating-order trials. "
        "Batch latency includes image decoding, preprocessing, transfers, hierarchy reduction and optional embeddings. "
        "p95 is descriptive of the retained observations, not a service-level guarantee.",
        "",
        "| Backend | Device | Embeddings | Batch | Median ms | p95 ms | Images/s | Trial median range ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in performance:
        if row["preset"] == "europe_v3" and row["threads"] == 4:
            lines.append(
                f"| {row['backend']} | {row['device']} | {row['embeddings']} | {row['batch']} | "
                f"{row['median_batch_ms']:.1f} | {row['p95_batch_ms']:.1f} | {row['images_per_second']:.1f} | "
                f"{row['trial_median_min_ms']:.1f}–{row['trial_median_max_ms']:.1f} |"
            )
    lines += [
        "",
        "## Startup and process memory",
        "",
        "Cold first image includes lazy load and first execution; RSS is the process high-water mark across the batch sweep.",
        "",
        "| Backend | Device | Embeddings | Median cold first image s | Peak RSS range MiB |",
        "|---|---|---|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in resources:
        if row["threads"] == 4:
            groups[(row["backend"], row["device"], row["embeddings"])].append(row)
    for (backend, device, embeddings), rows in sorted(groups.items()):
        lines.append(
            f"| {backend} | {device} | {embeddings} | "
            f"{statistics.median(r['cold_first_image_seconds'] for r in rows):.2f} | "
            f"{min(r['peak_rss_mib'] for r in rows):.0f}–{max(r['peak_rss_mib'] for r in rows):.0f} |"
        )
    lines += [
        "",
        "The machine-readable companion includes all ranks, known-only/per-class metrics, full-list and "
        "one-thread comparisons, prepared-runtime timings, raw timing counts, and resource observations. "
        "See the evaluation workflow for commands and interpretation limits.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality", "benchmarks", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.quality, args.benchmarks, args.output)
