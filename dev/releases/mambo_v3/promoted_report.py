"""Publish the selected TTA comparison from verified full-run evidence and fresh timings."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from deployment.mambo_deploy.augmentation import DEFAULT_TTA
from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.defaults_report import SERIES
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.tail_report import collect

from .figure_export import save_figure


def quality(source, output):
    data = json.loads(source.read_text())
    study = {"tta": DEFAULT_TTA, "source_sha256": file_hash(source), "revision": data["revision"], "models": {}}
    for model, _, _ in SERIES:
        key = model.replace("-tta", f":{DEFAULT_TTA}")
        row = data["models"][key]
        study["models"][model] = {
            **row,
            "report_zero": row["operating_points"]["zero"]["full"],
            "report_optimized": row["operating_points"]["optimized"]["full"],
        }
        # Eleven-model tail domains belong to the exploratory comparison, not this five-model table.
        study["models"][model].pop("operating_points")
    tails = collect(study)
    tails["tta"] = DEFAULT_TTA
    write_json(output / "mambo-promoted-thresholds.json", study)
    write_json(output / "mambo-promoted-tail.json", tails)
    rows = tails["rows"]
    with (output / "mambo-promoted-tail.csv").open("w", newline="") as stream:
        flattened = [{**{k: v for k, v in r.items() if k not in ("metrics", "classes")}, **r["metrics"]} for r in rows]
        writer = csv.DictWriter(stream, fieldnames=list(flattened[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)
    lookup = {(r["model"], r["scope"], r["rank"], r["cutoff"], r["domain"]): r for r in rows}
    text = ""
    for rank in ("species", "genus", "family"):
        text += f"### {rank.title()}\n\n"
        text += "| Pipeline | Confidence | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |\n|---|---|---:|---:|---:|\n"
        for model, label, _ in SERIES:
            for scope, name in (("zero", "None"), ("optimized", "Calibrated")):
                full = lookup[model, scope, rank, -1, "per_model"]
                tail = lookup[model, scope, rank, 5, "common"]
                a, b = full["metrics"], tail["metrics"]
                text += (
                    f"| {label} | {name} | {a['accuracy']:.2%} / {b['accuracy']:.2%} | "
                    f"{a['f1']:.4f} / {b['f1']:.4f} | {full['overall_coverage']:.2%} |\n"
                )
        text += "\n"
    text += "### Support retained and comparison figure\n\n"
    text += (
        "Counts below describe truth classes outside the truncated average, not rejected images.\n"
        "The prediction range counts accepted predictions into excluded classes, divided by all\n"
        "52,788 reporting images. Truth and prediction counts must not be added.\n\n"
        "| Confidence | Rank | Shared classes | Truth outside: images / % | Accepted predictions outside: images / % (pipeline range) |\n"
        "|---|---|---:|---:|---:|\n"
    )
    for scope, name in (("zero", "None"), ("optimized", "Calibrated")):
        for rank in ("species", "genus", "family"):
            selected = [lookup[m, scope, rank, 5, "common"] for m, _, _ in SERIES]
            row = selected[0]
            n = row["report_images"]
            truth = n - row["truth_images_in_retained_classes"]
            outside = [round(r["overall_coverage"] * n) - r["accepted_predictions_in_retained_classes"] for r in selected]
            lo, hi = min(outside), max(outside)
            text += (
                f"| {name} | {rank.title()} | {row['class_count']} | {truth:,} / {truth / n:.2%} | "
                f"{lo:,}–{hi:,} / {lo / n:.2%}–{hi / n:.2%} |\n"
            )
    (output / "quality-tables.md").write_text(text)


def performance(root, baseline, output):
    old = json.loads(baseline.read_text())
    plan = json.loads((root / "plan.json").read_text())
    if plan["status"] != "complete" or len(plan["completed"]) != 12:
        raise ValueError("Require all twelve fresh-process trials")
    grouped, resources, provenance = defaultdict(list), defaultdict(list), {}
    for name in plan["completed"]:
        path = root / name / "report.json"
        r = json.loads(path.read_text())
        s = r["settings"]
        bank = hashlib.sha256(json.dumps(r["samples"], sort_keys=True).encode()).hexdigest()
        if r["status"] != "complete" or s["tta"] != DEFAULT_TTA or bank != old["timing_bank_sha256"]:
            raise ValueError("Changed recipe or timing bank")
        if any(r[k] != old[k] for k in ("bundle_sha256", "manifest_sha256")):
            raise ValueError("Changed benchmark inputs")
        if s["threads"] != 4 or s["embeddings"] or s["precision"] != "auto":
            raise ValueError("Unexpected timing configuration")
        model, device = s["backend"] + "-tta", s["device"]
        for cell in r["cells"]:
            if cell["preset"] != "north_europe":
                raise ValueError("Expected northern Europe")
            grouped[model, device, cell["batch_size"]].append(cell["end_to_end"])
        resources[model, device].append(r["peak_rss_kib_linux"] / 1024)
        provenance[str(path)] = file_hash(path)
    data = {
        "tta": DEFAULT_TTA,
        "sources_sha256": {str(baseline): file_hash(baseline), **provenance},
        "speed": [r for r in old["speed"] if not r["model"].endswith("-tta") and r["preset"] == "north_europe"],
        "resources": [r for r in old["resources"] if not r["model"].endswith("-tta")],
    }
    expected = {(m, d, b) for m in ("torch-tta", "onnx-tta") for d in ("cpu", "cuda:0") for b in ((1, 8) if d == "cpu" else (1, 8, 32))}
    if set(grouped) != expected:
        raise ValueError("Incomplete benchmark matrix")
    for (model, device, batch), trials in grouped.items():
        if len(trials) != 3 or any(len(t["seconds"]) != 7 for t in trials):
            raise ValueError("Require three trials of seven observations")
        values = [v for t in trials for v in t["seconds"]]
        data["speed"].append(
            {
                "model": model,
                "device": device,
                "batch": batch,
                "images_per_second": batch / statistics.median(values),
                "trial_min_ips": batch / max(t["median_seconds"] for t in trials),
                "trial_max_ips": batch / min(t["median_seconds"] for t in trials),
            }
        )
    for (model, device), values in resources.items():
        data["resources"].append({"model": model, "device": device, "rss_mib": statistics.median(values)})
    write_json(output / "mambo-promoted-speed.json", data)
    text = "| Pipeline | CPU B1 | GPU B1 | GPU B8 | GPU B32 |\n|---|---:|---:|---:|---:|\n"
    for model, label, _ in SERIES:
        values = [
            next(r["images_per_second"] for r in data["speed"] if (r["model"], r["device"], r["batch"]) == (model, device, batch))
            for device, batch in (("cpu", 1), ("cuda:0", 1), ("cuda:0", 8), ("cuda:0", 32))
        ]
        text += f"| {label} | " + " | ".join(f"{v:.2f}" for v in values) + " |\n"
    (output / "speed-table.md").write_text(text)


def render_speed(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-promoted-speed-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, device, batches in zip(axes, ("cpu", "cuda:0"), ((1, 8), (1, 8, 32)), strict=True):
        for model, label, color in SERIES:
            rows = [next(r for r in data["speed"] if (r["model"], r["device"], r["batch"]) == (model, device, b)) for b in batches]
            values = [r["images_per_second"] for r in rows]
            ax.errorbar(
                range(len(batches)),
                values,
                yerr=[
                    [max(0, v - r["trial_min_ips"]) for v, r in zip(values, rows)],
                    [max(0, r["trial_max_ips"] - v) for v, r in zip(values, rows)],
                ],
                marker="o",
                capsize=3,
                color=color,
                label=label,
            )
        ax.set(
            title="CPU · FP32" if device == "cpu" else "GPU · automatic precision",
            xlabel="Images per batch",
            ylabel="End-to-end images / second",
            xticks=range(len(batches)),
            xticklabels=batches,
            ylim=(0, None),
        )
        ax.grid(alpha=0.15)
    fig.suptitle("Release throughput · northern Europe · TTA: ±30° with 25% padding")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3)
    fig.text(
        0.03,
        0.02,
        "i7-12800H / RTX 3080 Ti Laptop; four preparation/runtime threads; same image bank.\n"
        "Three fresh processes × seven observations; bars: trial-median range. Decode through CPU results included.\n"
        "V2 and single-view V3 reuse earlier measurements; laptop conditions vary between campaigns.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.84))
    save_figure(fig, output, "mambo-promoted-speed", dpi=140)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("docs/assets/mambo-defaults-comparison.json"))
    parser.add_argument("--render-speed", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.quality:
        quality(args.quality, args.output)
    if args.performance:
        performance(args.performance, args.baseline, args.output)
    if args.render_speed:
        render_speed(json.loads(args.render_speed.read_text()), args.output)
