"""Compare pinned mini_metrics accuracy by training and evaluation class support."""

import argparse
import importlib.metadata
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json

BINS = {
    "training": [(0, 1), (1, 25), (25, 100), (100, 500), (500, 2000), (2000, 10000), (10000, None)],
    "flemming": [(1, 5), (5, 20), (20, 100), (100, 500), (500, None)],
}
PRESETS = ("north_europe", "europe", "full")


def counts(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source = tomllib.loads(Path("dev/releases/mambo_v3/construction.toml").read_text())["source"]
    if file_hash(args.metadata) != source["sha256"]:
        raise ValueError("Metadata differs from pinned release source")
    table = pq.read_table(args.metadata, columns=["speciesKey", "set"])
    splits = pc.cast(table["set"], pa.int32())
    if splits.null_count or table["speciesKey"].null_count:
        raise ValueError("Missing split or species")
    if not pc.all(pc.and_(pc.greater_equal(splits, 0), pc.less_equal(splits, 9))).as_py():
        raise ValueError("Unexpected split codes")
    train = table.filter(pc.greater_equal(splits, 2))
    frequencies = {str(row["values"]): row["counts"] for row in train["speciesKey"].value_counts().to_pylist()}
    write_json(
        args.output,
        {
            "metadata_sha256": source["sha256"],
            "training_rows": train.num_rows,
            "counts": frequencies,
            "definition": "V3 source metadata rows with set 2..9; 0=test, 1=validation; no extra deduplication. "
            "Common reference axis, not a claim about V2 effective training exposures.",
        },
    )


def measure(args):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import evaluate_file

    provenance = json.loads(importlib.metadata.distribution("mini_metrics").read_text("direct_url.json"))
    if provenance["vcs_info"]["commit_id"] != REVISION:
        raise ValueError("Wrong mini_metrics revision")
    training = json.loads(args.counts.read_text())
    roots = {"v2": args.v2, "v3-torch": args.v3 / "torch-cuda-0-prediction", "v3-onnx": args.v3 / "onnx-cuda-0-prediction"}
    result = {
        "revision": REVISION,
        "policy": "threshold=0; optimal=False; simple=True; hierarchical=False",
        "training_definition": training["definition"],
        "metadata_sha256": training["metadata_sha256"],
        "training_rows": training["training_rows"],
        "bins": BINS,
        "cells": [],
        "source_sha256": {},
    }
    identity = None
    for model, root in roots.items():
        report = json.loads((root / "report.json").read_text())
        if report["status"] != "complete":
            raise ValueError("Incomplete quality run")
        for preset in PRESETS:
            source = root / preset / "mini_metric.csv"
            digest = file_hash(source)
            if digest != report["csv_sha256"][preset]:
                raise ValueError("Changed prediction CSV")
            data = MetricDF.from_source(source)
            data = data[np.asarray(data.level) == 0]
            if np.any(data.threshold != 0):
                raise ValueError("Expected unthresholded predictions")
            current = sorted(zip(data.filename.tolist(), data.label.tolist(), strict=True))
            if identity is None:
                identity = current
                support = Counter(data.label.tolist())
                result["species_support"] = {
                    label: {"flemming": n, "training": training["counts"].get(label, 0)} for label, n in sorted(support.items())
                }
            if current != identity:
                raise ValueError("Evaluation populations differ")
            result["source_sha256"][str(source)] = digest
            for axis, bins in BINS.items():
                values = np.array([result["species_support"][label][axis] for label in data.label])
                for lower, upper in bins:
                    selected = (values >= lower) & (values < upper if upper is not None else True)
                    subset = data[selected]
                    cell = {
                        "model": model,
                        "preset": preset,
                        "axis": axis,
                        "lower": lower,
                        "upper": upper,
                        "images": len(subset),
                        "species": len(set(subset.label)),
                        "known_images": int(np.asarray(subset.known_label).sum()),
                    }
                    for scope, known in (("all", False), ("known", True)):
                        cell[scope] = (
                            finite_json(
                                evaluate_file(
                                    subset,
                                    threshold=0,
                                    optimal=False,
                                    known_only=known,
                                    simple=True,
                                    hierarchical=False,
                                    pattern=r"^(accuracy|micro_accuracy)$",
                                    verbose=0,
                                )
                            )
                            if (cell["known_images"] if known else cell["images"])
                            else None
                        )
                    result["cells"].append(cell)
            print(model, preset, flush=True)
    write_json(args.output, result)


def render(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(args.data.read_text())
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "mambo-frequency-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for row, axis in enumerate(BINS):
        bins = data["bins"][axis]
        for col, preset in enumerate(PRESETS):
            ax = axes[row, col]
            for model, label, color in [("v2", "V2", "#8064a2"), ("v3-torch", "V3 PyTorch", "#098e92"), ("v3-onnx", "V3 ONNX", "#e8872e")]:
                cells = [c for c in data["cells"] if c["model"] == model and c["preset"] == preset and c["axis"] == axis]
                values = [c["all"]["accuracy"]["0"] if c["all"] else np.nan for c in cells]
                ax.plot(range(len(bins)), values, marker="o", label=label, color=color, linestyle="--" if model == "v3-onnx" else "-")
            counts = [str(c["species"]) for c in cells]
            labels = [
                ("0" if lo == 0 else f"{lo}–{hi - 1}" if hi else f"{lo}+") + "\nn=" + n for (lo, hi), n in zip(bins, counts, strict=True)
            ]
            ax.set(
                xticks=range(len(bins)),
                xticklabels=labels,
                ylim=(0, 1.05),
                title={"north_europe": "Northern Europe", "europe": "Europe", "full": "Global"}[preset],
                xlabel="V3 training metadata rows / species" if axis == "training" else "Flemming images / species",
            )
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel("Macro species accuracy")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
    fig.suptitle("Accuracy versus class frequency · all Flemming truth", fontsize=16)
    fig.text(
        0.02,
        0.015,
        "Pinned mini_metrics, no threshold optimization. n = species per bin; gaps = empty bins. "
        "All 522 truth species retained.\nTraining axis uses the common V3 source split (set 2–9), not verified V2 training exposure. "
        "Curves are descriptive; small bins are uncertain.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    args.output.mkdir(parents=True, exist_ok=True)
    svg = args.output / "mambo-frequency-accuracy.svg"
    fig.savefig(svg, metadata={"Date": None}, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(args.output / "mambo-frequency-accuracy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("counts")
    prepare.add_argument("--metadata", type=Path, required=True)
    compute = sub.add_parser("measure")
    for name in ("counts", "v2", "v3"):
        compute.add_argument("--" + name, type=Path, required=True)
    plot = sub.add_parser("render")
    plot.add_argument("--data", type=Path, required=True)
    for command in (prepare, compute, plot):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "render" and args.output.exists():
        raise FileExistsError(args.output)
    {"counts": counts, "measure": measure, "render": render}[args.command](args)
