"""Render worker scaling and experimental lookahead with matching throughput units."""

import argparse
import json
import statistics
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json


def aggregate(root):
    result = {"cells": [], "streams": [], "sources_sha256": {}}
    for backend in ("torch", "onnx"):
        path = root / f"mambo-loading-scaling-{backend}" / "report.json"
        data = json.loads(path.read_text())
        if data["status"] != "complete":
            raise ValueError("Incomplete diagnostic")
        result["sources_sha256"][str(path)] = file_hash(path)
        for batch in (8, 32, 64):
            for workers in (1, 2, 4, 8):
                cells = [c for c in data["cells"] if c["batch"] == batch and c["workers"] == workers]
                if len(cells) != 3:
                    raise ValueError("Missing trials")
                cell = {"backend": backend, "batch": batch, "workers": workers}
                for boundary in ("preparation", "prepared", "end_to_end"):
                    cell[boundary] = batch / statistics.median(v for c in cells for v in c[boundary]["seconds"])
                result["cells"].append(cell)
        result["streams"].extend(
            {"backend": backend, "workers": x["workers"], "overlap": x["overlap"], "images_per_second": x["images"] / x["median_seconds"]}
            for x in data["streams"]
        )
    return result


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-loading-v1", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, backend in zip(axes[:2], ("torch", "onnx"), strict=True):
        for batch, color in zip((8, 32, 64), ("#8064a2", "#098e92", "#e8872e"), strict=True):
            cells = [c for c in data["cells"] if c["backend"] == backend and c["batch"] == batch]
            ax.plot([c["workers"] for c in cells], [c["end_to_end"] for c in cells], marker="o", label=f"Batch {batch}", color=color)
        ax.set(
            title=f"{backend.title()} · synchronous API",
            xlabel="Preparation workers",
            xticks=[1, 2, 4, 8],
            ylabel="End-to-end images / second",
            ylim=(0, 175),
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    ax = axes[2]
    for i, overlap in enumerate((False, True)):
        values = [
            next(c["images_per_second"] for c in data["streams"] if c["backend"] == b and c["workers"] == 4 and c["overlap"] == overlap)
            for b in ("torch", "onnx")
        ]
        bars = ax.bar(
            np.arange(2) + (i - 0.5) * 0.35,
            values,
            0.35,
            label="One-batch lookahead" if overlap else "Sequential",
            color="#098e92" if overlap else "#888888",
        )
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(
        title="Experimental stream · batch 32",
        xticks=[0, 1],
        xticklabels=["PyTorch", "ONNX"],
        ylabel="End-to-end images / second",
        ylim=(0, 220),
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.suptitle("Loading and scheduling still limit GPU throughput", fontsize=16)
    fig.text(
        0.015,
        0.015,
        "RTX 3080 Ti Laptop; automatic precision; northern Europe; runtime CPU threads fixed at 4.\n"
        "Worker sweep: 3 ordered trials × 3 observations. Lookahead: 128 images, 4 workers, 5 repeats in one process; "
        "experimental, not the API default.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.91))
    svg = output / "mambo-loading-scaling.svg"
    fig.savefig(svg, metadata={"Date": None}, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output / "mambo-loading-scaling.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "mambo-loading-scaling.json", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.root and not args.data:
        parser.error("Supply --root or --data")
    render(json.loads(args.data.read_text()) if args.data else aggregate(args.root), args.output)
