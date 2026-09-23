"""Compact comparison of bounded TTA quality and its measured GPU inference cost."""

import argparse
import json
from pathlib import Path

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import write_json

ORDER = (
    "none",
    "hflip",
    "d4",
    "padded_scale",
    "padded_rotation",
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "light_noise",
    "mild_blur",
    "five_crop",
    "ten_crop",
)
LABELS = (
    "None",
    "Horizontal flip",
    "D4 rotations/reflections",
    "Padded scale",
    "Padded ±10° rotation",
    "Brightness",
    "Contrast",
    "Gamma",
    "Gaussian noise",
    "Salt-and-pepper noise",
    "Mild blur",
    "Five crops",
    "Ten crops",
)


def collect(quality, root):
    metrics = json.loads(quality.read_text())
    data = {
        "quality": [
            {"backend": row["backend"], "profile": row["profile"], "macro_accuracy": row["presets"]["north_europe"]["all"]["accuracy"]["0"]}
            for row in metrics["variants"]
        ],
        "timing": [],
        "sources_sha256": {str(quality): file_hash(quality)},
    }
    samples = None
    for backend in ("torch", "onnx"):
        path = root / f"mambo-tta-timing-{backend}" / "report.json"
        report = json.loads(path.read_text())
        if report["status"] != "complete" or {c["profile"] for c in report["cells"]} != set(ORDER):
            raise ValueError("Incomplete TTA timing matrix")
        if samples is not None and samples != report["samples"]:
            raise ValueError("Different timing samples")
        samples = report["samples"]
        data["sources_sha256"][str(path)] = file_hash(path)
        for cell in report["cells"]:
            data["timing"].append({"backend": backend, **cell})
    return data


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-tta-v1", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    y = np.arange(len(ORDER))
    for i, (backend, label, color) in enumerate((("torch", "PyTorch", "#098e92"), ("onnx", "ONNX", "#e8872e"))):
        quality = [
            next(v["macro_accuracy"] for v in data["quality"] if v["backend"] == backend and v["profile"] == name) * 100 for name in ORDER
        ]
        speed = [next(v["images_per_second"] for v in data["timing"] if v["backend"] == backend and v["profile"] == name) for name in ORDER]
        axes[0].scatter(quality, y + (i - 0.5) * 0.23, label=label, color=color, s=40)
        axes[1].barh(y + (i - 0.5) * 0.3, speed, height=0.3, color=color, label=label)
    axes[0].set(
        yticks=y, yticklabels=LABELS, xlabel="Macro species accuracy (%)", title="Fixed 1,024-image qualification subset", xlim=(65, 83)
    )
    axes[0].invert_yaxis()
    axes[1].set(xlabel="End-to-end images / second", title="GPU · batch 32 · four preparation workers")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.57, 0.945), ncol=2, frameon=False)
    fig.suptitle("TTA candidates: quality and inference cost · northern Europe", fontsize=15)
    fig.text(
        0.02,
        0.02,
        "Quality: pinned mini_metrics, all truth, threshold 0; 201 represented species. Exploratory subset, not full-set efficacy.\n"
        "Speed: RTX 3080 Ti Laptop; one process per backend, one warmup + three observations per policy; decode and CPU results included.\n"
        "Dots use a restricted accuracy axis for readability. All policies are opt-in; none changes the release default.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.90))
    svg = output / "mambo-tta-tradeoffs.svg"
    fig.savefig(svg, metadata={"Date": None}, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output / "mambo-tta-tradeoffs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "mambo-tta-tradeoffs.json", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.data and (not args.quality or not args.root):
        parser.error("Supply --data or both --quality and --root")
    render(json.loads(args.data.read_text()) if args.data else collect(args.quality, args.root), args.output)
