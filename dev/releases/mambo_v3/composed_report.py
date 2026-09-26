"""Exploratory composed-TTA figure from retained metric evidence."""

import argparse
import json
from pathlib import Path

from .figure_export import save_figure

SERIES = (
    ("v2", "MAMBO v2", "#8064a2"),
    ("torch", "V3 single view", "#777777"),
    ("torch-tta", "Previous padded scale", "#098e92"),
    ("torch:rotation30_pad15_3", "±30° / pad15 · 3 views", "#2d6cc0"),
    ("torch:rotation30_pad25_3", "±30° / pad25 · 3 views", "#e8872e"),
    ("torch:wide_rotation_mixed_padding_5", "Mixed padding · 5 views", "#98440b"),
)


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "composed-tta-full-v1"})
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for level, rank in enumerate(("species", "genus", "family")):
        k = str(level)
        for i, (model, label, color) in enumerate(SERIES):
            points = data["models"][model]["operating_points"]
            for col, scope in enumerate(("zero", "optimized")):
                point = points[scope]
                a, b = point["full"]["f1"][k], point["tail"][k]["metrics"]["f1"]
                axes[level, col].plot([a, b], [i, i], color=color, alpha=0.5)
                axes[level, col].scatter(a, i, facecolors="white", edgecolors=color, s=45)
                axes[level, col].scatter(b, i, color=color, s=45)
            cov = [points[f"coverage_{c}"]["full"]["coverage"][k] * 100 for c in (70, 80, 90)]
            score = [points[f"coverage_{c}"]["full"]["f1"][k] for c in (70, 80, 90)]
            axes[level, 2].plot(cov, score, marker="o", color=color, label=label)
        for col, setting in enumerate(("Unthresholded", "Calibrated")):
            axes[level, col].set(
                title=f"{rank.title()} · {setting} Macro-F1",
                xlim=(0, 1),
                ylim=(len(SERIES) - 0.4, -0.6),
                yticks=range(len(SERIES)),
                yticklabels=[s[1] for s in SERIES] if col == 0 else [],
            )
        axes[level, 2].set(title=f"{rank.title()} · Matched-coverage full F1", xlabel="Realized coverage (%)", ylabel="Macro-F1")
        for ax in axes[level]:
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Composed TTA · shared 52,788-image reporting partition · native PyTorch", fontsize=16)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color="gray", markerfacecolor="white", linestyle="none", label="Full support"),
            Line2D([], [], marker="o", color="gray", linestyle="none", label="Support >5 in truth and accepted predictions"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        title="Fill = averaging domain (left and middle columns)",
    )
    fig.text(
        0.03,
        0.02,
        "Calibration: 5,852 separate images, per-recipe/rank mini_metrics Macro-F1 thresholds. All truth; legacy northern Europe.\n"
        "Matched-coverage points use label-free reporting confidences; actual coverage includes score ties. "
        "Lines are guides, not fitted curves.\n"
        "Support >5 classes are common across all compared native/ONNX pipelines within each operating point; no evaluation rows removed.\n"
        "Recipe exploration used this dataset. These are descriptive comparisons, "
        "not independent validation; CSV includes all metrics/backends.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.91))
    save_figure(fig, output, "mambo-composed-tta", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    render(data, args.output)
