"""Compare full and truncated macro metrics at both confidence settings."""

import argparse
import json
from pathlib import Path

from dev.releases.mambo_v3.defaults_report import SERIES

from .figure_export import save_figure


def render_paired(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = {(r["model"], r["scope"], r["rank"], r["cutoff"], r["domain"]): r for r in data["rows"]}
    if {r["report_images"] for r in data["rows"]} != {data.get("report_images", 52788)} or {r["scope"] for r in data["rows"]} != {
        "zero",
        "optimized",
    }:
        raise ValueError("Expected both confidence settings on the shared reporting partition")
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-paired-tail-v1"})
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    for level, rank in enumerate(("species", "genus", "family")):
        for col, metric in enumerate(("accuracy", "f1", "coverage")):
            ax = axes[level, col]
            for i, (model, _, color) in enumerate(SERIES):
                for scope, marker, offset in (("zero", "o", -0.16), ("optimized", "s", 0.16)):
                    full = rows[model, scope, rank, -1, "per_model"]
                    tail = rows[model, scope, rank, 5, "common"]
                    y = i + offset
                    if metric == "coverage":
                        value = full["overall_coverage"] * 100
                        ax.scatter(value, y, color=color, marker=marker, s=35)
                        ax.annotate(f"{value:.1f}%", (value, y), xytext=(-7, -3), textcoords="offset points", ha="right", fontsize=8)
                    else:
                        factor = 100 if metric == "accuracy" else 1
                        a, b = full["metrics"][metric] * factor, tail["metrics"][metric] * factor
                        ax.plot([a, b], [y, y], color=color, alpha=0.4)
                        ax.scatter(a, y, facecolors="white", edgecolors=color, marker=marker, s=45, zorder=3)
                        ax.scatter(b, y, color=color, marker=marker, s=45, zorder=3)
            title = {"accuracy": "Macro accuracy (%)", "f1": "Macro-F1", "coverage": "Acceptance coverage (%)"}[metric]
            ax.set(
                title=f"{rank.title()} · {title}",
                yticks=range(5),
                yticklabels=[label for _, label, _ in SERIES] if col == 0 else [],
                xlim=(0, 1) if metric == "f1" else (0, 105),
                ylim=(4.6, -0.6),
            )
            ax.grid(axis="x", alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
    shape_handles = [
        Line2D([], [], marker=marker, color="gray", markerfacecolor="white", linestyle="none", label=label)
        for marker, label in (("o", "Unthresholded"), ("s", "Calibrated"))
    ]
    fill_handles = [
        Line2D([], [], marker="o", color="gray", markerfacecolor=fill, linestyle="none", label=label)
        for fill, label in (("white", "Full support"), ("gray", "Truncated (support >5)"))
    ]
    fig.suptitle(data.get("dataset_title", "V2 vs V3 vs V3 + TTA · matched reporting images · legacy northern Europe"), fontsize=16)
    fig.legend(handles=shape_handles, title="Shape = confidence setting", loc="upper center", bbox_to_anchor=(0.30, 0.955), ncol=2)
    fig.legend(handles=fill_handles, title="Fill = averaging domain", loc="upper center", bbox_to_anchor=(0.70, 0.955), ncol=2)
    fig.text(
        0.03,
        0.02,
        f"Same {data.get('report_images', 52788):,} reporting images throughout; thresholds fitted on "
        f"{data.get('calibration_images', 5852):,} separate images using mini_metrics Macro-F1.\n"
        "Hollow → filled changes the averaging domain, not predictions. "
        ">5 requires truth AND accepted-prediction support in every pipeline.\n"
        "Retained classes differ between confidence settings. No evaluation rows removed; per-class FP/FN remain intact.\n"
        f"Coverage is unchanged by class truncation. TTA: {data.get('tta', 'padded_scale')}; "
        + (
            "recipe selected on Flemming, not this test set."
            if data.get("independent_recipe")
            else "recipe selection used the same dataset, so results remain descriptive."
        ),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    output.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output, "mambo-threshold-tail", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_paired(json.loads(args.data.read_text()), args.output)
