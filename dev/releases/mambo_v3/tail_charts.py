"""Plot full and truncated macro metrics, optionally comparing confidence settings."""

import argparse
import json
from pathlib import Path

from dev.releases.mambo_v3.defaults_report import SERIES


def render(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if data.get("population") != "full_dataset":
        raise ValueError("Expected full-dataset results")
    rows = {(r["model"], r["rank"], r["cutoff"], r["domain"]): r for r in data["rows"] if r["scope"] == "zero"}
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mambo-tail-v1", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for level, rank in enumerate(("species", "genus", "family")):
        retained = rows["v2", rank, 5, "common"]
        total = retained["report_images"]
        excluded = total - retained["truth_images_in_retained_classes"]
        for col, (metric, title, factor) in enumerate((("accuracy", "Macro accuracy (%)", 100), ("f1", "Macro-F1", 1))):
            ax = axes[level, col]
            for i, (model, _, color) in enumerate(SERIES):
                baseline = rows[model, rank, -1, "per_model"]["metrics"][metric] * factor
                truncated = rows[model, rank, 5, "common"]["metrics"][metric] * factor
                ax.plot([baseline, truncated], [i, i], color=color, alpha=0.5)
                ax.scatter(baseline, i, facecolors="white", edgecolors=color, s=65, zorder=3)
                ax.scatter(truncated, i, color=color, s=65, zorder=3)
            ax.set(
                title=(
                    f"{rank.title()} · {title}\n>5: {retained['class_count']} classes; "
                    f"{excluded:,} truth images outside ({excluded / total:.2%})"
                ),
                yticks=range(5),
                yticklabels=[label for _, label, _ in SERIES] if col == 0 else [],
                xlim=(0, factor),
                ylim=(4.6, -0.6),
            )
            ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Full support and tail-truncated metrics · legacy northern Europe", fontsize=15)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", color="gray", markerfacecolor="white", linestyle="none", label="Full support (>−1; per-model classes)"
            ),
            Line2D([], [], marker="o", color="gray", linestyle="none", label="Support >5 in truth AND predictions (common classes)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
    )
    fig.text(
        0.03,
        0.02,
        "58,640 Flemming images; confidence threshold 0; pinned mini_metrics. TTA: padded scale.\n"
        "No evaluation rows removed: FP/FN remain; only the macro averaging domain changes.\n"
        "Outside counts refer to truth classes, not confidence rejection. Truncation excludes predicted-only classes.\n"
        "Descriptive comparison; TTA was selected on a subset of the same dataset.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "mambo-defaults-tail.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    fig.savefig(output / "mambo-defaults-tail.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_paired(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = {(r["model"], r["scope"], r["rank"], r["cutoff"], r["domain"]): r for r in data["rows"]}
    if {r["report_images"] for r in data["rows"]} != {52788} or {r["scope"] for r in data["rows"]} != {"zero", "optimized"}:
        raise ValueError("Expected both confidence settings on the 52,788-image reporting partition")
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
    fig.suptitle("V2 vs V3 vs V3 + TTA · matched reporting images · legacy northern Europe", fontsize=16)
    fig.legend(handles=shape_handles, title="Shape = confidence setting", loc="upper center", bbox_to_anchor=(0.30, 0.955), ncol=2)
    fig.legend(handles=fill_handles, title="Fill = averaging domain", loc="upper center", bbox_to_anchor=(0.70, 0.955), ncol=2)
    fig.text(
        0.03,
        0.02,
        "Same 52,788 reporting images throughout; thresholds fitted on 5,852 separate images using mini_metrics Macro-F1.\n"
        "Hollow → filled changes the averaging domain, not predictions. "
        ">5 requires truth AND accepted-prediction support in every pipeline.\n"
        "Retained classes differ between confidence settings. No evaluation rows removed; per-class FP/FN remain intact.\n"
        f"Coverage is unchanged by class truncation. TTA: {data.get('tta', 'padded_scale')}; "
        "recipe selection used the same dataset, so results remain descriptive.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "mambo-threshold-tail.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    fig.savefig(output / "mambo-threshold-tail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired", action="store_true", help="Compare both confidence settings on the reporting partition")
    args = parser.parse_args()
    (render_paired if args.paired else render)(json.loads(args.data.read_text()), args.output)
