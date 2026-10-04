"""Render per-run mechanism diagnostics from analysis tables, without model loads."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import GROUPS
from .data import digest, write_json


def save_figure(fig, path):
    """Keep a lightweight preview and a vector version for publication."""
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=160)


def plot_comparisons(root, output, reports, sources):
    """Show every seed and treatment; never pool different study roots/budgets."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    any_epochs = False
    for report in reports:
        run = report["run"]
        name = run.get("id", f"{run['variant']}_seed{run['seed']}")
        path = root / name / "epochs.csv"
        if path.exists() and run["variant"] in ("full", "fixed_adjustment", "ce"):
            sources[str(path.relative_to(root))] = digest(path)
            rows = pd.read_csv(path)
            for ax, metric in zip(axes, ("accuracy", "head_recall", "tail_recall")):
                ax.plot(rows.epoch, rows[metric] * 100, label=f"{run['variant']} / {run['seed']}")
                ax.set(xlabel="Completed epoch", ylabel=metric + " (%)")
                ax.axvline(1, color="gray", linestyle=":")
            any_epochs = True
    if any_epochs:
        axes[0].legend(fontsize=7)
        fig.suptitle("AMP epoch trajectories; dotted line: end of head-only warmup")
        fig.tight_layout()
        save_figure(fig, output / "learning-dynamics")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, report in enumerate(reports):
        run, metrics = report["run"], report["metrics"]
        name = run.get("id", f"{run['variant']}_seed{run['seed']}")
        label = f"{run['variant']} / {run['seed']}"
        flows = pd.read_csv(root / name / "confusion_flows.csv")
        tail_error = flows[(flows.true_group == "tail") & (flows.predicted_group == "tail")].error_probability.iloc[0] * 100
        if "geometry" in metrics:
            axes[0].scatter(metrics["geometry"]["effective_rank"], tail_error, label=label)
        else:
            axes[0].scatter(i, tail_error, label=label)
        for j, rank in enumerate(("genus", "family")):
            result = metrics.get("parents", {}).get(rank, {})
            if "balanced" in result:
                axes[1].scatter(i + j * 0.2, result["balanced"]["accuracy"] * 100, marker=("o", "x")[j])
    axes[0].set(xlabel="Prototype effective rank (run index if unavailable)", ylabel="Rare-to-rare error probability (%)")
    axes[0].legend(fontsize=6)
    axes[1].set(
        ylabel="Equal-parent recall (%)",
        title="Genus: circles; family: crosses",
        xticks=range(len(reports)),
        xticklabels=[f"{r['run']['variant']} / {r['run']['seed']}" for r in reports],
    )
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    save_figure(fig, output / "geometry-and-hierarchy")
    plt.close(fig)


def plot_report(root, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = json.loads((root / "report.json").read_text())["runs"]
    sources = {"report.json": digest(root / "report.json")}
    for report in reports:
        run = report["run"]
        name = run.get("id", f"{run['variant']}_seed{run['seed']}")
        tables = {}
        for key in ["classes", "confusion_flows", "reliability"]:
            path = root / name / f"{key}.csv"
            tables[key] = pd.read_csv(path)
            sources[str(path.relative_to(root))] = digest(path)
        classes, flows, reliability = (tables[k] for k in ["classes", "confusion_flows", "reliability"])
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        x = np.log10(classes.train_count)
        axes[0, 0].scatter(x, classes.recall * 100, s=12, alpha=0.35)
        for label, group in classes.groupby("frequency_group", sort=False):
            axes[0, 0].scatter(np.log10(group.train_count).mean(), group.recall.mean() * 100, s=65, label=label)
        axes[0, 0].set(xlabel="log10 training count", ylabel="Class recall (%)", title="Performance-frequency relationship")
        axes[0, 0].legend()
        axes[0, 1].scatter(x, classes.predicted_mass_balanced * len(classes), s=12, alpha=0.35)
        axes[0, 1].axhline(1, color="gray", linestyle="--")
        axes[0, 1].set(
            xlabel="log10 training count",
            ylabel="Balanced predicted mass × class count",
            title="Prediction preference under equal class weights",
        )
        matrix = flows.pivot(index="true_group", columns="predicted_group", values="error_probability").reindex(
            index=GROUPS, columns=GROUPS
        )
        im = axes[1, 0].imshow(matrix * 100, vmin=0, cmap="magma")
        fig.colorbar(im, ax=axes[1, 0], label="Error probability (%)")
        for i in range(3):
            for j in range(3):
                axes[1, 0].text(j, i, f"{matrix.iloc[i, j] * 100:.2f}%", ha="center", va="center", color="cyan")
        axes[1, 0].set(
            xticks=range(3),
            xticklabels=GROUPS,
            yticks=range(3),
            yticklabels=GROUPS,
            xlabel="Predicted frequency group",
            ylabel="True frequency group",
            title="Error flows per true class; correct predictions excluded",
        )
        for prior, values in reliability.groupby("prior", sort=False):
            values = values[values.mass > 0]
            (line,) = axes[1, 1].plot(values.confidence, values.accuracy, label=prior)
            axes[1, 1].scatter(values.confidence, values.accuracy, s=12 + 180 * values.mass, color=line.get_color())
        axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1, 1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Reliability: empty bins omitted")
        axes[1, 1].legend()
        fig.suptitle(f"{run['variant']} | seed {run['seed']} | {run['epochs']} epochs | {report['split']}")
        fig.tight_layout()
        save_figure(fig, output / name)
        plt.close(fig)
    plot_comparisons(root, output, reports, sources)
    plot_gates(root, output, reports, sources)
    write_json(
        output / "provenance.json",
        {"input_root": str(root), "source_sha256": digest(Path(__file__)), "matplotlib": matplotlib.__version__, "inputs": sources},
    )


def plot_gates(root, output, reports, sources):
    """Training gates use zero-based log epochs; show completed epochs consistently."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    found = False
    for report in reports:
        run = report["run"]
        name = run.get("id", f"{run['variant']}_seed{run['seed']}")
        path = root / name / "learning.csv"
        if not path.exists() or path.stat().st_size <= 1:
            continue
        rows = pd.read_csv(path)
        if "phase" not in rows:
            continue
        rows = rows[rows.phase == "train"]
        sources[str(path.relative_to(root))] = digest(path)
        for ax, group in zip(axes, GROUPS):
            column = f"gate/species/{group}/mean"
            if column not in rows or not rows[column].notna().any():
                continue
            ax.plot(rows.epoch + 1, rows[column], label=f"{run['variant']} / {run['seed']}")
            found = True
    if found:
        for ax, group in zip(axes, GROUPS):
            ax.set(xlabel="Completed epoch", ylabel="Mean training gate", title=group, ylim=(0, 1))
            ax.axvline(1, color="gray", linestyle=":")
        axes[0].legend(fontsize=6)
        fig.suptitle("EMLA confidence gates; epoch aggregates, not an overfitting detector")
        fig.tight_layout()
        save_figure(fig, output / "adjustment-gates")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_report(args.report, args.output)


if __name__ == "__main__":
    main()
