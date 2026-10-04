"""Render per-run mechanism diagnostics from analysis tables, without model loads."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import GROUPS
from .data import digest, write_json


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
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
    write_json(
        output / "provenance.json",
        {"input_root": str(root), "source_sha256": digest(Path(__file__)), "matplotlib": matplotlib.__version__, "inputs": sources},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_report(args.report, args.output)


if __name__ == "__main__":
    main()
