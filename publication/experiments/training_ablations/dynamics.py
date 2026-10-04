"""Reconstruct epoch dynamics without downloading checkpoints or final logits."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import epoch_analysis, verified
from .data import digest, write_json


def analyze_dynamics(root, output):
    output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((root / "prepared.json").read_text())
    spec = json.loads(verified(root / "classes.json", prepared["files"]["classes.json"]).read_text())
    rows, learning, inputs, skipped = [], [], {}, []
    for run in json.loads((root / "plan.json").read_text()):
        attempts = sorted((root / "runs" / run["id"]).glob("attempt-*"))
        if not attempts or not (attempts[-1] / "complete.json").exists():
            skipped.append(run["id"])
            continue
        attempt = attempts[-1]
        manifest = json.loads((attempt / "complete.json").read_text())
        if json.loads(verified(attempt / "run.json", manifest["run.json"]).read_text()) != run:
            raise ValueError("Run differs from frozen plan")
        epochs, scalars, hashes = epoch_analysis(attempt, spec["counts"])
        identity = {key: run[key] for key in ["id", "variant", "seed", "epochs"]}
        rows.extend([{**identity, **row} for row in epochs.to_dict("records")])
        learning.extend([{**identity, **row} for row in scalars.to_dict("records")])
        inputs[run["id"]] = {
            "completion_manifest_sha256": digest(attempt / "complete.json"),
            "run_sha256": manifest["run.json"],
            "logs": hashes,
        }
    if not rows:
        raise ValueError("No completed epoch confusion artifacts")
    curves = pd.DataFrame(rows)
    curves.to_csv(output / "epochs.csv", index=False)
    pd.DataFrame(learning).to_csv(output / "learning.csv", index=False)
    curves[curves.epoch <= 2].to_csv(output / "early-phases.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for name, variants in [
        ("adjustment", ["full", "fixed_adjustment", "ce"]),
        ("geometry", ["full", "no_regularization", "no_normalization", "no_normalization_no_regularization"]),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for (variant, seed), values in curves[curves.variant.isin(variants)].groupby(["variant", "seed"]):
            for ax, metric in zip(axes, ["accuracy", "head_recall", "tail_recall"]):
                ax.plot(values.epoch, values[metric] * 100, label=f"{variant} / {seed}")
                ax.set(xlabel="Completed epoch", ylabel=metric + " (%)")
                ax.axvline(1, color="gray", linestyle=":")
        axes[0].legend(fontsize=6)
        fig.suptitle("AMP epoch confusion metrics; end of warmup marked at epoch 1")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
    write_json(
        output / "provenance.json",
        {
            "prepared_sha256": digest(root / "prepared.json"),
            "plan_sha256": digest(root / "plan.json"),
            "source_sha256": digest(Path(__file__)),
            "analysis_sha256": digest(Path(__file__).with_name("analysis.py")),
            "skipped_incomplete": skipped,
            "inputs": inputs,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze_dynamics(args.root, args.output)


if __name__ == "__main__":
    main()
