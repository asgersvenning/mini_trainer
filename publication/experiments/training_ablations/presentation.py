"""Assemble one frozen study's verified analyses into paired presentation artifacts."""

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from .analysis_plots import plot_report
from .data import digest, write_json
from .study import factorial_contrasts

# Positive differences always mean treatment minus comparator, not improvement.
CONTRASTS = (
    ("normalization, R on", "full", "no_normalization"),
    ("normalization, R off", "no_regularization", "no_normalization_no_regularization"),
    ("regularization, N on", "full", "no_regularization"),
    ("regularization, N off", "no_normalization", "no_normalization_no_regularization"),
    ("adaptive adjustment vs CE", "full", "ce"),
    ("adaptive vs fixed adjustment", "full", "fixed_adjustment"),
    ("fixed adjustment vs CE", "fixed_adjustment", "ce"),
    ("hierarchy, R on", "hierarchy_regularized", "species_regularized"),
    ("hierarchy, R off", "hierarchy_unregularized", "species_unregularized"),
    ("hierarchy, R on", "hierarchy_regularized", "full"),
    ("hierarchy, R off", "hierarchy_unregularized", "no_regularization"),
    ("regularization, H on", "hierarchy_regularized", "hierarchy_unregularized"),
    ("regularization, H off", "species_regularized", "species_unregularized"),
)
METRICS = (
    "empirical.accuracy",
    "balanced.accuracy",
    "tail_recall",
    "head_recall",
    "empirical.nll",
    "balanced.nll",
    "empirical.brier",
    "balanced.brier",
    "empirical.ece",
    "balanced.ece",
    "frequency_recall.spearman",
    "frequency_recall.slope",
    "frequency_balanced_soft_mass.spearman",
    "tail_to_tail_error",
    "tail_to_mid_error",
    "tail_to_head_error",
    "parents.genus.balanced.accuracy",
    "parents.family.balanced.accuracy",
    "geometry.effective_rank",
)


def flatten(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (int, float)) or value is None:
            result[name] = value
    return result


def paired_contrasts(plan, endpoints):
    lookup = {(row["variant"], row["seed"]): row for row in endpoints}
    planned = {(row["variant"], row["seed"]) for row in plan}
    rows, missing = [], []
    for label, treatment, comparator in CONTRASTS:
        for seed in sorted({row["seed"] for row in plan}):
            keys = [(name, seed) for name in (treatment, comparator)]
            if not all(key in planned for key in keys):
                continue
            if not all(key in lookup for key in keys):
                missing.append({"contrast": label, "seed": seed, "treatment": treatment, "comparator": comparator})
                continue
            left, right = (lookup[key] for key in keys)
            for metric in METRICS:
                a, b = left.get(metric), right.get(metric)
                if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
                    rows.append(
                        {
                            "contrast": label,
                            "seed": seed,
                            "treatment": treatment,
                            "comparator": comparator,
                            "metric": metric,
                            "treatment_value": a,
                            "comparator_value": b,
                            "difference": a - b,
                        }
                    )
    return rows, missing


def assemble(study, reports, output, require_complete=False, plots=True):
    study, output = Path(study), Path(output)
    plan = json.loads((study / "plan.json").read_text())
    expected = {run["id"]: run for run in plan}
    if len(expected) != len(plan) or len({(r["variant"], r["seed"]) for r in plan}) != len(plan):
        raise ValueError("Duplicate run identity in plan")
    if len({r["epochs"] for r in plan}) != 1:
        raise ValueError("Keep different training budgets in separate presentations")
    prepared_hash = digest(study / "prepared.json")
    sources, runs, locations, versions = {}, {}, {}, set()
    source_reports = []
    for root in map(Path, reports):
        provenance = json.loads((root / "provenance.json").read_text())
        if provenance["prepared_sha256"] != prepared_hash:
            raise ValueError("Report belongs to a different prepared study")
        versions.add((provenance["analysis_sha256"], provenance["contrast_code_sha256"]))
        source_reports.append(provenance)
        for path in (root / "report.json", root / "provenance.json"):
            sources[str(path.resolve())] = digest(path)
        for entry in json.loads((root / "report.json").read_text())["runs"]:
            run = entry["run"]
            name = run.get("id", f"{run['variant']}_seed{run['seed']}")
            if name in runs:
                raise ValueError(f"Duplicate report for {name}; supply disjoint reports")
            if run != expected.get(name):
                raise ValueError(f"Run differs from frozen plan: {name}")
            runs[name], locations[name] = entry, root / name
    if len(versions) > 1:
        raise ValueError("Analysis implementations differ; regenerate comparable reports")
    if len({r["split"] for r in runs.values()}) > 1:
        raise ValueError("Cannot mix evaluation splits")
    missing_runs = sorted(set(expected) - set(runs))
    if require_complete and missing_runs:
        raise ValueError(f"Missing {len(missing_runs)} planned runs: {missing_runs}")
    if not runs:
        raise ValueError("No analyzed runs")
    endpoints = []
    for name, entry in runs.items():
        values = flatten(entry["metrics"])
        classes = pd.read_csv(locations[name] / "classes.csv")
        flows = pd.read_csv(locations[name] / "confusion_flows.csv")
        for group in ("tail", "head"):
            values[f"{group}_recall"] = classes.loc[classes.frequency_group == group, "recall"].mean()
        for group in ("tail", "mid", "head"):
            cell = flows[(flows.true_group == "tail") & (flows.predicted_group == group)]
            values[f"tail_to_{group}_error"] = cell.error_probability.iloc[0]
        endpoints.append({**entry["run"], "split": entry["split"], **{key: values.get(key) for key in METRICS}})
    pairs, missing_pairs = paired_contrasts(plan, endpoints)
    interactions = factorial_contrasts(endpoints, metrics=METRICS)
    output.mkdir(parents=True, exist_ok=False)
    # Retain the small source tables so plots can be reproduced without training/logits.
    combined = output / "analysis"
    combined.mkdir()
    for name, location in locations.items():
        destination = combined / name
        destination.mkdir()
        for path in sorted(location.glob("*.csv")):
            sources[str(path.resolve())] = digest(path)
            shutil.copyfile(path, destination / path.name)
    write_json(combined / "report.json", {"runs": list(runs.values()), "skipped_incomplete": missing_runs})
    write_json(combined / "provenance.json", {"source_reports": source_reports})
    pd.DataFrame(endpoints).to_csv(output / "endpoints.csv", index=False)
    frame = pd.DataFrame(
        pairs, columns=["contrast", "seed", "treatment", "comparator", "metric", "treatment_value", "comparator_value", "difference"]
    )
    frame.to_csv(output / "paired-contrasts.csv", index=False)
    if not frame.empty:
        frame.groupby(["contrast", "treatment", "comparator", "metric"]).difference.agg(["count", "mean", "min", "max"]).to_csv(
            output / "replicates.csv"
        )
    write_json(output / "interactions.json", interactions)
    write_json(
        output / "coverage.json",
        {
            "complete": not missing_runs,
            "analyzed": len(runs),
            "planned": len(plan),
            "missing_runs": missing_runs,
            "missing_pairs": missing_pairs,
        },
    )
    shutil.copyfile(study / "plan.json", output / "plan.json")
    if plots:
        plot_report(combined, output / "diagnostics")
        plot_pairs(frame, output)
    write_json(
        output / "provenance.json",
        {
            "prepared_sha256": prepared_hash,
            "plan_sha256": digest(study / "plan.json"),
            "presentation_sha256": digest(Path(__file__)),
            "inputs": sources,
        },
    )
    (output / "README.txt").write_text(
        f"Analyzed {len(runs)}/{len(plan)} planned runs. "
        + ("COMPLETE.\n" if not missing_runs else "PRELIMINARY.\n")
        + "Contrasts are treatment minus comparator within seed, cohort, split and training budget.\n"
        "Rates and rate differences use fractions; multiply by 100 for percent/percentage points.\n"
        "Replicate ranges describe observed seeds, not confidence intervals. Missing pairs are not null effects.\n"
        "Equal-class ECE pools confidence bins with equal total weight per true class.\n"
        "Epoch confusion curves use AMP; final prediction metrics use FP32 reload.\n"
        "A longer cosine schedule is a separate experiment, not continuation of the shorter schedule.\n"
        "Normalization contrasts test the implemented head package; hierarchy contrasts include its objective.\n"
        "No embedding-space analyses are included.\n"
    )


def plot_pairs(frame, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, max(3, 0.45 * frame.contrast.nunique() + 1)), sharey=True)
    for ax, metric in zip(axes, ("balanced.accuracy", "tail_recall", "balanced.nll", "balanced.ece")):
        selected = frame[frame.metric == metric]
        labels = list(dict.fromkeys(selected.contrast))
        for seed, group in selected.groupby("seed"):
            scale = 1 if metric.endswith("nll") else 100
            ax.scatter(group.difference * scale, [labels.index(label) for label in group.contrast], label=f"seed {seed}")
        ax.axvline(0, color="gray", linestyle=":")
        ax.set(
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Difference (NLL)" if metric.endswith("nll") else "Difference (percentage points)",
            title=metric,
        )
        ax.tick_params(axis="y", labelsize=7)
    if not frame.empty:
        axes[0].legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"paired-contrasts.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    assemble(args.study, args.reports, args.output, args.require_complete)


if __name__ == "__main__":
    main()
