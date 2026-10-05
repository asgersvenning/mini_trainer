"""Pair lower-unique-support results with their full-support controls."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import verified
from .data import digest, write_json

VARIANTS = ("species_regularized", "hierarchy_regularized")
RUN_FIELDS = ("normalized", "hidden", "regularization", "loss", "optimizer", "rank_weights", "lr", "weight_decay", "epochs", "screening")


def validate_evaluation_rows(full, limited, split):
    """Require identical labels, taxonomy and sample ordering in the held-out split."""
    columns = ["sample_id", "label", "speciesKey", "genusKey", "familyKey"]
    left = full.loc[full.split == split, columns].reset_index(drop=True)
    right = limited.loc[limited.split == split, columns].reset_index(drop=True)
    if left.empty or not left.equals(right):
        raise ValueError(f"The {split} samples differ between support cohorts")
    return len(left)


def flattened(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flattened(value, name))
        elif isinstance(value, (int, float)) or value is None:
            result[name] = value
    return result


def load_cohort(study, analysis):
    study, analysis = Path(study), Path(analysis)
    prepared = json.loads((study / "prepared.json").read_text())
    for filename in ("classes.json", "samples.parquet"):
        verified(study / filename, prepared["files"][filename])
    classes = json.loads((study / "classes.json").read_text())
    samples = pd.read_parquet(study / "samples.parquet")
    plan = [run for run in json.loads((study / "plan.json").read_text()) if run.get("variant") in VARIANTS]
    seeds = {run["seed"] for run in plan}
    pairs = {(run["variant"], run["seed"]) for run in plan}
    expected_pairs = {(variant, seed) for variant in VARIANTS for seed in seeds}
    if not seeds or pairs != expected_pairs:
        raise ValueError("Expected only the paired regularized species and hierarchy treatments")
    provenance = json.loads((analysis / "provenance.json").read_text())
    if provenance["prepared_sha256"] != digest(study / "prepared.json"):
        raise ValueError("Analysis report belongs to a different prepared cohort")
    reports = {entry["run"]["id"]: entry for entry in json.loads((analysis / "report.json").read_text())["runs"]}
    expected = {run["id"] for run in plan}
    if not expected <= set(reports):
        raise ValueError("Analysis report is missing one or more selected paired runs")
    endpoint_rows = {}
    for run in plan:
        attempts = sorted((study / "runs" / run["id"]).glob("attempt-*"))
        if not attempts:
            raise ValueError(f"No retained run attempt: {run['id']}")
        attempt = attempts[-1]
        manifest = json.loads((attempt / "complete.json").read_text())
        for filename in ("run.json", "evaluation.json", "predictions.npz"):
            verified(attempt / filename, manifest[filename])
        stored_run = json.loads((attempt / "run.json").read_text())
        if stored_run != run or reports[run["id"]]["run"] != run:
            raise ValueError(f"Run identity differs from frozen plan: {run['id']}")
        evaluation = json.loads((attempt / "evaluation.json").read_text())
        if evaluation["split"] != reports[run["id"]]["split"]:
            raise ValueError(f"Evaluation split differs from analysis: {run['id']}")
        with np.load(attempt / "predictions.npz", allow_pickle=False) as predictions:
            selected = samples.loc[samples.split == evaluation["split"]]
            if not np.array_equal(predictions["sample_id"], selected.sample_id.to_numpy()) or not np.array_equal(
                predictions["target"], selected.label.to_numpy()
            ):
                raise ValueError(f"Prediction alignment differs from prepared split: {run['id']}")
        report_root = analysis / run["id"]
        metrics = flattened(reports[run["id"]]["metrics"])
        class_table = pd.read_csv(report_root / "classes.csv")
        flows = pd.read_csv(report_root / "confusion_flows.csv")
        for group in ("tail", "mid", "head"):
            metrics[f"{group}_recall"] = class_table.loc[class_table.frequency_group == group, "recall"].mean()
        for group in ("tail", "mid", "head"):
            flow = flows[(flows.true_group == "tail") & (flows.predicted_group == group)]
            metrics[f"tail_to_{group}_error"] = flow.error_probability.iloc[0]
        endpoint_rows[(run["variant"], run["seed"])] = {
            "run": run,
            "split": evaluation["split"],
            **metrics,
        }
    return {
        "study": study,
        "analysis": analysis,
        "prepared": prepared,
        "analysis_provenance": provenance,
        "classes": classes,
        "samples": samples,
        "plans": plan,
        "endpoints": endpoint_rows,
    }


def effect_rows(full_endpoints, limited_endpoints):
    seeds = {seed for _, seed in full_endpoints}
    if seeds != {seed for _, seed in limited_endpoints}:
        raise ValueError("Support cohorts do not contain matching seeds")
    if {variant for variant, _ in full_endpoints} != set(VARIANTS) or {variant for variant, _ in limited_endpoints} != set(VARIANTS):
        raise ValueError("Both support cohorts must contain both paired treatments")
    rows = []
    endpoints = [*full_endpoints.values(), *limited_endpoints.values()]
    keys = sorted(set.intersection(*(set(endpoint) for endpoint in endpoints)))
    for seed in sorted(seeds):
        for metric in keys:
            if metric in {"run", "split"}:
                continue
            values = {variant: (full_endpoints[(variant, seed)].get(metric), limited_endpoints[(variant, seed)].get(metric)) for variant in VARIANTS}
            if not all(value is not None and np.isfinite(value) for pair in values.values() for value in pair):
                continue
            species_change = values["species_regularized"][1] - values["species_regularized"][0]
            hierarchy_change = values["hierarchy_regularized"][1] - values["hierarchy_regularized"][0]
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "species_full_support": values["species_regularized"][0],
                    "species_lower_support": values["species_regularized"][1],
                    "species_support_effect": species_change,
                    "hierarchy_full_support": values["hierarchy_regularized"][0],
                    "hierarchy_lower_support": values["hierarchy_regularized"][1],
                    "hierarchy_support_effect": hierarchy_change,
                    "hierarchy_minus_species_full_support": values["hierarchy_regularized"][0] - values["species_regularized"][0],
                    "hierarchy_minus_species_lower_support": values["hierarchy_regularized"][1] - values["species_regularized"][1],
                    "hierarchy_x_support_difference_in_differences": hierarchy_change - species_change,
                }
            )
    return rows


def compare(full, limited, output):
    """Write paired support effects after strict held-out and recipe checks."""
    if full["classes"] != limited["classes"]:
        raise ValueError("Class vocabulary, ordering, taxonomy or training counts differ")
    for key in ("analysis_sha256", "contrast_code_sha256"):
        if full["analysis_provenance"][key] != limited["analysis_provenance"][key]:
            raise ValueError("Support cohorts were analyzed with different code")
    seeds = {run["seed"] for run in full["plans"]}
    if seeds != {run["seed"] for run in limited["plans"]}:
        raise ValueError("Support cohorts do not contain matching seeds")
    split = None
    for seed in sorted(seeds):
        for variant in VARIANTS:
            left, right = full["endpoints"][(variant, seed)], limited["endpoints"][(variant, seed)]
            if any(left["run"].get(key) != right["run"].get(key) for key in RUN_FIELDS):
                raise ValueError(f"Training recipes differ for {variant}, seed {seed}")
            if left["split"] != right["split"]:
                raise ValueError("Evaluation splits differ between support cohorts")
            split = split or left["split"]
    if split != "validation":
        raise ValueError("Support sensitivity is validation-only until analysis choices are frozen")
    samples = validate_evaluation_rows(full["samples"], limited["samples"], split)
    rows = effect_rows(full["endpoints"], limited["endpoints"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows).sort_values(["metric", "seed"])
    table.to_csv(output / "support-effects.csv", index=False)
    interaction = table[
        [
            "seed",
            "metric",
            "species_support_effect",
            "hierarchy_support_effect",
            "hierarchy_minus_species_full_support",
            "hierarchy_minus_species_lower_support",
            "hierarchy_x_support_difference_in_differences",
        ]
    ]
    interaction.to_csv(output / "hierarchy-by-support.csv", index=False)
    write_json(
        output / "provenance.json",
        {
            "full_support_prepared_sha256": digest(full["study"] / "prepared.json"),
            "lower_support_prepared_sha256": digest(limited["study"] / "prepared.json"),
            "full_support_plan_sha256": digest(full["study"] / "plan.json"),
            "lower_support_plan_sha256": digest(limited["study"] / "plan.json"),
            "full_support_config_sha256": digest(full["study"] / "config.json"),
            "lower_support_config_sha256": digest(limited["study"] / "config.json"),
            "full_support_analysis_sha256": digest(full["analysis"] / "provenance.json"),
            "lower_support_analysis_sha256": digest(limited["analysis"] / "provenance.json"),
            "script_sha256": digest(Path(__file__)),
            "evaluation_split": split,
            "identical_evaluation_samples": samples,
            "lower_unique_support_cap": json.loads((limited["study"] / "config.json").read_text()).get("train_support_cap"),
            "seeds": sorted(seeds),
            "treatments": list(VARIANTS),
            "difference_convention": "lower unique support minus full support; hierarchy interaction is hierarchy effect minus species-only effect",
            "interpretation": "Seed-wise descriptive contrasts; two seeds do not establish population uncertainty.",
        },
    )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_study", type=Path)
    parser.add_argument("lower_support_study", type=Path)
    parser.add_argument("full_analysis", type=Path)
    parser.add_argument("lower_support_analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(load_cohort(args.full_study, args.full_analysis), load_cohort(args.lower_support_study, args.lower_support_analysis), args.output)


if __name__ == "__main__":
    main()
