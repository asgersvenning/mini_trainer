"""Artifact-only frequency, confusion, calibration and prototype diagnostics."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata

from .data import digest, write_json
from .study import factorial_contrasts

GROUPS = ("tail", "mid", "head")


def frequency_groups(counts):
    """Match the study's stable, equal-class-count frequency thirds."""
    counts = np.asarray(counts)
    if counts.ndim != 1 or len(counts) < 3 or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("At least three positive finite training counts required")
    groups = np.empty(len(counts), dtype=int)
    for index, members in enumerate(np.array_split(np.argsort(counts, kind="stable"), 3)):
        groups[members] = index
    return groups


def association(x, y):
    """Descriptive rank association and linear slope; constants are undefined."""
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "spearman": None, "slope": None}
    return {"n": len(x), "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]), "slope": float(np.polyfit(x, y, 1)[0])}


def prediction_analysis(logits, target, counts, bins=10):
    """Analyze empirical and equal-observed-class priors without fitting calibration."""
    counts = np.asarray(counts)
    groups = frequency_groups(counts)
    logits, target = np.asarray(logits, dtype=float), np.asarray(target)
    k = len(counts)
    if bins < 1 or logits.ndim != 2 or logits.shape != (len(target), k) or not len(target) or not np.isfinite(logits).all():
        raise ValueError("Invalid logits, targets or reliability bins")
    if target.ndim != 1 or not np.issubdtype(target.dtype, np.integer) or (target < 0).any() or (target >= k).any():
        raise ValueError("Targets must be valid integer class indices")
    predicted = logits.argmax(1)
    matrix = np.bincount(target * k + predicted, minlength=k * k).reshape(k, k)
    support = matrix.sum(1)
    observed = support > 0
    conditional = np.divide(matrix, support[:, None], out=np.zeros((k, k)), where=support[:, None] > 0)
    recall = np.where(observed, conditional.diagonal(), np.nan)
    balanced_mass = conditional[observed].mean(0)
    per_class = pd.DataFrame(
        {
            "class_index": np.arange(k),
            "train_count": counts,
            "support": support,
            "frequency_group": [GROUPS[i] for i in groups],
            "recall": recall,
            "predicted_mass_empirical": matrix.sum(0) / len(target),
            "predicted_mass_balanced": balanced_mass,
        }
    )
    summary = {
        "samples": len(target),
        "observed_classes": int(observed.sum()),
        "missing_classes": np.flatnonzero(~observed).tolist(),
        "frequency_recall": association(np.log(counts), recall),
        "frequency_balanced_prediction_mass": association(np.log(counts), balanced_mass),
    }
    flows = []
    errors = conditional.copy()
    np.fill_diagonal(errors, 0)
    for source, label in enumerate(GROUPS):
        selected = (groups == source) & observed
        for destination, other in enumerate(GROUPS):
            mass = errors[selected][:, groups == destination].sum(1)
            total = errors[selected].sum(1)
            flows.append(
                {
                    "true_group": label,
                    "predicted_group": other,
                    "source_classes": int(selected.sum()),
                    "destination_classes": int((groups == destination).sum()),
                    "error_probability": float(mass.mean()) if len(mass) else None,
                    "share_of_source_errors": float(mass.sum() / total.sum()) if total.sum() else None,
                }
            )
    probabilities = softmax(logits, axis=1)
    prior_weights = {"empirical": np.full(len(target), 1 / len(target)), "balanced": 1 / support[target] / observed.sum()}
    for prior, weight in prior_weights.items():
        per_class["soft_prediction_mass_" + prior] = weight @ probabilities
    summary["frequency_balanced_soft_mass"] = association(np.log(counts), per_class.soft_prediction_mass_balanced)
    confidence = probabilities.max(1)
    correct = predicted == target
    nll = logsumexp(logits, axis=1) - logits[np.arange(len(target)), target]
    brier = (probabilities**2).sum(1) - 2 * probabilities[np.arange(len(target)), target] + 1
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    for prior, weight in prior_weights.items():
        ece = 0.0
        for index in range(bins):
            selected = bin_id == index
            mass = weight[selected].sum()
            acc = float(np.average(correct[selected], weights=weight[selected])) if mass else None
            conf = float(np.average(confidence[selected], weights=weight[selected])) if mass else None
            if mass:
                ece += mass * abs(acc - conf)
            reliability.append(
                {
                    "prior": prior,
                    "bin": index,
                    "lower": index / bins,
                    "upper": (index + 1) / bins,
                    "samples": int(selected.sum()),
                    "mass": float(mass),
                    "accuracy": acc,
                    "confidence": conf,
                }
            )
        summary[prior] = {
            "accuracy": float(weight @ correct),
            "nll": float(weight @ nll),
            "brier": float(weight @ brier),
            "ece": float(ece),
        }
    source, destination = np.nonzero(matrix - np.diag(matrix.diagonal()))
    pairs = pd.DataFrame(
        {
            "true_class": source,
            "predicted_class": destination,
            "errors": matrix[source, destination],
            "conditional_probability": conditional[source, destination],
        }
    )
    return summary, per_class, pd.DataFrame(flows), pd.DataFrame(reliability), pairs


def prototype_analysis(weights, counts, samples=8192, seed=20261004):
    """Common angular-reference head; not the unnormalized head's BN/bias forward."""
    weights = np.asarray(weights, dtype=float)
    groups = frequency_groups(counts)
    if weights.ndim != 2 or weights.shape[0] != len(counts) or weights.shape[1] <= 2 or not np.isfinite(weights).all():
        raise ValueError("Invalid prototype matrix")
    norms = np.linalg.norm(weights, axis=1)
    if (norms == 0).any() or samples < 1:
        raise ValueError("Nonzero prototypes and positive null sample count required")
    directions = weights / norms[:, None]
    cosine = np.clip(directions @ directions.T, -1, 1)
    np.fill_diagonal(cosine, -np.inf)
    nearest = cosine.argmax(1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(directions @ directions.T), 0)
    spectrum = eigenvalues[eigenvalues > 0] / eigenvalues.sum()
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(counts), dtype=int)
    mass = np.zeros(len(counts))
    for start in range(0, samples, 512):
        points = rng.normal(size=(min(512, samples - start), weights.shape[1]))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        scores = np.sqrt(weights.shape[1] - 2) * np.arcsin(np.clip(points @ directions.T, -1 + 1e-7, 1 - 1e-7))
        # Exact angular ties have no unique winner; do not favor the first index.
        maxima = scores == scores.max(1, keepdims=True)
        wins += np.bincount(scores.argmax(1), minlength=len(counts))
        if (maxima.sum(1) > 1).any():
            raise ValueError("Duplicate/tied prototypes make hard null occupancy ambiguous")
        mass += softmax(scores, axis=1).sum(0)
    occupancy = wins / samples
    frame = pd.DataFrame(
        {
            "class_index": np.arange(len(counts)),
            "weight_norm": norms,
            "nearest_class": nearest,
            "nearest_angle_degrees": np.degrees(np.arccos(cosine.max(1))),
            "nearest_frequency_group": [GROUPS[i] for i in groups[nearest]],
            "null_win_probability": occupancy,
            "null_softmax_mass": mass / samples,
            "null_win_standard_error": np.sqrt(occupancy * (1 - occupancy) / samples),
        }
    )
    summary = {
        "null_reference": "unit prototypes, uniform sphere, cosine-to-z logits, zero bias",
        "null_samples": samples,
        "null_seed": seed,
        "mean_resultant_length": float(np.linalg.norm(directions.mean(0))),
        "effective_rank": float(np.exp(-(spectrum * np.log(spectrum)).sum())),
        "frequency_null_occupancy": association(np.log(counts), occupancy),
    }
    summary["group_angles"] = []
    for source, name in enumerate(GROUPS):
        for destination, other in enumerate(GROUPS):
            values = cosine[np.ix_(groups == source, groups == destination)]
            values = values[np.isfinite(values)]  # Exclude the masked self pairs.
            summary["group_angles"].append(
                {
                    "source": name,
                    "destination": other,
                    "directed_pairs": len(values),
                    "mean_angle_degrees": float(np.degrees(np.arccos(values)).mean()) if len(values) else None,
                }
            )
    return summary, frame


def verified(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Missing or changed artifact: {path}")
    return path


def analyze(root, output, variants=None, geometry=False, samples=8192, seed=20261004):
    """Consume a prepared study or verified local mirror, leaving inputs untouched."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Write analysis outside the immutable study root")
    output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((root / "prepared.json").read_text())
    for name in ["classes.json", "samples.parquet"]:
        verified(root / name, prepared["files"][name])
    spec = json.loads((root / "classes.json").read_text())
    frame = pd.read_parquet(root / "samples.parquet")
    mapping = spec["cls2idx"]
    if sorted(mapping.values()) != list(range(len(spec["counts"]))):
        raise ValueError("Class ordering is not a complete bijection")
    taxa = frame[["label", "speciesKey", "genusKey", "familyKey"]].drop_duplicates().sort_values("label")
    if taxa.label.tolist() != list(range(len(mapping))) or any(mapping[str(r.speciesKey)] != r.label for r in taxa.itertuples()):
        raise ValueError("Taxonomy or class ordering differs from class specification")
    reports, provenance, skipped = [], {}, []
    contrast_rows = []
    for directory in sorted((root / "runs").iterdir()):
        attempts = sorted(directory.glob("attempt-*"))
        if not attempts or not (attempts[-1] / "complete.json").exists():
            skipped.append(directory.name)
            continue
        attempt = attempts[-1]
        manifest = json.loads((attempt / "complete.json").read_text())
        for name in ["run.json", "evaluation.json"]:
            verified(attempt / name, manifest[name])
        run = json.loads((attempt / "run.json").read_text())
        if not run.get("variant") or (variants and run["variant"] not in variants):
            continue
        evaluation = json.loads((attempt / "evaluation.json").read_text())
        path = verified(attempt / "predictions.npz", manifest["predictions.npz"])
        data = np.load(path, allow_pickle=False)
        selected = frame[frame.split == evaluation["split"]]
        if not np.array_equal(data["sample_id"], selected.sample_id.to_numpy()) or not np.array_equal(
            data["target"], selected.label.to_numpy()
        ):
            raise ValueError("Prediction sample identity/order or split differs from prepared cohort")
        summary, classes, flows, reliability, pairs = prediction_analysis(data["logits"], data["target"], spec["counts"])
        data.close()
        for rank in ["genusKey", "familyKey"]:
            labels = taxa[rank].to_numpy()
            pairs["same_" + rank] = labels[pairs.true_class] == labels[pairs.predicted_class]
        classes = classes.merge(taxa, left_on="class_index", right_on="label", validate="one_to_one").drop(columns="label")
        if geometry:
            from mini_trainer.modeling import Classifier, classification_module

            weights = verified(attempt / "model/weights/last.pt", manifest["model/weights/last.pt"])
            from mini_trainer.hierarchical.model import HierarchicalClassifier

            head_cls = HierarchicalClassifier if "hierarchy" in spec else Classifier
            model, _ = head_cls.build(weights=str(weights), device="cpu", model_args={"pretrained": False}, skip_spherical_init=True)
            head = classification_module(model)
            geometry_summary, geometry_classes = prototype_analysis(head.linear.weight.detach().numpy(), spec["counts"], samples, seed)
            summary["geometry"] = geometry_summary
            classes = classes.merge(geometry_classes, on="class_index", validate="one_to_one")
            del model, head
        destination = output / directory.name
        destination.mkdir()
        for name, table in [("classes", classes), ("confusion_flows", flows), ("reliability", reliability), ("error_pairs", pairs)]:
            table.to_csv(destination / f"{name}.csv", index=False)
        report = {"run": run, "split": evaluation["split"], "metrics": summary}
        contrast_rows.append(
            {
                **run,
                "split": evaluation["split"],
                "balanced_nll": summary["balanced"]["nll"],
                "balanced_brier": summary["balanced"]["brier"],
                "tail_to_tail_error": float(
                    flows[(flows.true_group == "tail") & (flows.predicted_group == "tail")].error_probability.iloc[0]
                ),
            }
        )
        write_json(destination / "summary.json", report)
        reports.append(report)
        provenance[directory.name] = {
            "attempt": str(attempt),
            "completion_manifest_sha256": digest(attempt / "complete.json"),
            "verified": {n: manifest[n] for n in ["run.json", "evaluation.json", "predictions.npz"]},
        }
        if geometry:
            provenance[directory.name]["verified"]["model/weights/last.pt"] = manifest["model/weights/last.pt"]
    if not reports:
        raise ValueError("No completed matching runs")
    write_json(output / "report.json", {"runs": reports, "skipped_incomplete": skipped})
    # Preserve the study's complete-cube and within-seed comparability checks.
    contrasts = factorial_contrasts(contrast_rows, metrics=("balanced_nll", "balanced_brier", "tail_to_tail_error"))
    write_json(output / "factorial.json", contrasts)
    write_json(
        output / "provenance.json",
        {
            "input_root": str(root),
            "prepared_sha256": digest(root / "prepared.json"),
            "training_commit": prepared.get("git_commit"),
            "analysis_sha256": digest(Path(__file__)),
            "contrast_code_sha256": digest(Path(__file__).with_name("study.py")),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy"] + (["torch"] if geometry else [])},
            "variants": variants,
            "geometry": geometry,
            "null_seed": seed,
            "null_samples": samples,
            "runs": provenance,
        },
    )
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--geometry", action="store_true", help="Also load completed checkpoints on CPU")
    parser.add_argument("--null-samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    analyze(args.root, args.output, args.variants, args.geometry, args.null_samples, args.seed)


if __name__ == "__main__":
    main()
