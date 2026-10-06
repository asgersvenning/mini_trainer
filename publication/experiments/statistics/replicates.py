"""Compound calibration/sampling bootstrap of mini_metrics results for one evidence-snapshot study.

Each replicate draws a label-stratified calibration/reporting partition of observations
(mini_metrics' own split), then resamples observations with replacement within each part,
so calibration and reporting stay disjoint. Replicate 0 keeps its partition unresampled.
Every run of the study uses the same draws, so replicate results pair across runs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.special import softmax


def study_ranks(taxonomy):
    """Rank key columns chosen by mini_trainer's rule: ranks with more than one taxon in the vocabulary."""
    from mini_trainer.integrations.gbif import TAXONOMY_KEYS, select_levels

    columns = [f"{rank}Key" for rank in TAXONOMY_KEYS if f"{rank}Key" in taxonomy]
    rows = [
        OrderedDict((c.removesuffix("Key"), (str(v), "")) for c, v in zip(columns, row))
        for row in taxonomy[columns].itertuples(index=False)
    ]
    return [f"{level}Key" for level in select_levels(None, rows)]


COLUMNS = {
    "replicates": ["run_id", "replicate", "setting", "level", "metric", "value"],
    "per_class": ["run_id", "replicate", "level", "class", "metric", "value", "weight"],
    "thresholds": ["run_id", "replicate", "level", "threshold"],
}
PER_CLASS_METRICS = "^(accuracy|precision|recall|f1|coverage)$"


PARENT_RULES = ("leaf_sum", "winner_ancestor")


PREPARE_WORKERS = 4  # Concurrent run preparations; each holds one run's score matrix in memory.
BLOCK = 5000  # Rows per block, so full-vocabulary score matrices never expand in memory at once.


def blocks(frame):
    """Yield score columns ``c<j>`` as float64 arrays of at most BLOCK rows."""
    columns = [i for i, c in enumerate(frame.columns) if c.startswith("c") and c[1:].isdigit()]
    for start in range(0, len(frame), BLOCK):
        yield frame.iloc[start : start + BLOCK, columns].to_numpy(np.float64)


def native_predictions(scores, classes):
    """Per-rank argmax and confidence from each rank's native log-probabilities."""
    predictions, confidences = [], []
    for rank, frame in enumerate(scores):
        keys = classes[classes["rank"] == rank].sort_values("class_index").key.to_numpy()
        winners, maxima = zip(*((block.argmax(1), block.max(1)) for block in blocks(frame)))
        predictions.append(keys[np.concatenate(winners)])
        confidences.append(np.exp(np.concatenate(maxima)))
    return np.stack(predictions, 1), np.stack(confidences, 1)


def top1_predictions(scores, taxonomy, rule, ranks):
    """Per-rank predictions and confidences for a flat head from its species scores.

    ``leaf_sum`` sums species probabilities within each parent; it is the native parent
    output of the ablations' bottom-up HierarchicalClassifier. ``winner_ancestor`` maps the
    winning species to its ancestors and keeps its confidence (mini_metrics add_combinations).
    """
    taxa = taxonomy.sort_values("class_id")
    groups = {}
    for rank in ranks:  # Species-to-parent indicator matrices: probabilities @ matrix sums mass per parent.
        names, members = np.unique(taxa[rank].to_numpy(), return_inverse=True)
        groups[rank] = names, csr_matrix((np.ones(len(members)), (np.arange(len(members)), members)))
    predictions, confidences = [], []
    for block in blocks(scores):
        probabilities = softmax(block, axis=1)
        winner = probabilities.argmax(1)
        block_predictions, block_confidences = [], []
        for rank in ranks:
            if rule == "winner_ancestor" or rank == ranks[0]:
                block_predictions.append(taxa[rank].to_numpy()[winner])
                block_confidences.append(probabilities.max(1))
                continue
            names, members = groups[rank]
            mass = np.asarray(probabilities @ members)
            block_predictions.append(names[mass.argmax(1)])
            block_confidences.append(mass.max(1))
        predictions.append(np.stack(block_predictions, 1))
        confidences.append(np.stack(block_confidences, 1))
    return np.concatenate(predictions), np.concatenate(confidences)


def draw_replicates(images, replicates, seed, calibration_fraction):
    """Return an int8 (images x replicates) matrix: +k reporting draws, -k calibration draws."""
    from mini_metrics.data import MetricDF

    observation = images.observation_id.fillna(images.image_id).to_numpy()
    units, unit_of_image = np.unique(observation, return_inverse=True)
    unit_label = pd.Series(images.speciesKey.to_numpy()).groupby(unit_of_image).first().to_numpy()
    frame = MetricDF(
        {
            "instance_id": np.arange(len(units)),
            "filename": units.astype(str).astype(object),
            "level": np.zeros(len(units), dtype=np.int64),
            "label": unit_label.astype(object),
            "prediction": unit_label.astype(object),
            "confidence": np.ones(len(units)),
            "threshold": np.zeros(len(units)),
        }
    )
    weights = np.zeros((len(images), replicates + 1), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for replicate in range(replicates + 1):
        parts = frame.split((calibration_fraction, 1 - calibration_fraction), seed=int(rng.integers(2**63)))
        unit_weight = np.zeros(len(units), dtype=np.int64)
        for sign, part in zip((-1, 1), parts):
            chosen = np.asarray(part.instance_id)
            counts = np.ones(len(chosen), dtype=np.int64)
            if replicate:
                counts = rng.multinomial(len(chosen), np.full(len(chosen), 1 / len(chosen)))
            unit_weight[chosen] = sign * counts
        if np.abs(unit_weight).max() > 127:
            raise ValueError("A unit was drawn more than 127 times; int8 weights would overflow")
        weights[:, replicate] = unit_weight[unit_of_image]
    return weights


def metric_frame(labels, predictions, confidences, rows, known=None):
    """Expand drawn rows (with repeats) to a MetricDF; each drawn copy is a separate instance."""
    from mini_metrics.data import MetricDF

    n, ranks = len(rows), labels.shape[1]
    extra = {} if known is None else {"known_label": known[rows].T.ravel()}
    return MetricDF(
        extra
        | {
            "instance_id": np.tile(np.arange(n), ranks),
            "filename": np.tile(np.arange(n), ranks).astype(str).astype(object),
            "level": np.repeat(np.arange(ranks), n),
            "label": labels[rows].T.ravel().astype(object),
            "prediction": predictions[rows].T.ravel().astype(object),
            "confidence": confidences[rows].T.ravel(),
            "threshold": np.zeros(n * ranks),
        }
    )


def evaluate_run(task):
    from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold, evaluate_file

    run_id, predictions_path, weights_path, columns = task
    wide = pd.read_parquet(predictions_path).pivot(index="row", columns="level")
    labels, predictions = wide["label"].to_numpy(), wide["prediction"].to_numpy()
    confidences, known = wide["confidence"].to_numpy(), wide["known"].to_numpy().astype(bool)
    weights = np.load(weights_path, mmap_mode="r")
    aggregate, per_class, thresholds = [], [], []
    for replicate in columns:
        w = np.asarray(weights[:, replicate], dtype=np.int64)
        calibration, reporting = np.repeat(np.arange(len(w)), np.maximum(-w, 0)), np.repeat(np.arange(len(w)), np.maximum(w, 0))
        tau = OptimalConfidenceThreshold(crit=MacroF1)(metric_frame(labels, predictions, confidences, calibration, known), verbose=0)
        tau = [float(tau[level]) for level in range(labels.shape[1])]
        report = metric_frame(labels, predictions, confidences, reporting, known)
        thresholds += [(run_id, replicate, level, t) for level, t in enumerate(tau)]
        settings = [("calibrated", tau, False), ("zero", [0.0] * len(tau), False)]
        if not known.all():  # Open-set data: also score only taxa inside the vocabulary.
            settings += [("calibrated_known_only", tau, True), ("zero_known_only", [0.0] * len(tau), True)]
        for setting, threshold, known_only in settings:
            values = evaluate_file(
                report, threshold=threshold, known_only=known_only, simple=True, hierarchical=False, verbose=0, pattern="^(?!optimal)"
            )
            aggregate += [
                (run_id, replicate, setting, level, name, float(v)) for name, by_level in values.items() for level, v in by_level.items()
            ]
        values = evaluate_file(report, threshold=tau, per_class=True, simple=True, hierarchical=False, verbose=0, pattern=PER_CLASS_METRICS)
        for name, by_level in values.items():
            per_class += [
                (run_id, replicate, level, cls, name, float(v), float(weight))
                for level, classes in by_level.items()
                for cls, (v, weight) in classes.items()
            ]
    return aggregate, per_class, thresholds


def write_task(task, output, study):
    """Evaluate one task and write its rows to disk, keeping worker results out of memory."""
    run_id, _, _, columns = task
    for kind, rows in zip(COLUMNS, evaluate_run(task)):
        path = output / kind / run_id / f"{columns[0]:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows, columns=COLUMNS[kind]).assign(study=study), preserve_index=False), path)


def prepare_run(run_id, snapshot, study, catalog, images, taxonomy, classes, ranks, labels, known, flat_parent_rule, output):
    """Write one run's top-1 predictions in the mini_metrics long layout; return its parent rule."""
    paths = catalog.query("study == @study and kind == 'scores' and run_id == @run_id").sort_values("rank").path
    scores = [pd.read_parquet(snapshot / path) for path in paths]
    if any(not np.array_equal(frame.image_id.to_numpy(), images.image_id.to_numpy()) for frame in scores):
        raise ValueError(f"Scores are not aligned with the study's evaluation images: {run_id}")
    # Hierarchical heads output every rank natively; a flat head's parents need a chosen rule.
    if len(scores) == len(ranks):
        rule = "native"
        predictions, confidences = native_predictions(scores, classes)
    else:
        rule = flat_parent_rule
        predictions, confidences = top1_predictions(scores[0], taxonomy, rule, ranks)
    # Long table in the mini_metrics column layout: one row per image and rank.
    rows, levels = np.indices(labels.shape)
    table = {"study": [study] * rows.size, "run_id": [run_id] * rows.size, "row": rows.ravel()}
    table |= {"image_id": images.image_id.to_numpy()[rows.ravel()]}
    table |= {"level": levels.ravel(), "label": labels.ravel(), "prediction": predictions.astype(str).ravel(), "known": known.ravel()}
    path = output / "predictions" / f"{run_id}.parquet"
    pq.write_table(pa.table(table | {"confidence": confidences.ravel()}), path)
    return rule


def run_study(snapshot, study, output, replicates, seed, calibration_fraction, workers, chunk, flat_parent_rule):
    snapshot, output = Path(snapshot), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    catalog = pd.read_csv(snapshot / "catalog.csv")
    files = catalog[catalog.study == study].set_index("kind")
    images = pd.read_parquet(snapshot / files.loc["images", "path"])
    taxonomy = pd.read_parquet(snapshot / files.loc["taxonomy", "path"])
    runs = pd.read_parquet(snapshot / "runs.parquet").query("study == @study and status == 'complete'")
    if runs.split.nunique() != 1:
        raise ValueError("Runs of one study must share an evaluation split")
    images = images[images.split == runs.split.iloc[0]].reset_index(drop=True)
    ranks = study_ranks(taxonomy)
    labels = images[ranks].to_numpy().astype(str)

    weights = draw_replicates(images, replicates, seed, calibration_fraction)
    np.save(output / "weights.npy", weights)
    pq.write_table(
        pa.table({"image_id": images.image_id, **{f"r{b}": weights[:, b] for b in range(weights.shape[1])}}), output / "weights.parquet"
    )

    (output / "predictions").mkdir()
    classes = pd.read_parquet(snapshot / files.loc["classes", "path"])
    # Known: the true taxon is in the models' vocabulary at that rank (all true for closed-set studies).
    vocabulary = [set(classes.loc[classes["rank"] == r, "key"].astype(str)) for r in range(len(ranks))]
    known = np.stack([pd.Series(labels[:, r]).isin(vocabulary[r]).to_numpy() for r in range(len(ranks))], 1)
    # Full-vocabulary score matrices are large (Global Lepidoptera: 633k x 12k); prepare a few runs at once.
    with ProcessPoolExecutor(min(workers, PREPARE_WORKERS)) as pool:
        prepare = partial(
            prepare_run,
            snapshot=snapshot,
            study=study,
            catalog=catalog,
            images=images,
            taxonomy=taxonomy,
            classes=classes,
            ranks=ranks,
            labels=labels,
            known=known,
            flat_parent_rule=flat_parent_rule,
            output=output,
        )
        parent_rules = dict(zip(runs.run_id, pool.map(prepare, runs.run_id)))
    columns = range(weights.shape[1])
    tasks = [
        (run_id, output / "predictions" / f"{run_id}.parquet", output / "weights.npy", columns[i : i + chunk])
        for run_id in runs.run_id
        for i in range(0, len(columns), chunk)
    ]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(partial(write_task, output=output, study=study), tasks))
    # Small tables become single files; per-class rows stay one Parquet dataset per run.
    for kind in ("replicates", "thresholds"):
        parts = sorted((output / kind).rglob("*.parquet"))
        pq.write_table(pa.concat_tables(pq.read_table(path) for path in parts), output / f"{kind}.parquet")
        shutil.rmtree(output / kind)

    (output / "weights.npy").unlink()
    import mini_metrics

    # Hash the imported source: an installed version string cannot identify a PYTHONPATH checkout.
    package = Path(mini_metrics.__file__).parent
    source = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(package.glob("*.py")))).hexdigest()
    (output / "resampling.json").write_text(
        json.dumps(
            {
                "study": study,
                "snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
                "split": runs.split.iloc[0],
                "replicates": replicates,
                "seed": seed,
                "numpy": np.__version__,
                "calibration_fraction": calibration_fraction,
                "partition": "mini_metrics MetricDF.split of observations, stratified by species label",
                "resampling_unit": "observation_id (image_id when missing), with replacement within each part",
                "replicate_0": "partition only, no resampling",
                "ranks": ranks,
                "rank_rule": "mini_trainer select_levels: ranks with more than one taxon in the vocabulary",
                "known_only_settings": "added when some true taxa are outside the vocabulary; thresholds calibrated on all rows",
                "weights": "int8 images x replicates; +k reporting draws, -k calibration draws",
                "parent_rules": parent_rules,
                "parent_rule_definitions": {
                    "native": "hierarchical head; each rank's own output (bottom-up heads sum species probabilities)",
                    "leaf_sum": "flat head; parent probability is the sum of its species probabilities",
                    "winner_ancestor": "flat head; ancestor of the winning species with its confidence",
                },
                "calibration": "OptimalConfidenceThreshold(crit=MacroF1) defaults per level on the calibration draws",
                "settings": {"calibrated": "thresholds from calibration draws", "zero": "threshold 0, no abstention"},
                "mini_metrics": {"version": importlib.metadata.version("mini_metrics"), "path": str(package), "source_sha256": source},
                "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path, help="Local evidence snapshot directory")
    parser.add_argument("study")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--chunk", type=int, default=25, help="Replicates per worker task")
    parser.add_argument(
        "--flat-parent-rule",
        choices=PARENT_RULES,
        default="leaf_sum",
        help="Parent predictions for flat heads (leaf_sum matches the bottom-up head)",
    )
    args = parser.parse_args()
    run_study(
        args.snapshot,
        args.study,
        args.output,
        args.replicates,
        args.seed,
        args.calibration_fraction,
        args.workers,
        args.chunk,
        args.flat_parent_rule,
    )


if __name__ == "__main__":
    main()
