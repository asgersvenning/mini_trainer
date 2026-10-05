"""Export completed ablation cohorts as a self-describing snapshot of tidy tables.

Each cohort directory holds ``study/`` (prepared study) and optionally ``analysis/``.
Every file carries its identifying columns; ``catalog.csv`` selects files and
``runs.parquet`` holds the per-run factors, so readers never parse paths.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import create
from .training_ablations.analysis import frequency_groups, verified
from .training_ablations.data import digest, write_json

SCHEMA_VERSION = 1
CURVES = {"epochs", "learning"}
RUN_SETTINGS = ("size", "dtype", "batch_size")
TRAINING = ("wall_seconds", "images_per_second_including_validation_and_logging", "peak_allocated_bytes")


def dataset_name(config):
    return "plantnet300k" if "data_index" in config else "global_lepidoptera"


def export_cohort(cohort, output, source_metadata=None):
    """Write one cohort's tables and return their catalog rows."""
    study, name = cohort / "study", cohort.name
    config = json.loads((study / "config.json").read_text())
    prepared = json.loads((study / "prepared.json").read_text())
    for filename in ("classes.json", "samples.parquet"):
        verified(study / filename, prepared["files"][filename])
    classes = json.loads((study / "classes.json").read_text())
    samples = pd.read_parquet(study / "samples.parquet")
    dataset = dataset_name(config)
    if "gbifID" not in samples and source_metadata and config.get("source_metadata"):
        # PlantNet images are named by their source hash; recover the observation from source metadata.
        source = pd.read_csv(source_metadata, dtype=str, usecols=["PN_hash", "PN_observation_id"]).set_index("PN_hash")
        samples["gbifID"] = samples.sample_id.map(lambda path: Path(path).stem).map(source.PN_observation_id)
    caveat = "validation-only screening" if config.get("screening") else ""
    if "gbifID" not in samples or samples.gbifID.isna().any():
        caveat = "; ".join(filter(None, [caveat, "missing observation ids"]))
    catalog = []

    def write(table, set_, kind, run_id=None):
        path = Path(set_) / name / f"{run_id or kind}.parquet"
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table if isinstance(table, pa.Table) else pa.Table.from_pandas(table, preserve_index=False), output / path)
        catalog.append(
            {
                "path": path.as_posix(),
                "set": set_,
                "kind": kind,
                "schema_version": SCHEMA_VERSION,
                "study": name,
                "dataset": dataset,
                "run_id": run_id,
                "rows": pq.read_metadata(output / path).num_rows,
                "caveat": caveat,
            }
        )

    samples = samples.rename(columns={"sample_id": "image_id", "label": "class_id", "gbifID": "observation_id"})
    for column in ("observation_id", "scientificName"):
        if column not in samples:
            samples[column] = pd.Series(dtype="string", index=samples.index)
    # Source names can vary within a species; keep its first name as a label only.
    taxonomy = samples[["class_id", "speciesKey", "genusKey", "familyKey"]].drop_duplicates().sort_values("class_id")
    taxonomy["scientificName"] = taxonomy.class_id.map(samples.groupby("class_id").scientificName.first())
    if taxonomy.class_id.tolist() != list(range(len(classes["counts"]))):
        raise ValueError(f"Taxonomy does not cover the class ordering: {name}")
    taxonomy = taxonomy.assign(
        study=name, train_count=classes["counts"], frequency_group=[("tail", "mid", "head")[i] for i in frequency_groups(classes["counts"])]
    )
    write(taxonomy, "taxonomy", "taxonomy")
    # Training rows of capped-support cohorts are draws, so image_id repeats there.
    image_columns = ["image_id", "split", "class_id", "speciesKey", "genusKey", "familyKey", "observation_id"]
    write(samples[image_columns].assign(study=name), "images", "images")

    qualified = study / "qualified.json"
    settings = {key: config.get(key) for key in RUN_SETTINGS}
    settings |= {"backbone": prepared.get("pretrained_enum"), "num_classes": len(classes["counts"])}
    if qualified.exists():
        settings["batch_size"] = json.loads(qualified.read_text())["batch_size"]
    runs, aggregates, confusions = [], {}, []
    for run in json.loads((study / "plan.json").read_text()):
        row = {"study": name, "dataset": dataset, "run_id": run["id"], "git_commit": prepared.get("git_commit")}
        # Equal hashes mark studies that share classes and images, so R can pair them.
        row |= settings | {"classes_sha256": prepared["files"]["classes.json"], "samples_sha256": prepared["files"]["samples.parquet"]}
        row |= {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in run.items() if key != "id"}
        attempts = sorted((study / "runs" / run["id"]).glob("attempt-*"))
        complete = attempts and (attempts[-1] / "complete.json").exists()
        row["status"] = "complete" if complete else "incomplete" if attempts else "not_started"
        runs.append(row)
        if not complete:
            continue
        attempt = attempts[-1]
        manifest = json.loads((attempt / "complete.json").read_text())
        evaluation = json.loads(verified(attempt / "evaluation.json", manifest["evaluation.json"]).read_text())
        row |= {"attempt": attempt.name, "split": evaluation["split"]}
        training = json.loads((attempt / "train.json").read_text())
        row |= {key: training.get(key) for key in TRAINING}
        with np.load(verified(attempt / "predictions.npz", manifest["predictions.npz"]), allow_pickle=False) as data:
            logits, target, image_id = data["logits"], data["target"], data["sample_id"]
        selected = samples[samples.split == evaluation["split"]]
        if not (np.array_equal(image_id, selected.image_id) and np.array_equal(target, selected.class_id)):
            raise ValueError(f"Predictions are not aligned with the prepared split: {run['id']}")
        correct = logits.argmax(1) == target
        recall = pd.Series(correct).groupby(target).mean()
        if not np.isclose(correct.mean(), evaluation["accuracy"]) or not np.isclose(recall.mean(), evaluation["macro_recall"]):
            raise ValueError(f"Exported logits do not reproduce the recorded evaluation: {run['id']}")
        columns = [pa.array([name] * len(target)).dictionary_encode(), pa.array([run["id"]] * len(target)).dictionary_encode()]
        columns.append(pa.array(image_id))
        columns += [pa.array(logits[:, j]) for j in range(logits.shape[1])]
        names = ["study", "run_id", "image_id", *(f"c{j}" for j in range(logits.shape[1]))]
        write(pa.table(columns, names=names), "scores", "scores", run["id"])
        # Sparse per-epoch validation confusions recorded during training (AMP), one row per non-zero cell.
        for path in sorted((attempt / "model/logs/figures").glob("epoch-*/Confusion_matrix_lvl0/counts.npz")):
            with np.load(path, allow_pickle=False) as data:
                cells = {"true_class": data["rows"], "predicted_class": data["columns"], "count": data["counts"]}
            confusions.append(pd.DataFrame(cells).assign(study=name, run_id=run["id"], epoch=int(path.parents[1].name.split("-")[-1])))
        analysis = cohort / "analysis" / run["id"]
        for path in sorted(analysis.glob("*.csv")):
            aggregates.setdefault(path.stem, []).append(pd.read_csv(path).assign(study=name, run_id=run["id"]))
        if (analysis / "summary.json").exists():
            metrics = json.loads((analysis / "summary.json").read_text())["metrics"]
            aggregates.setdefault("summary", []).append(pd.json_normalize(metrics).assign(study=name, run_id=run["id"]))
    if confusions:
        write(pd.concat(confusions, ignore_index=True), "curves", "epoch_confusion")
    # Analyzer-specific tables are auxiliary: derivable from scores or specific to these ablations.
    for kind, frames in aggregates.items():
        write(pd.concat(frames, ignore_index=True), "curves" if kind in CURVES else "aux", kind)
    return runs, catalog


def export(cohorts, output, source_metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runs, catalog = [], []
    for cohort in cohorts:
        cohort_runs, cohort_catalog = export_cohort(Path(cohort), output, source_metadata)
        runs += cohort_runs
        catalog += cohort_catalog
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(runs), preserve_index=False), output / "runs.parquet")
    catalog.append({"path": "runs.parquet", "set": "runs", "kind": "runs", "schema_version": SCHEMA_VERSION, "rows": len(runs)})
    catalog = pd.DataFrame(catalog)
    catalog.to_csv(output / "catalog.csv", index=False)
    schemas = {}
    for row in catalog.itertuples():
        # Union across files: convenience aggregates gain columns for hierarchical or gate-logging runs.
        fields = schemas.setdefault(row.kind, {})
        fields |= {re.sub(r"^c\d+$", "c<class_id>", field.name): str(field.type) for field in pq.read_schema(output / row.path)}
    write_json(output / "schemas.json", schemas)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    revision += "-dirty" if dirty else ""
    lock = digest(root / "uv.lock")
    for name in ("uv.lock", "pyproject.toml"):
        shutil.copyfile(root / name, output / name)
    (output / "README.md").write_text(
        f"# Evidence snapshot\n\nExported by `publication.experiments.evidence` at `{revision}` (uv.lock sha256 `{lock}`).\n"
        "Select files with `catalog.csv`; join on `study`, `run_id`, `image_id` and `class_id`; per-run factors and the\n"
        "training commit are in `runs.parquet`, column types in `schemas.json`, checksums in `manifest.json`.\n"
        "`aux/` holds analyzer-specific tables (prototype geometry, flows, reliability); core sets are generic.\n"
        "Scores are raw species logits (`c<class_id>`); genus/family scores aggregate them through `taxonomy/`.\n"
    )
    names = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    write_json(output / "manifest.json", create(output, names, revision))
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="New snapshot directory")
    parser.add_argument("cohorts", type=Path, nargs="+", help="Cohort directories containing study/ and optionally analysis/")
    parser.add_argument("--source-metadata", type=Path, help="PlantNet metadata CSV with PN_hash and PN_observation_id")
    args = parser.parse_args()
    export(args.cohorts, args.output, args.source_metadata)


if __name__ == "__main__":
    main()
