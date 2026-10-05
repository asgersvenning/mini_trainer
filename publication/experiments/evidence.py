"""Export completed ablation cohorts as a self-describing snapshot of tidy tables.

Each cohort directory holds ``study/`` (prepared study), ``predictions/<run_id>/``
(``ParquetResultCollector`` output plus ``prediction.json``) and optionally ``analysis/``.
Every file carries its identifying columns; ``catalog.csv`` selects files and
``runs.parquet`` holds the per-run factors, so readers never parse paths.
"""

import argparse
import hashlib
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

SCHEMA_VERSION = 2
RUN_SETTINGS = ("size", "dtype", "batch_size")
TRAINING = ("wall_seconds", "images_per_second_including_validation_and_logging", "peak_allocated_bytes")


RANK_KEYS = ("speciesKey", "genusKey", "familyKey", "orderKey", "classKey")
RANK_NAMES = ("species", "genus", "family", "order", "class")


def dataset_name(config):
    if "dataset" in config:
        return config["dataset"]
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
    caveat = "missing observation ids" if "gbifID" not in samples or samples.gbifID.isna().any() else ""
    catalog = []

    def write(table, set_, kind, run_id=None, rank=None):
        path = Path(set_) / name / (f"{run_id}/rank-{rank}.parquet" if rank is not None else f"{run_id or kind}.parquet")
        (output / path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(table, list):  # Streamed shards
            with pq.ParquetWriter(output / path, table[0].schema) as writer:
                for shard in table:
                    writer.write_table(shard)
        else:
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
                "rank": rank,
                "rows": pq.read_metadata(output / path).num_rows,
                "caveat": caveat,
            }
        )

    samples = samples.rename(columns={"sample_id": "image_id", "label": "class_id", "gbifID": "observation_id"})
    for column in ("observation_id", "scientificName"):
        if column not in samples:
            samples[column] = pd.Series(dtype="string", index=samples.index)
    ranks = [key for key in RANK_KEYS if key in samples]
    if "taxonomy" in classes:  # The vocabulary's own taxonomy; evaluated images may miss or exceed it.
        rows = [[index, *path] for index, path in enumerate(classes["taxonomy"].values())]
        taxonomy = pd.DataFrame(rows, columns=["class_id", *RANK_KEYS[: len(rows[0]) - 1]])
        taxonomy["scientificName"] = pd.Series(dtype="string")
    else:  # Source names can vary within a species; keep its first name as a label only.
        taxonomy = samples[["class_id", *ranks]].drop_duplicates().sort_values("class_id")
        taxonomy["scientificName"] = taxonomy.class_id.map(samples.groupby("class_id").scientificName.first())
    if taxonomy.class_id.tolist() != list(range(len(classes["counts"]))):
        raise ValueError(f"Taxonomy does not cover the class ordering: {name}")
    taxonomy = taxonomy.assign(
        study=name, train_count=classes["counts"], frequency_group=[("tail", "mid", "head")[i] for i in frequency_groups(classes["counts"])]
    )
    write(taxonomy, "taxonomy", "taxonomy")
    # Training rows of capped-support cohorts are draws, so image_id repeats there.
    image_columns = ["image_id", "split", "class_id", *ranks, "observation_id"]
    write(samples[image_columns].assign(study=name), "images", "images")

    qualified = study / "qualified.json"
    settings = {key: config.get(key) for key in RUN_SETTINGS}
    settings |= {"backbone": prepared.get("pretrained_enum"), "num_classes": len(classes["counts"])}
    if qualified.exists():
        settings["batch_size"] = json.loads(qualified.read_text())["batch_size"]
    runs, aggregates, confusions, learning = [], {}, [], []
    rank_classes = {}
    report = cohort / "analysis" / "report.json"
    analysis_split = json.loads(report.read_text())["runs"][0]["split"] if report.exists() else None
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
        training = json.loads((attempt / "train.json").read_text())
        row |= {"attempt": attempt.name} | {key: training.get(key) for key in TRAINING}
        predicted = cohort / "predictions" / run["id"]
        record = json.loads((predicted / "prediction.json").read_text())
        if record["run"] != run or record["weights_sha256"] != training["weights_sha256"]:
            raise ValueError(f"Predictions belong to a different run or checkpoint: {run['id']}")
        source = json.dumps(record.get("source"), sort_keys=True).encode()
        row |= {"split": record["split"], "prediction_source_sha256": hashlib.sha256(source).hexdigest()}
        mapping = json.loads((predicted / "classes.json").read_text())
        # Flat heads store {key: index}; hierarchical heads {rank: {key: index}}.
        mapping = mapping if isinstance(next(iter(mapping.values())), dict) else {"0": mapping}
        for rank, classes_at_rank in mapping.items():
            if rank_classes.setdefault(rank, classes_at_rank) != classes_at_rank:
                raise ValueError(f"Class indices differ between runs at rank {rank}: {run['id']}")
        if mapping["0"] != classes["cls2idx"]:
            raise ValueError(f"Species indices differ from the prepared class order: {run['id']}")
        index = pq.read_table(predicted / "index.parquet").to_pandas()
        selected = samples[samples.split == record["split"]]
        # Labels are class indices (ablations) or species keys (Gefion data indexes).
        truth = selected.class_id if pd.api.types.is_integer_dtype(index.label_0) else selected.speciesKey
        # Stored paths are image IDs (ablations) or full paths ending in them (mt_predict).
        ids = selected.image_id.to_numpy().astype(str)
        paths = index.path.to_numpy().astype(str)
        same_images = len(paths) == len(ids) and all(p == i or p.endswith("/" + i) for p, i in zip(paths, ids))
        aligned = same_images and np.array_equal(index.label_0.astype(str), truth.astype(str))
        if not aligned or index.row.tolist() != list(range(len(index))):
            raise ValueError(f"Predictions are not aligned with the prepared split: {run['id']}")
        # rank-<r> holds float16 log-probabilities of each native output rank; embeddings the head input.
        for part in sorted(path for path in predicted.iterdir() if path.is_dir()):
            shards, expected = [], 0
            for shard in sorted(part.glob("part-*.parquet")):
                table = pq.read_table(shard)
                rows = table["row"].to_numpy()
                if not np.array_equal(rows, np.arange(expected, expected + len(rows))):
                    raise ValueError(f"Shard rows are not contiguous: {shard}")
                identity = [pa.array([name] * len(rows)).dictionary_encode(), pa.array([run["id"]] * len(rows)).dictionary_encode()]
                identity.append(pa.array(ids[rows]))
                values = table.drop_columns(["row"])
                shards.append(pa.table(identity + values.columns, names=["study", "run_id", "image_id", *values.column_names]))
                expected += len(rows)
            if expected != len(index):
                raise ValueError(f"{part.name} covers {expected} of {len(index)} images: {run['id']}")
            if part.name == "embeddings":
                write(shards, "embeddings", "embeddings", run["id"])
            else:
                write(shards, "scores", "scores", run["id"], rank=int(part.name.split("-")[1]))
        # Sparse per-epoch validation confusions recorded during training (AMP), one row per non-zero cell.
        for path in sorted((attempt / "model/logs/figures").glob("epoch-*/Confusion_matrix_lvl0/counts.npz")):
            with np.load(path, allow_pickle=False) as data:
                cells = {"true_class": data["rows"], "predicted_class": data["columns"], "count": data["counts"]}
            confusions.append(pd.DataFrame(cells).assign(study=name, run_id=run["id"], epoch=int(path.parents[1].name.split("-")[-1])))
        log = attempt / "model/logs/learning.jsonl"
        if log.exists():  # Raw per-epoch training/validation logs, one JSON record per line.
            learning.append(pd.read_json(log, lines=True).assign(study=name, run_id=run["id"]))
        analysis = cohort / "analysis" / run["id"]
        for path in sorted(analysis.glob("*.csv")):
            # Analyzer tables must describe the exported split.
            if analysis_split == record["split"]:
                aggregates.setdefault(path.stem, []).append(pd.read_csv(path).assign(study=name, run_id=run["id"]))
        if (analysis / "summary.json").exists() and analysis_split == record["split"]:
            metrics = json.loads((analysis / "summary.json").read_text())["metrics"]
            aggregates.setdefault("summary", []).append(pd.json_normalize(metrics).assign(study=name, run_id=run["id"]))
    if rank_classes:
        rank_names = classes.get("rank_names", RANK_NAMES)
        rows = [(name, int(r), rank_names[int(r)], index, key) for r, mapping in rank_classes.items() for key, index in mapping.items()]
        columns = ["study", "rank", "rank_name", "class_index", "key"]
        write(pd.DataFrame(rows, columns=columns).sort_values(["rank", "class_index"]), "taxonomy", "classes")
    if confusions:
        write(pd.concat(confusions, ignore_index=True), "curves", "epoch_confusion")
    if learning:
        write(pd.concat(learning, ignore_index=True), "curves", "learning")
    # Analyzer-specific tables are auxiliary: derivable from scores or specific to these ablations.
    for kind, frames in aggregates.items():
        write(pd.concat(frames, ignore_index=True), "aux", kind)
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
        fields |= {
            re.sub(r"^e\d+$", "e<dimension>", re.sub(r"^c\d+$", "c<class_id>", field.name)): str(field.type)
            for field in pq.read_schema(output / row.path)
        }
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
        "Scores are float16 log-probabilities of each native output rank (`rank` in `catalog.csv`, `c<class_index>`\n"
        "in `classes.json` order); flat heads output only species. Embeddings (`e<dimension>`) are the head input.\n"
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
