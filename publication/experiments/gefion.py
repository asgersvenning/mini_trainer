"""Adapt the Gefion backbone x head runs to evidence-study cohorts and predict their test sets.

``prepare`` writes ``COHORT/study`` in the layout ``publication.experiments.evidence`` reads,
from each run's recorded config, class specification and checkpoint, plus a test-only data
index with paths on the current machine. ``predict`` reruns each run's recorded Gefion predict
configuration through ``ParquetResultCollector`` into ``COHORT/predictions/<run_id>``.
Provenance comes from these configs; unknown values are recorded as null.
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from .training_ablations.data import digest, write_json

RANK_KEYS = ("speciesKey", "genusKey", "familyKey", "orderKey", "classKey")
RANK_NAMES = ("species", "genus", "family", "order", "class")
DATASET_ROOTS = {"global_lepi": "/dcai/projects/iu_0126/datasets/global_lepi/", "flemming": "/dcai/projects/iu_0126/datasets/flemming/"}


def run_factors(run_dir, weights):
    """Settings that vary between Gefion runs, read from the run's training config."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    model, data = config["model_builder_kwargs"], config["dataloader_builder_kwargs"]
    optimizer = config["optimizer_builder_kwargs"]
    head = run_dir.name.split("_")[0]
    commits = set()
    for metadata in (run_dir.parents[2] / "wandb").glob("run-*/files/wandb-metadata.json"):
        record = json.loads(metadata.read_text())
        if any(run_dir.name in str(argument) for argument in record.get("args", [])):
            commits.add(record.get("git", {}).get("commit"))
    return {
        "id": f"{model['model_type']}_{head}",
        "backbone": model["model_type"],
        "head": head,
        "normalized": model.get("normalized"),
        "hidden": model.get("hidden"),
        "droprate": model.get("droprate"),
        "loss": "emla" if config.get("criterion_builder_kwargs", {}).get("weighted") else "ce",
        "regularization": config.get("regularizer_builder_kwargs", {}).get("strength"),
        "optimizer": optimizer["optimizer_cls"].rsplit(".", 1)[-1],
        "lr": optimizer["lr"],
        "weight_decay": optimizer["weight_decay"],
        "warmup_epochs": config.get("lr_schedule_builder_kwargs", {}).get("warmup_epochs"),
        "epochs": config["epochs"],
        "batch_size": data["batch_size"],
        "dtype": config.get("dtype"),
        "training_commit": commits.pop() if len(commits) == 1 else None,
        "weights": str(weights),
    }


def test_samples(evaluation, index, taxonomy, flemming_labels=None, parquet=None):
    """Test images with species class index (-1 when outside the vocabulary) and rank keys."""
    test = [i for i, split in enumerate(index["split"]) if split == "test"]
    if evaluation == "flemming":  # Truth, including out-of-vocabulary species, from Gefion's predictions.
        labels = flemming_labels.assign(sample_id=flemming_labels.filename.str.split("/").str[-2:].str.join("/"))
        wide = labels.pivot(index="sample_id", columns="level", values="label").astype(str)
        order = labels.drop_duplicates("instance_id").sample_id
        frame = pd.DataFrame({"sample_id": order.to_numpy()})
        keys = wide.loc[frame.sample_id].to_numpy().tolist()
    elif evaluation == "plantnet":
        frame = pd.DataFrame({"sample_id": [index["path"][i] for i in test]})
        keys = [index["label"][i] for i in test]
    else:
        frame = pd.DataFrame({"sample_id": [index["path"][i].removeprefix(DATASET_ROOTS[evaluation]) for i in test]})
        keys = [taxonomy[index["label"][i]] for i in test]
    frame["path"] = f"/work/{evaluation}/" + frame.sample_id
    width = max(map(len, keys))
    for rank in range(width):
        frame[RANK_KEYS[rank]] = [str(k[rank]) for k in keys]
    if parquet is not None:  # Global Lepidoptera observation IDs, joined on species directory and file name.
        source = pd.read_parquet(parquet, columns=["speciesKey", "filename", "gbifID"]).astype(str)
        image = frame.sample_id.str.removeprefix("images/").str.split("/", n=1, expand=True)
        joined = pd.DataFrame({"speciesKey": image[0], "filename": image[1]}).merge(source, how="left", validate="many_to_one")
        frame["gbifID"] = joined.gbifID.to_numpy()
    return frame.assign(split="test")


def prepare(cohort, evaluation, runs, weights_root, index, flemming_labels=None, parquet=None):
    cohort, study = Path(cohort), Path(cohort) / "study"
    study.mkdir(parents=True, exist_ok=False)
    run_dirs = [Path(run) for run in runs]
    specs = [json.loads((r / "class_spec.json").read_text()) for r in run_dirs]
    spec = specs[0]
    hierarchical = next((s for s in specs if "labels" in s), {})
    species = spec["cls2idx"]["0"] if isinstance(spec["cls2idx"].get("0"), dict) else spec["cls2idx"]
    taxonomy = {key: [str(k) for k in path] for key, path in hierarchical.get("labels", {}).items()}
    index = json.loads(Path(index).read_text())
    if not taxonomy:  # PlantNet: rank keys per species from the index labels.
        taxonomy = {label[0]: label for label in index["label"]}
    train = pd.Series(
        [label[0] if isinstance(label, list) else label for label, split in zip(index["label"], index["split"]) if split == "train"]
    )
    counts = train.value_counts().reindex(sorted(species, key=species.get), fill_value=0)
    samples = test_samples(evaluation, index, taxonomy, flemming_labels, parquet)
    samples["label"] = samples.speciesKey.map(species).fillna(-1).astype(int)
    ranks = [key for key in RANK_KEYS if key in samples]
    write_json(
        study / "classes.json",
        {
            "cls2idx": species,
            "counts": counts.astype(int).tolist(),
            "rank_names": list(RANK_NAMES[: len(ranks)]),
            "taxonomy": {key: taxonomy[key] for key in sorted(species, key=species.get)},
        },
    )
    samples.drop(columns="path").to_parquet(study / "samples.parquet", index=False)
    write_json(
        study / "data_index.json",
        {
            "path": samples.path.tolist(),
            "split": samples.split.tolist(),
            "label": samples.speciesKey.tolist(),
            "class": samples.label.tolist(),
        },
    )
    plan = []
    for run_dir in run_dirs:
        weights = Path(weights_root) / run_dir.relative_to(run_dir.parents[3]) / "weights" / "last.pt"
        run = run_factors(run_dir, weights)
        plan.append({k: v for k, v in run.items() if k != "weights"})
        attempt = study / "runs" / run["id"] / "attempt-000"
        (attempt / "model/logs").mkdir(parents=True)
        write_json(attempt / "run.json", plan[-1])
        write_json(attempt / "train.json", {"weights_sha256": digest(weights), "weights": str(weights)})
        write_json(attempt / "complete.json", {})
        write_json(attempt / "predict.json", {"source_run": str(run_dir), "evaluation": evaluation})
        summary = run_dir / "logs" / "summary.csv"
        if summary.exists():  # Per-epoch training/evaluation records, as the ablations' learning.jsonl.
            log = pd.read_csv(summary).rename(columns={"type": "phase"})
            log.to_json(attempt / "model/logs/learning.jsonl", orient="records", lines=True)
    write_json(study / "plan.json", plan)
    write_json(study / "config.json", {"dataset": evaluation, "trained_on": run_dirs[0].name.rsplit("_", 1)[-1], "source": "gefion"})
    commits = {run["training_commit"] for run in plan} - {None}
    write_json(
        study / "prepared.json",
        {
            "files": {name: digest(study / name) for name in ("classes.json", "samples.parquet")},
            "git_commit": commits.pop() if len(commits) == 1 else None,
            "pretrained_enum": None,
        },
    )


def predict(cohort, batch_size=256, workers=32, device="cuda"):
    """Rerun each run's recorded Gefion predict config into the streaming collector."""
    cohort = Path(cohort)
    study = cohort / "study"
    for run in json.loads((study / "plan.json").read_text()):
        output = cohort / "predictions" / run["id"]
        if (output / "prediction.json").exists():
            continue
        attempt = study / "runs" / run["id"] / "attempt-000"
        record = json.loads((attempt / "predict.json").read_text())
        recorded = sorted((Path(record["source_run"]) / "predict").glob("*/config.yaml"))[0]
        config = yaml.safe_load(recorded.read_text())
        training = json.loads((attempt / "train.json").read_text())
        if digest(Path(training["weights"])) != training["weights_sha256"]:
            raise ValueError(f"Checkpoint changed: {run['id']}")
        config |= {
            "input": str(study),
            "weights": training["weights"],
            "data_index": str(study / "data_index.json"),
            "output": str(cohort / "predictions"),
            "name": run["id"],
            "device": device,
            "collector_cls": "mini_trainer.logging.collector.ParquetResultCollector",
            "collector_cls_kwargs": {},
            "dataloader_builder_kwargs": {"batch_size": batch_size, "num_workers": workers},
        }
        path = cohort / "predict-configs" / f"{run['id']}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config))
        # Explicit flags: mt_predict's CLI defaults for --output and --name override config values.
        command = ["mt_hpredict", "--config", str(path), "--head", run["head"], "--output", config["output"], "--name", run["id"]]
        subprocess.run(command, check=True)
        source = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
        identity = {"run": run, "split": "test", "attempt": attempt.name, "weights_sha256": training["weights_sha256"]}
        write_json(output / "prediction.json", identity | {"source": {"git_commit": source, "predict_config": config}})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("cohort", type=Path)
    make.add_argument("--evaluation", choices=["global_lepi", "plantnet", "flemming"], required=True)
    make.add_argument("--runs", type=Path, nargs="+", required=True, help="Gefion run directories (config.yaml, class_spec.json)")
    make.add_argument("--weights-root", type=Path, required=True, help="Root mirroring <campaign>/results/runs/<run>/weights")
    make.add_argument("--index", type=Path, required=True, help="Gefion data_index.json of the training dataset")
    make.add_argument("--flemming-labels", type=Path, help="Flemming mini_metric.csv of a three-rank run")
    make.add_argument("--parquet", type=Path, help="Global Lepidoptera parquet with gbifID")
    run = commands.add_parser("predict")
    run.add_argument("cohort", type=Path)
    run.add_argument("--device", default="cuda")
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.command == "prepare":
        labels = pd.read_csv(args.flemming_labels, dtype={"label": str}) if args.flemming_labels else None
        prepare(args.cohort, args.evaluation, args.runs, args.weights_root, args.index, labels, args.parquet)
    else:
        predict(args.cohort, args.batch_size, args.workers, args.device)


if __name__ == "__main__":
    main()
