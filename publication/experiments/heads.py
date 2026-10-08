"""Export classification-head weights as tables, in the layout planned for evidence snapshots.

One row per output class and rank: ``study, run_id, rank, class_index, key, bias, w0..w{D-1}``.
Weights are the effective ones the forward pass uses (weight normalization applied). For normalized
heads, logits are ``cosine_to_zscore(normalize(e) @ w) + bias``, with ``e`` the evidence embeddings.

    python -m publication.experiments.heads fetch OUTPUT STUDY/RUN_ID [...] [--archive NAME]
    python -m publication.experiments.heads export OUTPUT COHORT [...] --revision COMMIT

``fetch`` downloads archived checkpoints for a quick look; ``export`` writes an evidence-format snapshot
(``heads/<study>/<run_id>.parquet``, ``catalog.csv``, ``schemas.json``, ``manifest.json``) for every
complete run of each cohort directory (``<cohort>/study``), checking each checkpoint against its training record.
"""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

REGISTRY = Path(__file__).with_name("erda-snapshots.json")


def head_table(checkpoint):
    """Every output layer of a mini_trainer checkpoint's head, one row per class and rank."""
    import torch

    from mini_trainer.modeling import Classifier
    from mini_trainer.utils import import_class

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = Classifier.extract_metadata(state)
    model, _ = import_class(metadata["classifier_class"]).build(weights=state)
    head = getattr(model, metadata["backbone_output_name"]).eval()
    # Flat dict for flat heads; {rank: {key: index}} for hierarchical ones.
    vocabularies = metadata["cls2idx"] if isinstance(next(iter(metadata["cls2idx"].values())), dict) else {"0": metadata["cls2idx"]}
    frames = []
    with torch.no_grad():
        for rank in range(len(getattr(head, "layers", [None]))):  # Conditional/independent heads have a layer per rank
            weight, bias = head._weight_bias(rank)
            keys = {index: key for key, index in vocabularies[str(rank)].items()}
            columns = {"rank": rank, "class_index": range(len(weight)), "key": [str(keys[i]) for i in range(len(weight))]}
            columns["bias"] = bias.float().numpy() if bias is not None else 0.0
            frames.append(pd.concat([pd.DataFrame(columns), pd.DataFrame(weight.float().numpy()).add_prefix("w")], axis=1))
    return pd.concat(frames, ignore_index=True)


def archive_url(name=None):
    """The newest registered checkpoint archive (or the named one) as a read URL."""
    registry = json.loads(REGISTRY.read_text())
    names = [s["name"] for s in registry["snapshots"] if s["kind"] == "final checkpoint archive"]
    return f"{registry['read_base_url']}/{name or names[-1]}"


def write(table, output, study, run_id):
    target = output / "heads" / study / f"{run_id}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.insert(0, "run_id", run_id)
    table.insert(0, "study", study)
    table.to_parquet(target, index=False)
    print(f"{study}/{run_id}: {table.groupby('rank').size().to_dict()} classes per rank", flush=True)
    return target.relative_to(output).as_posix(), len(table)


def fetch(output, runs, archive=None):
    base = archive_url(archive)
    catalog = pd.read_csv(f"{base}/catalog.csv").set_index(["study", "run_id"])
    for run in runs:
        study, run_id = run.split("/", 1)
        row = catalog.loc[(study, run_id)]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            urllib.request.urlretrieve(f"{base}/{row.path}", checkpoint)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row.sha256:
                raise ValueError(f"Downloaded checkpoint differs from the archive catalog: {run}")
            write(head_table(checkpoint), output, study, run_id)


def export(output, cohorts, revision):
    from .artifacts import create
    from .evidence import SCHEMA_VERSION, dataset_name
    from .training_ablations.data import digest

    output.mkdir(parents=True, exist_ok=False)
    catalog = []
    for cohort in map(Path, cohorts):
        study, root = cohort.name, cohort / "study"
        dataset = dataset_name(json.loads((root / "config.json").read_text()))
        for run in json.loads((root / "plan.json").read_text()):
            attempts = sorted((root / "runs" / run["id"]).glob("attempt-*"))
            if not attempts or not (attempts[-1] / "complete.json").exists():
                raise ValueError(f"Run is not complete: {study}/{run['id']}")
            training = json.loads((attempts[-1] / "train.json").read_text())
            # Gefion attempts record their checkpoint's location; ablation attempts keep it inside the attempt.
            checkpoint = Path(training.get("weights", attempts[-1] / "model/weights/last.pt"))
            if digest(checkpoint) != training["weights_sha256"]:
                raise ValueError(f"Checkpoint differs from its training record: {study}/{run['id']}")
            path, rows = write(head_table(checkpoint), output, study, run["id"])
            catalog.append(
                {"path": path, "set": "heads", "kind": "heads", "schema_version": SCHEMA_VERSION, "study": study}
                | {"dataset": dataset, "run_id": run["id"], "rank": None, "rows": rows, "caveat": ""}
            )
    pd.DataFrame(catalog).to_csv(output / "catalog.csv", index=False)
    example = pd.read_parquet(output / catalog[0]["path"])
    schema = {
        ("w<dimension>" if column[1:].isdigit() and column[0] == "w" else column): str(dtype) for column, dtype in example.dtypes.items()
    }
    (output / "schemas.json").write_text(json.dumps({"heads": schema}, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Head weights\n\nOne table per run: `study, run_id, rank, class_index, key, bias, w<dimension>` (float32), the\n"
        "effective final-layer weights each head uses. Join on `(study, rank, class_index)` with the evidence\n"
        "`classes.parquet`. Normalized heads give logits `sqrt(D - 2) * (acos(-cos(e, w)) - pi / 2) + bias`.\n"
    )
    names = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    (output / "manifest.json").write_text(json.dumps(create(output, names, revision), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("fetch", help="Archived checkpoints by STUDY/RUN_ID")
    one.add_argument("output", type=Path)
    one.add_argument("runs", nargs="+", help="STUDY/RUN_ID, as in the checkpoint archive's catalog.csv")
    one.add_argument("--archive", help="Checkpoint archive name; default: newest registered")
    many = commands.add_parser("export", help="Evidence-format snapshot of every run in the cohorts")
    many.add_argument("output", type=Path, help="New snapshot directory")
    many.add_argument("cohorts", nargs="+", type=Path, help="Cohort directories containing study/")
    many.add_argument("--revision", required=True, help="Exporting source revision, recorded in the manifest")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.output, args.runs, args.archive)
    else:
        export(args.output, args.cohorts, args.revision)


if __name__ == "__main__":
    main()
