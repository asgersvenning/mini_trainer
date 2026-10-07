"""Export classification-head weights as tables, in the layout planned for evidence snapshots.

One row per output class and rank: ``study, run_id, rank, class_index, key, bias, w0..w{D-1}``.
Weights are the effective ones the forward pass uses (weight normalization applied). For normalized
heads, logits are ``cosine_to_zscore(normalize(e) @ w) + bias``, with ``e`` the evidence embeddings.

    python -m publication.experiments.heads OUTPUT STUDY/RUN_ID [STUDY/RUN_ID ...] [--archive NAME]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="Directory; writes heads/<study>/<run_id>.parquet")
    parser.add_argument("runs", nargs="+", help="STUDY/RUN_ID, as in the checkpoint archive's catalog.csv")
    parser.add_argument("--archive", help="Checkpoint archive name; default: newest registered")
    args = parser.parse_args()
    base = archive_url(args.archive)
    catalog = pd.read_csv(f"{base}/catalog.csv").set_index(["study", "run_id"])
    for run in args.runs:
        study, run_id = run.split("/", 1)
        row = catalog.loc[(study, run_id)]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            urllib.request.urlretrieve(f"{base}/{row.path}", checkpoint)
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row.sha256:
                raise ValueError(f"Downloaded checkpoint differs from the archive catalog: {run}")
            table = head_table(checkpoint)
        target = args.output / "heads" / study / f"{run_id}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.insert(0, "run_id", run_id)
        table.insert(0, "study", study)
        table.to_parquet(target, index=False)
        print(f"{run}: {table.groupby('rank').size().to_dict()} classes per rank -> {target}")


if __name__ == "__main__":
    main()
