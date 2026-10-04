"""Inference-only embedding diagnostics on a fixed, class-balanced validation sample."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import frequency_groups, verified
from .data import digest, write_json


def select_samples(frame, per_class=32, seed=20261004):
    if per_class < 1:
        raise ValueError("Positive per-class sample limit required")
    rng = np.random.default_rng(seed)
    indices = []
    for _, group in frame.sort_values("sample_id").groupby("label", sort=True):
        indices.extend(rng.choice(group.index, min(per_class, len(group)), replace=False))
    return frame.loc[indices].sort_values(["label", "sample_id"]).reset_index(drop=True)


def embedding_geometry(values, target, counts):
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms == 0).any():
        raise ValueError("Embedding directions must be finite and nonzero")
    values = values / norms
    labels = np.unique(target)
    means = np.stack([values[target == c].mean(0) for c in labels])
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero class centroid has no angular direction")
    centers = means / norms
    cosine = np.clip(centers @ centers.T, -1, 1)
    np.fill_diagonal(cosine, -np.inf)
    nearest = cosine.argmax(1)
    groups = frequency_groups(counts)
    rows = []
    for i, c in enumerate(labels):
        angles = np.degrees(np.arccos(np.clip(values[target == c] @ centers[i], -1, 1)))
        rows.append(
            {
                "class_index": int(c),
                "samples": int((target == c).sum()),
                "frequency_group": ("tail", "mid", "head")[groups[c]],
                "mean_within_class_angle": float(angles.mean()),
                "resultant_norm": float(norms[i, 0]),
                "nearest_class": int(labels[nearest[i]]),
                "nearest_centroid_angle": float(np.degrees(np.arccos(cosine[i, nearest[i]]))) if len(labels) > 1 else None,
            }
        )
    return pd.DataFrame(rows)


def extract(root, output, variants=None, device="cpu", per_class=32, seed=20261004):
    import torch

    from mini_trainer.data.loader import get_dataset_dataloader
    from mini_trainer.hierarchical.model import HierarchicalClassifier
    from mini_trainer.modeling import Classifier, classification_module

    output.mkdir(parents=True, exist_ok=False)
    prepared = json.loads((root / "prepared.json").read_text())
    for name in ["classes.json", "samples.parquet", "config.json"]:
        verified(root / name, prepared["files"][name])
    config = json.loads((root / "config.json").read_text())
    spec = json.loads((root / "classes.json").read_text())
    frame = select_samples(pd.read_parquet(root / "samples.parquet", filters=[("split", "=", "validation")]), per_class, seed)
    frame.to_csv(output / "samples.csv", index=False)
    provenance = {
        "prepared_sha256": digest(root / "prepared.json"),
        "source_sha256": digest(Path(__file__)),
        "sample_sha256": digest(output / "samples.csv"),
        "seed": seed,
        "per_class": per_class,
        "runs": {},
    }
    for directory in sorted((root / "runs").iterdir()):
        attempts = sorted(directory.glob("attempt-*"))
        if not attempts or not (attempts[-1] / "complete.json").exists():
            continue
        attempt = attempts[-1]
        manifest = json.loads((attempt / "complete.json").read_text())
        run = json.loads(verified(attempt / "run.json", manifest["run.json"]).read_text())
        if not run.get("variant") or (variants and run["variant"] not in variants):
            continue
        weights = verified(attempt / "model/weights/last.pt", manifest["model/weights/last.pt"])
        head_cls = HierarchicalClassifier if run.get("rank_weights") is not None else Classifier
        model, preprocess = head_cls.build(weights=str(weights), device=device, model_args={"pretrained": False}, skip_spherical_init=True)
        model.eval()
        values = []
        head = classification_module(model)

        def observe(module, inputs):
            values.append(module.preclassification(inputs[0]).detach().float().cpu())

        handle = head.register_forward_pre_hook(observe)
        _, loaders = get_dataset_dataloader(
            {"path": [str(Path(config["images"]) / p) for p in frame.sample_id], "class": frame.label.tolist()},
            resize_size=config["size"],
            modes=("validation",),
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            device=device,
            cache=None,
        )
        try:
            with torch.inference_mode():
                for images, _ in loaders[0]:
                    model(preprocess(images.to(device)))
        finally:
            handle.remove()
        embeddings = torch.cat(values).numpy()
        embedding_geometry(embeddings, frame.label.to_numpy(), spec["counts"]).to_csv(output / f"{directory.name}.csv", index=False)
        provenance["runs"][directory.name] = {"weights_sha256": digest(weights), "run_sha256": digest(attempt / "run.json")}
        del model, head
    write_json(output / "provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    extract(args.root, args.output, args.variants, args.device)


if __name__ == "__main__":
    main()
