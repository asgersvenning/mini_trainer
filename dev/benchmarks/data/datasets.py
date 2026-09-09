"""Read-only real-data inventories and reproducible stratified validation splits."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mini_trainer.data import find_images


def prepare_real(root: Path, output: Path, *, name: str, seed: int, class_spec: Path | None = None):
    names = sorted(path.name for path in (root / "train").iterdir() if path.is_dir())
    if not names:
        raise ValueError("Training directory contains no classes.")
    if name == "blair":
        if class_spec is None:
            raise ValueError("Blair requires an explicit reviewed --class-spec; no online taxonomy lookup is performed.")
        spec = json.loads(class_spec.read_text())
        if set(spec["labels"]) != set(names):
            raise ValueError("Blair class specification must exactly cover training image classes.")
        targets = {
            label: [spec["cls2idx"][str(level)][value] for level, value in enumerate(values)] for label, values in spec["labels"].items()
        }
    else:
        spec = {"num_classes": len(names), "cls2idx": {name: i for i, name in enumerate(names)}}
        targets = {name: [i] for i, name in enumerate(names)}
    records = []
    for split in ("train", "test"):
        for path_string in find_images(str(root / split)):
            path = Path(path_string)
            label = path.parent.name
            if label not in targets:
                raise ValueError(f"Unknown class in {split}: {label}")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "split": split,
                    "label": targets[label][0],
                    "targets": targets[label],
                    "sha256": digest,
                }
            )
    test_hashes = {record["sha256"] for record in records if record["split"] == "test"}
    if not test_hashes:
        raise ValueError("A held-out test directory is required.")
    overlap = [record for record in records if record["split"] == "train" and record["sha256"] in test_hashes]
    # Keep the supplied test set intact; remove byte-identical training counterparts.
    records = [record for record in records if record["split"] == "test" or record["sha256"] not in test_hashes]
    groups = defaultdict(dict)
    for record in records:
        if record["split"] == "train":
            groups[record["label"]].setdefault(record["sha256"], []).append(record)
    for label, grouped in sorted(groups.items()):
        hashes = sorted(grouped)
        if len(hashes) < 2:
            raise ValueError(f"Class {label} needs at least two unique training images.")
        rng = np.random.default_rng(np.random.SeedSequence([seed, label]))
        rng.shuffle(hashes)
        for digest in hashes[: max(1, len(hashes) // 5)]:
            for record in grouped[digest]:
                record["split"] = "val"
    if set(groups) != {target[0] for target in targets.values()}:
        raise ValueError("Some classes have no training images after duplicate removal.")
    split_hashes = {split: {r["sha256"] for r in records if r["split"] == split} for split in ("train", "val", "test")}
    if any(split_hashes[a] & split_hashes[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Cross-split duplicate images detected; review conflicting class labels.")
    manifest = {
        "schema_version": 1,
        "dataset": name,
        "seed": seed,
        "split_policy": "20 percent of per-class unique training files for validation; official test preserved",
        "excluded_train_test_duplicates": [record["path"] for record in overlap],
        "class_spec": spec,
        "records": records,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, spec
