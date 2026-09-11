"""Small image classification problem with an exact pixel-based oracle."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def oracle(images: np.ndarray) -> np.ndarray:
    """Classify uint8 NHWC images using the two signal channels, without labels."""
    means = images[..., :2].mean(axis=(1, 2))
    bits = means >= 128
    return bits[:, 0].astype(np.int64) * 2 + bits[:, 1]


def generate(directory: str | Path, *, seed: int = 42, train_per_class: int = 32, val_per_class: int = 8, test_per_class: int = 16):
    """Write independent balanced splits with bounded noise and a 100% oracle.

    Red and green encode two binary factors at intensities 32 or 224, with
    integer noise in [-16, 16]. Blue is nuisance noise. The parent label is the
    red factor. Every pixel stays on its factor's side of the 128 threshold.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    counts = (train_per_class, val_per_class, test_per_class)
    if seed < 0 or any(count < 1 for count in counts):
        raise ValueError("Use a nonnegative seed and positive split sizes.")
    records = []
    for split_id, (split, count) in enumerate(zip(("train", "val", "test"), counts, strict=True)):
        for label in range(4):
            for index in range(count):
                rng = np.random.default_rng(np.random.SeedSequence([seed, split_id, label, index]))
                pixels = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
                center = np.array([32 + 192 * (label // 2), 32 + 192 * (label % 2)])
                pixels[..., :2] = center + rng.integers(-16, 17, size=(8, 8, 2))
                relative = Path(split) / f"class_{label}" / f"{index:05d}.png"
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(target)
                records.append(
                    {
                        "path": relative.as_posix(),
                        "split": split,
                        "label": label,
                        "parent": label // 2,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
    manifest = {
        "schema_version": 1,
        "generator": "bounded_color_bits_v1",
        "seed": seed,
        "oracle_accuracy": 1.0,
        "chance_accuracy": 0.25,
        "records": records,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
