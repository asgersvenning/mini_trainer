import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from dev.benchmarks.datasets import prepare_real
from dev.benchmarks.summarize import summarize


def make_dataset(root):
    for label_id, label in enumerate(("a", "b")):
        for index in range(6):
            split = "train" if index < 5 else "test"
            path = root / split / label / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index + 10 * label_id, 0, 0)).save(path)


def test_real_split_is_reproducible_and_groups_duplicates(tmp_path):
    root = tmp_path / "images"
    make_dataset(root)
    shutil.copy(root / "train/a/0.png", root / "train/a/duplicate.png")
    shutil.copy(root / "test/b/5.png", root / "train/b/leaked.png")
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        output.mkdir()
    first, spec = prepare_real(root, outputs[0], name="mnist", seed=42)
    second, _ = prepare_real(root, outputs[1], name="mnist", seed=42)
    assert first == second
    assert spec["cls2idx"] == {"a": 0, "b": 1}
    assert first["excluded_train_test_duplicates"] == ["train/b/leaked.png"]
    records = first["records"]
    hashes = {split: {r["sha256"] for r in records if r["split"] == split} for split in ("train", "val", "test")}
    assert hashes["train"].isdisjoint(hashes["val"] | hashes["test"])
    assert hashes["val"].isdisjoint(hashes["test"])
    assert sum(r["split"] == "test" for r in records) == 2
    assert (root / "train/b/leaked.png").exists()  # Inventory filtering never modifies source data.


def test_blair_requires_explicit_covering_taxonomy(tmp_path):
    root = tmp_path / "images"
    make_dataset(root)
    with pytest.raises(ValueError, match="explicit reviewed"):
        prepare_real(root, tmp_path, name="blair", seed=42)
    spec = {
        "labels": {"a": ["species_a", "parent"], "b": ["species_b", "parent"]},
        "num_classes": [2, 1],
        "cls2idx": {"0": {"species_a": 0, "species_b": 1}, "1": {"parent": 0}},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    manifest, restored = prepare_real(root, tmp_path, name="blair", seed=42, class_spec=path)
    assert restored == spec
    assert {tuple(record["targets"]) for record in manifest["records"]} == {(0, 0), (1, 0)}
    del spec["labels"]["b"]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="exactly cover"):
        prepare_real(root, tmp_path, name="blair", seed=42, class_spec=path)


def test_summary_preserves_failures_and_unmeasured_fields(tmp_path):
    path = tmp_path / "failed"
    path.mkdir()
    (path / "report.json").write_text(json.dumps({"status": "failed", "device": "cuda:0", "dtype": "float16"}))
    summary = summarize(tmp_path)
    assert "| failed | failed | cuda:0 / float16 | — | — |" in summary
    assert "CPU results do not validate GPU" in summary
    assert "No reports produced" in summarize(Path(tmp_path / "missing"))
