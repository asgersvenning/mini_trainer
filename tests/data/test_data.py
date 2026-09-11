import csv
import json
import os
from collections import OrderedDict

import pytest
import torch

from mini_trainer.data import find_images, get_metadata, parse_class_spec
from mini_trainer.data import metadata as metadata_module
from mini_trainer.hierarchical.integration import HierarchicalResultCollector


@pytest.mark.parametrize("mapping_type", [dict, OrderedDict, list])
def test_inference_folders_include_unseen_species(tmp_path, monkeypatch, mapping_type):
    for species in ("111", "999"):
        folder = tmp_path / species
        folder.mkdir()
        (folder / "image.jpg").write_bytes(b"\xff\xd8\xff")
    known = ("111", "222", "333")
    labels = [known] if mapping_type is list else mapping_type({"111": known})
    cls2idx = {"0": {"111": 0}, "1": {"222": 0}, "2": {"333": 0}}
    calls = []

    def taxonomy(ids, levels):
        calls.append((ids, levels))
        return OrderedDict({"999": OrderedDict(species=("998", "synonym"), genus=("888", "genus"), family=("333", "family"))})

    monkeypatch.setattr(metadata_module, "create_taxonomy", taxonomy)
    ground_truth, paths = metadata_module.auto_find_images(str(tmp_path), cls2idx=cls2idx, labels=labels)
    actual = {os.path.basename(os.path.dirname(path)): tuple(label) for path, label in zip(paths, ground_truth, strict=True)}
    assert actual == {"111": known, "999": ("999", "888", "333")}
    assert calls == [(["999"], 3)]
    collector = HierarchicalResultCollector(cls2idx=cls2idx, scientific_names=False)
    collector.collect(paths=paths, predictions=[torch.zeros(2, 1) for _ in range(3)], labels=ground_truth)
    collector.save(str(tmp_path))
    with (tmp_path / "mini_metric.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    unseen = [row for row in rows if os.path.basename(os.path.dirname(row["filename"])) == "999"]
    assert [row["label"] for row in unseen] == ["999", "888", "333"]
    assert [row["known_label"] for row in unseen] == ["0", "0", "1"]


def test_inference_folder_taxonomy_failure_is_explicit(tmp_path, monkeypatch):
    folder = tmp_path / "999"
    folder.mkdir()
    (folder / "image.jpg").write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr(metadata_module, "create_taxonomy", lambda *args, **kwargs: OrderedDict())
    with pytest.raises(ValueError, match="999"):
        metadata_module.auto_find_images(str(tmp_path), cls2idx={"0": {"111": 0}, "1": {"222": 0}}, labels={})


def test_find_images(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    (d / "img1.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "img2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (d / "not_img.txt").write_text("hello")

    images = find_images(str(d))
    assert len(images) == 2
    basenames = sorted([os.path.basename(p) for p in images])
    assert basenames == ["img1.jpg", "img2.png"]


def test_get_metadata(tmp_path):
    p = tmp_path / "meta.json"
    data = {"path": ["a", "b"], "class": [0, 1], "split": ["train", "validation"]}
    with open(p, "w") as f:
        json.dump(data, f)

    meta = get_metadata(str(p))
    # It converts to numpy arrays
    assert len(meta["path"]) == 2
    assert meta["split"][0] == "train"

    with pytest.raises(FileNotFoundError):
        get_metadata(str(tmp_path / "nonexistent.json"))


def test_parse_class_spec(tmp_path):
    d = tmp_path / "classes"
    d.mkdir()
    (d / "cat").mkdir()
    (d / "dog").mkdir()
    (d / "file.txt").touch()

    spec = parse_class_spec(dir=str(d))
    assert spec["num_classes"] == 2
    assert spec["cls2idx"] == {"cat": 0, "dog": 1}

    # Save/Load
    p = tmp_path / "spec.json"
    parse_class_spec(path=str(p), dir=str(d))
    loaded = parse_class_spec(path=str(p))
    assert loaded == spec
