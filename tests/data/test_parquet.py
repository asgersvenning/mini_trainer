import json

import pytest
import torch

from mini_trainer.integrations.parquet import KCOLUMNS, combine_dicts, get_keys, set2split


def test_set2split():
    assert set2split(0) == "test"
    assert set2split(1) == "validation"
    assert set2split(2) == "train"
    assert set2split(99) == "train"


def test_combine_dicts():
    d1 = {"a": 1, "b": 2}
    d2 = {"a": 3, "b": 4}
    combined = combine_dicts([d1, d2])
    assert combined["a"] == [1, 3]
    assert combined["b"] == [2, 4]


def test_get_keys_normalizes_ids_in_taxonomic_order():
    row = {k: f" 00{i} " for i, k in reversed(list(enumerate(KCOLUMNS)))}
    assert get_keys(row) == [str(i) for i in range(len(KCOLUMNS))]


def _write_gbifxdl(path, species):
    """Write a minimal gbifxdl Parquet: ``species`` maps (species, genus) to row counts per ``set``."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = []
    for (key, genus), per_set in species.items():
        for split, count in per_set.items():
            rows += [{"filename": f"{key}_{split}_{i}.jpg", "set": split, "speciesKey": key, "genusKey": genus} for i in range(count)]
    columns = {name: [str(row.get(name, 7)) for row in rows] for name in ("filename", "set", *KCOLUMNS)}
    pq.write_table(pa.table(columns), path)


def test_class_count_floor_counts_non_test_rows_and_drops_their_samples(tmp_path):
    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.integrations.parquet import parquet_to_class_spec_hierarchical

    path = tmp_path / "data.parquet"
    # Species 2 has many test rows but only four elsewhere; the floor ignores test rows.
    _write_gbifxdl(path, {(1, 100): {"2": 3, "1": 2}, (2, 100): {"3": 4, "0": 50}, (3, 200): {"9": 6}})
    full = parquet_to_class_spec_hierarchical(str(path))
    floored = parquet_to_class_spec_hierarchical(str(path), min_count=5)
    assert list(full["cls2idx"]["0"]) == ["1", "2", "3"] and full["num_classes"] == [3, 2, 1]
    assert floored["cls2idx"]["0"] == {"1": 0, "3": 1} and floored["num_classes"] == [2, 2, 1]
    assert list(floored["labels"]) == ["1", "3"]

    def loaders(drop):
        return HierarchicalBuilder.build_dataloader(
            input_dir=str(path),
            output_dir=str(tmp_path),
            cls2idx=floored["cls2idx"],
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            resize_size=8,
            num_workers=0,
            drop_unknown_classes=drop,
        )

    labels, train, val = loaders(True)
    assert len(train.dataset) == 9 and len(val.dataset) == 2
    assert sorted({tuple(c) for c in labels}) == [(0, 0, 0), (1, 1, 0)]
    # Without the option, rows of excluded species stay in, with unknown classes.
    labels, train, _ = loaders(False)
    assert len(train.dataset) == 13 and (None, 0, 0) in {tuple(c) for c in labels}


def test_non_numeric_keys_build_the_vocabulary_and_load_from_folders_with_colons(tmp_path):
    """Catalogue of Life relabelling: string keys, colons in staged folder names, an extra legacySpeciesKey column."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image

    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.integrations.parquet import parquet_to_class_spec_hierarchical

    assert get_keys(dict.fromkeys(KCOLUMNS, " H9MH ")) == ["H9MH"] * len(KCOLUMNS)
    taxa = {"H9MH": ("G1", "F1", "1001"), "gbif:123": ("gbif:45", "F1", "1002"), "none:genus:X": ("none:family:Y", "F2", "1003")}
    rows = [
        {"filename": f"{i}.jpg", "set": split, "speciesKey": key, "genusKey": genus, "familyKey": family, "legacySpeciesKey": legacy}
        for key, (genus, family, legacy) in taxa.items()
        for i, split in enumerate(("2", "2", "1"))
    ]
    for row in rows:
        folder = tmp_path / "images" / row["speciesKey"]
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (120, 80, 40)).save(folder / row["filename"])
    columns = {name: [row.get(name, "K") for row in rows] for name in ("filename", "set", *KCOLUMNS, "legacySpeciesKey")}
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table(columns), path)

    spec = parquet_to_class_spec_hierarchical(str(path))
    assert set(spec["cls2idx"]["0"]) == set(taxa) and spec["num_classes"] == [3, 3, 2]
    assert json.loads(json.dumps(spec["cls2idx"])) == spec["cls2idx"]

    labels, train, val = HierarchicalBuilder.build_dataloader(
        input_dir=str(path),
        output_dir=str(tmp_path),
        cls2idx=spec["cls2idx"],
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        resize_size=8,
        num_workers=0,
    )
    assert len(train.dataset) == 6 and len(val.dataset) == 3 and None not in {c for row in labels for c in row}
    assert next(iter(val))[0].shape[-2:] == (8, 8)
