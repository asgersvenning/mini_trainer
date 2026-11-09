import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from mini_trainer.builders import BaseBuilder
from mini_trainer.data import ShardDataset, ShardError, ShardWriter, get_shard_dataloaders, is_shard_root, write_manifest
from mini_trainer.data.shards import epoch_plan, verify

pytest.importorskip("pyarrow")
COLUMNS = ["speciesKey", "genusKey", "familyKey"]


def jpeg(value):
    buffer = io.BytesIO()
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def build(root, counts={"train": 53, "val": 11}, per_shard=9):
    """A shard set whose image brightness and labels identify each sample: species = value of the sample's id."""
    splits = {}
    next_id = 0
    for split, n in counts.items():
        writer = ShardWriter(root, split, COLUMNS, max_bytes=per_shard * 700)
        for i in range(n):
            writer.add(
                f"s{next_id:05d}",
                jpeg(next_id % 256),
                {"speciesKey": str(next_id), "genusKey": str(next_id % 7), "familyKey": str(next_id % 7 % 3)},
            )
            next_id += 1
        splits[split] = writer.close()
    write_manifest(root, splits, COLUMNS, {"test": True})
    return root, splits


def spec(n=100):
    return {"0": {str(i): i for i in range(n)}, "1": {str(i): i for i in range(7)}, "2": {str(i): i for i in range(3)}}


def species(dataset):
    return sorted(int(label[0]) for _, label in dataset)


def dataset(root, split="train", batch_size=1, train=True, **kwargs):
    return ShardDataset(root, split, lambda data: torch.tensor([data[0]]), spec(), batch_size, train=train, multilabel=True, **kwargs)


def plan_ids(counts, **kwargs):
    starts = np.cumsum([0, *counts])
    return [int(starts[shard]) + i for shard, first, stop in epoch_plan(counts, **kwargs) for i in range(first, stop)]


@pytest.mark.parametrize(("world", "workers", "batch"), [(1, 1, 1), (2, 3, 4), (3, 2, 5), (8, 4, 2)])
def test_plan_reads_each_sample_at_most_once_and_equal_batches_per_rank(world, workers, batch):
    counts = [31, 29, 33, 30, 28, 32, 30]
    common = dict(epoch=3, seed=1, shuffle=True, world=world, num_workers=workers, batch_size=batch, drop_last=True)
    per_rank = [[plan_ids(counts, rank=r, worker=w, **common) for w in range(workers)] for r in range(world)]
    flat = [i for rank in per_rank for worker in rank for i in worker]
    assert len(flat) == len(set(flat))
    usable = sum(counts) // world // batch * batch
    assert all(sum(len(worker) for worker in rank) == usable for rank in per_rank)
    assert all(len(worker) % batch == 0 for rank in per_rank for worker in rank)


def test_evaluation_plan_covers_every_sample_exactly_once():
    counts = [7, 9, 8]
    common = dict(epoch=0, seed=0, shuffle=False, world=3, num_workers=2, batch_size=4, drop_last=False)
    ids = [i for r in range(3) for w in range(2) for i in plan_ids(counts, rank=r, worker=w, **common)]
    assert sorted(ids) == list(range(24))


def test_epoch_plans_are_deterministic_and_differ_between_epochs():
    counts = [10] * 20
    common = dict(seed=5, shuffle=True, rank=0, world=2, worker=0, num_workers=1, batch_size=1, drop_last=True)
    assert plan_ids(counts, epoch=4, **common) == plan_ids(counts, epoch=4, **common)
    assert plan_ids(counts, epoch=4, **common) != plan_ids(counts, epoch=5, **common)


def test_stream_yields_every_training_sample_with_its_label_and_resumes_the_same_epoch(tmp_path):
    root, splits = build(tmp_path)
    assert is_shard_root(root) and len(splits["train"]["shards"]) > 1 and verify(root, full=True) == 64
    data = dataset(root, shuffle_buffer=8)
    assert len(data) == 53 and species(data) == list(range(53))
    identified = ShardDataset(root, "train", lambda encoded: torch.tensor(list(encoded)), spec(), 1, train=True, multilabel=True)
    for encoded, label in identified:
        i = int(label[0])
        assert bytes(encoded.tolist()) == jpeg(i % 256) and label.tolist() == [i, i % 7, i % 7 % 3]

    data.set_epoch(2)
    first = [int(y[0]) for _, y in data]
    resumed = dataset(root, shuffle_buffer=8)
    resumed.set_epoch(2)
    assert [int(y[0]) for _, y in resumed] == first
    resumed.set_epoch(3)
    assert [int(y[0]) for _, y in resumed] != first


def test_loaders_decode_real_images_across_workers_with_exact_batch_counts(tmp_path):
    root, _ = build(tmp_path)
    train, val = get_shard_dataloaders(str(root), spec(), 8, batch_size=4, num_workers=2, multilabel=True, shard_shuffle_buffer=4)
    assert isinstance(train, DataLoader) and len(train) == 13 and len(val) == 3
    labels = [lab for images, targets in train for lab in targets[:, 0].tolist()]
    assert len(labels) == 52 == len(set(labels))
    images, targets = next(iter(val))
    assert images.dtype == torch.uint8 and images.shape == (4, 3, 8, 8) and targets.shape == (4, 3)
    assert sorted(lab for _, y in val for lab in y[:, 0].tolist()) == list(range(53, 64))


def test_builder_reads_a_shard_root_and_rejects_row_level_options(tmp_path):
    root, _ = build(tmp_path)
    arguments = dict(
        input_dir=str(root),
        output_dir=None,
        cls2idx=spec(),
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        resize_size=8,
        num_workers=0,
    )
    labels, train, val = BaseBuilder.build_dataloader(**arguments, multilabel=True, resample=False)
    assert labels is None and len(train) == 13 and len(train.dataset[3]) == 2
    with pytest.raises(ValueError, match="subsample"):
        BaseBuilder.build_dataloader(**arguments, multilabel=True, subsample=2)


@pytest.mark.parametrize("damage", ["payload", "header", "truncate", "missing", "index"])
def test_corrupt_shards_raise_with_the_shard_named(tmp_path, damage):
    root, splits = build(tmp_path)
    name = splits["train"]["shards"][0]["name"]
    tar = root / f"{name}.tar"
    if damage == "truncate":
        os.truncate(tar, tar.stat().st_size - 512)
    elif damage == "missing":
        tar.unlink()
    elif damage == "index":
        (root / f"{name}.parquet").write_bytes(b"not parquet")
    else:
        data = bytearray(tar.read_bytes())
        data[0 if damage == "header" else 512 + 40] ^= 0xFF
        tar.write_bytes(data)
    with pytest.raises(ShardError, match=name):
        list(dataset(root, shuffle_buffer=0))


def test_undecodable_images_and_unknown_labels_are_errors_not_skips(tmp_path):
    root, splits = build(tmp_path, counts={"train": 5, "val": 1})
    bad = ShardDataset(root, "train", lambda data: 1 / 0, spec(), 1, train=False)
    with pytest.raises(ShardError, match="Cannot decode sample 's00000'"):
        next(iter(bad))
    with pytest.raises(ShardError, match="is not in the class specification"):
        list(ShardDataset(root, "train", lambda d: d, {"0": {"0": 0}}, 1, train=False))


def test_inconsistent_manifest_is_rejected_and_cli_verifies(tmp_path):
    root, _ = build(tmp_path)
    done = subprocess.run([sys.executable, "-m", "mini_trainer.data.shards", "verify", str(root), "--full"], capture_output=True, text=True)
    assert done.returncode == 0 and "64 samples OK" in done.stdout
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["splits"]["train"]["n_samples"] += 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ShardError, match="declares 54"):
        dataset(root)


def test_lepinet_recipe_trains_and_resumes_from_a_shard_set_on_cpu(tmp_path):
    from mini_trainer.hierarchical.model import IndependentClassifier
    from mini_trainer.hierarchical.recipes import LepinetBuilder
    from mini_trainer.logging import configure_loggers
    from mini_trainer.train import main as train
    from tests.integration.test_integration_train import TinyMockModel

    root, _ = build(tmp_path / "shards")
    species_ids = [str(i) for i in range(64)]
    class_spec = tmp_path / "spec.json"
    class_spec.write_text(
        json.dumps(
            {
                "cls2idx": {
                    "0": {s: i for i, s in enumerate(species_ids)},
                    "1": {str(i): i for i in range(7)},
                    "2": {str(i): i for i in range(3)},
                },
                "labels": {s: [s, str(int(s) % 7), str(int(s) % 7 % 3)] for s in species_ids},
                "num_classes": [64, 7, 3],
            }
        )
    )
    arguments = dict(
        input=str(root),
        output=str(tmp_path / "out"),
        name="run",
        size=8,
        device="cpu",
        dtype="float32",
        seed=0,
        builder=LepinetBuilder,
        class_spec=str(class_spec),
        model_builder_kwargs={"model_type": TinyMockModel(), "pretrained": False, "cls": IndependentClassifier, "read_size": 12},
        dataloader_builder_kwargs={
            "batch_size": 4,
            "num_workers": 0,
            "read_size": 12,
            "class_sampling_power": 0.0,
            "shard_shuffle_buffer": 4,
        },
        augmentation_builder_kwargs={"size": 8},
        optimizer_builder_kwargs={
            "optimizer_cls": torch.optim.AdamW,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "backbone_lr": 1e-3,
            "decay_all": True,
        },
        criterion_builder_kwargs={"weights": [1, 1, 1]},
        regularizer_builder_kwargs={"strength": 0.0},
        lr_schedule_builder_kwargs={"warmup_epochs": 0.5, "start_factor": 0.01, "min_factor": 0.0, "pretrained_backbone": False},
        logger_builder_kwargs={"verbose": False, "macro_f1": True, "logger_cls": configure_loggers()},
    )
    precision = torch.get_float32_matmul_precision()
    try:
        train(epochs=2, **arguments)
        weights = tmp_path / "out" / "run" / "weights"
        assert torch.load(weights / "checkpoint_last.pth", weights_only=False)["epoch"] == 1
        train(epochs=3, checkpoint=str(weights / "checkpoint_last.pth"), **arguments)
    finally:
        torch.set_float32_matmul_precision(precision)
    assert torch.load(weights / "checkpoint_last.pth", weights_only=False)["epoch"] == 2
