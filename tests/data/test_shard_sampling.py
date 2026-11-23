"""Opt-in class-weighted sampling of a shard set's training split (``class_sampling_power``)."""

from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import mini_trainer.data.shards as shards
from mini_trainer.builders import BaseBuilder
from mini_trainer.data import ShardDataset, ShardWriter, write_manifest
from tests.data.test_shards import COLUMNS, jpeg, spec

pytest.importorskip("pyarrow")
SIZES = {0: 200, 1: 50, 2: 8, 3: 2}  # training samples per class


def build_skewed(root, per_shard=37):
    """Classes of very different sizes, interleaved over several shards; sample i's image encodes i % 256."""
    labels = np.concatenate([np.full(n, c) for c, n in SIZES.items()])
    labels = labels[np.random.default_rng(0).permutation(len(labels))]
    writer = ShardWriter(root, "train", COLUMNS, max_bytes=per_shard * 700)
    for i, c in enumerate(labels.tolist()):
        writer.add(f"s{i:05d}", jpeg(i % 256), {"speciesKey": str(c), "genusKey": str(c % 7), "familyKey": str(c % 7 % 3)})
    write_manifest(root, {"train": writer.close()}, COLUMNS, {"test": True})
    return root


def first_byte(data):
    return torch.tensor([data[0]])


def make(root, power=0.0, batch_size=1, **kwargs):
    return ShardDataset(root, "train", first_byte, spec(), batch_size, train=True, multilabel=True, class_sampling_power=power, **kwargs)


def labels_of(ds, epoch, workers=0):
    ds.set_epoch(epoch)
    return [int(label[0]) for _, label in DataLoader(ds, batch_size=None, num_workers=workers)]


def global_draw(ds, real_epoch):
    """Leaf labels of every draw of a real epoch, from the plan alone (no image read)."""
    counts = ds.draw_counts(real_epoch)
    return [int(c) for shard, k in enumerate(counts.tolist()) for c in ds._leaf_labels(shard)[ds.shard_draws(shard, real_epoch, k)]]


def test_class_frequency_of_the_draws_follows_count_to_the_one_minus_power(tmp_path):
    root = build_skewed(tmp_path / "s")
    total = sum(SIZES.values())
    for power in (0.5, 1.0):
        ds = make(root, power)
        assert len(ds) == total  # same number of draws per epoch as samples
        drawn = Counter(c for epoch in range(300) for c in global_draw(ds, epoch))
        expected = np.array([n ** (1 - power) for n in SIZES.values()])
        expected = expected / expected.sum()
        share = np.array([drawn[c] for c in SIZES]) / sum(drawn.values())
        assert np.allclose(share, expected, rtol=0.05), (power, share, expected)
    # and the loader yields exactly the planned draws
    ds = make(root, 0.5)
    assert sorted(labels_of(ds, 3)) == sorted(global_draw(ds, 3))


def test_power_zero_and_the_default_keep_todays_plan_exactly(tmp_path):
    root = build_skewed(tmp_path / "s")
    for epoch in (0, 1):
        assert labels_of(make(root, 0.0), epoch) == labels_of(
            ShardDataset(root, "train", first_byte, spec(), 1, train=True, multilabel=True), epoch
        )
    _, train, _ = BaseBuilder.build_dataloader(
        input_dir=str(root),
        output_dir=None,
        cls2idx=spec(),
        batch_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        resize_size=8,
        num_workers=0,
        multilabel=True,
        class_sampling_power=0.0,
        splits=("train", "train"),
    )
    assert train.dataset.class_sampling_power == 0.0 and train.dataset._shard_weight is None


def test_draws_are_deterministic_per_seed_and_epoch_and_resume(tmp_path):
    root = build_skewed(tmp_path / "s")
    a, b = make(root, 0.5), make(root, 0.5)
    assert labels_of(a, 2) == labels_of(b, 2)  # a fresh dataset at epoch 2 replays it (resume)
    assert labels_of(a, 2) != labels_of(a, 3)
    assert labels_of(make(root, 0.5, seed=1), 2) != labels_of(a, 2)


def test_ranks_workers_and_virtual_epochs_take_disjoint_parts_of_one_draw(tmp_path, monkeypatch):
    root = build_skewed(tmp_path / "s")
    full = Counter(global_draw(make(root, 0.5), 4))
    # workers: the same multiset whatever their number
    assert Counter(labels_of(make(root, 0.5), 4, workers=0)) == Counter(labels_of(make(root, 0.5), 4, workers=2)) == full
    # ranks: disjoint shares of the global draw, equal lengths, nothing drawn twice beyond the draw itself
    world, parts = 3, []
    for rank in range(world):
        monkeypatch.setattr(shards, "get_rank", lambda rank=rank: rank)
        monkeypatch.setattr(shards, "get_world_size", lambda: world)
        ds = make(root, 0.5)
        parts.append(Counter(labels_of(ds, 4)))
        assert sum(parts[-1].values()) == len(ds) == sum(full.values()) // world
    assert not (sum(parts, Counter()) - full)
    monkeypatch.undo()
    # virtual epochs: the parts of real epoch 2 together are its draw (up to V - 1 samples left out)
    ds = make(root, 0.5, virtual_epochs=3)
    parts_of_epoch = Counter(labels_of(ds, 6)) + Counter(labels_of(ds, 7)) + Counter(labels_of(ds, 8))
    assert not (parts_of_epoch - Counter(global_draw(ds, 2))) and sum(parts_of_epoch.values()) >= sum(SIZES.values()) - 2


def test_weighted_sampling_is_refused_outside_plain_training(tmp_path):
    root = build_skewed(tmp_path / "s")
    with pytest.raises(ValueError, match="training split"):
        ShardDataset(root, "train", first_byte, spec(), 1, train=False, multilabel=True, class_sampling_power=0.5)


def test_an_omitted_power_never_oversamples_a_shard_set(tmp_path):
    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.hierarchical.recipes import LepinetBuilder

    root = build_skewed(tmp_path / "s")
    arguments = dict(
        input_dir=str(root), output_dir=None, cls2idx=spec(), batch_size=4, device=torch.device("cpu"), dtype=torch.float32,
        resize_size=8, num_workers=0, splits=("train",),
    )  # fmt: skip
    # the recipe's 0.5 is a folder default: on a shard set the value must be stated
    with pytest.raises(ValueError, match="explicit class_sampling_power"):
        LepinetBuilder.build_dataloader(**arguments)
    _, train = LepinetBuilder.build_dataloader(**arguments, class_sampling_power=0.0)
    assert train.dataset._shard_weight is None
    # builders without a sampling default draw naturally when the option is absent
    for builder, extra in ((BaseBuilder, {"multilabel": True}), (HierarchicalBuilder, {})):
        _, train = builder.build_dataloader(**arguments, **extra)
        assert train.dataset._shard_weight is None and train.dataset.class_sampling_power == 0.0
