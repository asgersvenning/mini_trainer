"""Head/tail monitoring (opt-in): validation statistics by training-frequency bin and a training sample scored in
evaluation mode, both in summary.csv."""

import csv
import json
import math

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

import mini_trainer.logging.core as core
from mini_trainer.data import ShardDataset, get_fit_sample_dataloader, stratified_rows, training_leaf_labels
from tests.data.test_shards import build, dataset, spec


def first_byte(data):
    """A picklable reader for worker processes: the toy shards encode each sample's id in its brightness."""
    return torch.tensor([data[0]])


def logger(**kwargs):
    return core.MultiLogger(train_loader=[0], val_loader=[0], epochs=1, output=None, name="run", **kwargs)


def test_frequency_bins_average_recall_and_f1_over_the_classes_of_each_bin():
    # Classes 0-1 have < 10 training images, 2-3 have 10-19, 4 has >= 20.
    log = logger(frequency_bins=[10, 20], class_counts=[3, 9, 10, 19, 50])
    log.update(epoch=0, type="eval")
    log.log_labels_predictions([0, 0, 1, 2, 2, 3], [0, 1, 1, 2, 0, 4], level=0)
    log.log_frequency_bins()
    s = log.summary()
    # per class (correct, true, predicted): 0 (1, 2, 2), 1 (1, 1, 2), 2 (1, 2, 1), 3 (0, 1, 0); class 4 not in the truth
    assert s["recall"] == pytest.approx((0.5 + 1 + 0.5 + 0) / 4)
    assert s["recall/<10"] == pytest.approx((0.5 + 1) / 2)
    assert s["f1/<10"] == pytest.approx((2 / 4 + 2 / 3) / 2)
    assert s["recall/10-20"] == pytest.approx((0.5 + 0) / 2)
    assert s["f1/10-20"] == pytest.approx((2 / 3 + 0) / 2)
    assert math.isnan(s["recall/>=20"])  # no class of the bin in the truth


def test_fit_sample_statistics_and_columns_only_when_enabled():
    log = logger(frequency_bins=[10], class_counts=[5, 50, 50], fit_sample=True)
    log.update(epoch=0, type="eval")
    log.log_fit_sample(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
    s = log.summary()
    assert s["fit/acc1"] == pytest.approx(75.0)
    assert s["fit/recall"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert s["fit/f1"] == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)
    assert s["fit/recall/<10"] == pytest.approx(0.5)
    assert s["fit/recall/>=10"] == pytest.approx(1.0)
    default = logger().statistics
    assert not [k for k in default if k.startswith(("fit/", "recall", "f1/"))]
    with pytest.raises(ValueError, match="class_counts"):
        logger(frequency_bins=[10])
    with pytest.raises(ValueError, match="increasing"):
        logger(frequency_bins=[20, 10], class_counts=[1])


def test_stratified_rows_keep_up_to_n_rows_of_every_class_deterministically():
    labels = np.array([0] * 7 + [1] * 1 + [2] * 3)
    rows = stratified_rows(labels, 2, seed=3)
    assert np.all(np.diff(rows) > 0)
    assert np.bincount(labels[rows]).tolist() == [2, 1, 2]
    assert np.array_equal(rows, stratified_rows(labels, 2, seed=3))


def test_shard_positions_read_exactly_the_requested_training_samples_across_workers(tmp_path):
    root, _ = build(tmp_path / "shards")  # train: ids 0-52 over 6 shards
    train = dataset(root, "train", train=True)
    assert training_leaf_labels(train).tolist() == list(range(53))  # species = sample id
    rows = np.array([0, 4, 9, 10, 31, 52])
    fit = ShardDataset(root, "train", first_byte, spec(), 2, train=False, multilabel=True, positions=rows)
    assert len(fit) == 6
    for workers in (0, 2, 4):
        got = sorted(int(label[0]) for _, label in DataLoader(fit, batch_size=None, num_workers=workers))
        assert got == rows.tolist(), workers
    with pytest.raises(ValueError, match="evaluation mode"):
        dataset(root, "train", train=True, positions=rows)


def test_fit_loader_reads_training_rows_with_the_validation_reader(tmp_path):
    root, _ = build(tmp_path / "shards")
    train, val = dataset(root, "train", train=True), dataset(root, "val", train=False)
    loader = get_fit_sample_dataloader(train, val, np.array([1, 2, 50]), batch_size=2, num_workers=0, device=torch.device("cpu"))
    assert sorted(int(t[0]) for _, target in loader for t in target) == [1, 2, 50]


def run_dir_rows(path):
    with open(path / "logs" / "summary.csv") as f:
        return list(csv.DictReader(f))


def test_folder_training_logs_frequency_bins_and_the_fit_sample_in_summary(tmp_path):
    from mini_trainer.logging import configure_loggers
    from mini_trainer.train import main
    from tests.integration.test_integration_train import TinyMockModel

    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    for name, n in (("class_a", 12), ("class_b", 4)):  # class_b is the rare one
        (data / name).mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)).save(data / name / f"{i}.jpg")
    main(
        input=str(data),
        output=str(tmp_path / "out"),
        name="run",
        epochs=3,
        size=8,
        device="cpu",
        dtype="float32",
        seed=0,
        model_builder_kwargs={"model_type": TinyMockModel(), "pretrained": False},
        dataloader_builder_kwargs={"batch_size": 4, "num_workers": 0, "train_proportion": 0.75},
        logger_builder_kwargs={
            "logger_cls": configure_loggers(),
            "frequency_bins": [5],
            "fit_sample_per_class": 2,
            "fit_sample_every": 2,
        },
    )
    rows = run_dir_rows(tmp_path / "out" / "run")
    evals = [r for r in rows if r["type"] == "eval"]
    assert [r["epoch"] for r in evals] == ["0", "1", "2"]
    assert all(not math.isnan(float(r["recall"])) for r in evals)
    # scored after the 2nd epoch (every 2) and after the last; 2 of each class
    assert [math.isnan(float(r["fit/acc1"])) for r in evals] == [True, False, False]
    assert all(math.isnan(float(r["fit/acc1"])) for r in rows if r["type"] == "train")
    config = (tmp_path / "out" / "run" / "config.yaml").read_text()
    assert "fit_sample_per_class: 2" in config and "class_counts" not in config


def test_shard_training_logs_the_fit_sample(tmp_path):
    from mini_trainer.hierarchical.model import IndependentClassifier
    from mini_trainer.hierarchical.recipes import LepinetBuilder
    from mini_trainer.logging import configure_loggers
    from mini_trainer.train import main
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
    precision = torch.get_float32_matmul_precision()
    try:
        main(
            input=str(root),
            output=str(tmp_path / "out"),
            name="run",
            epochs=1,
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
            logger_builder_kwargs={"logger_cls": configure_loggers(), "frequency_bins": [1, 2], "fit_sample_per_class": 1},
        )
    finally:
        torch.set_float32_matmul_precision(precision)
    (row,) = [r for r in run_dir_rows(tmp_path / "out" / "run") if r["type"] == "eval"]
    # every training species has exactly one image: all in the ">=1" bin of [1, 2) and scored once each
    assert not math.isnan(float(row["fit/acc1"])) and not math.isnan(float(row["fit/recall/1-2"]))
    assert math.isnan(float(row["fit/recall/<1"])) and math.isnan(float(row["fit/recall/>=2"]))
