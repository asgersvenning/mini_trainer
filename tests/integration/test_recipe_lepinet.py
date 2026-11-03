import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mini_trainer.hierarchical.integration import HierarchicalBuilder, HierarchicalResultCollector
from mini_trainer.hierarchical.model import HierarchicalClassifier, IndependentClassifier
from mini_trainer.hierarchical.recipes import LepinetBuilder
from mini_trainer.integrations.parquet import KCOLUMNS
from mini_trainer.logging import configure_loggers
from mini_trainer.modeling import Classifier, classification_module
from mini_trainer.modeling.architectures import ResampleNormalize
from tests.integration.test_integration_train import TinyMockModel


def _dataset(root):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(0)
    rows = []
    # Species 3 has one non-test image, below the floor of 3; its test images become out-of-vocabulary.
    for species, genus, per_set in [(1, 10, {"2": 6, "1": 2, "0": 1}), (2, 20, {"3": 6, "1": 2, "0": 1}), (3, 20, {"4": 1, "0": 3})]:
        folder = root / "images" / str(species)
        folder.mkdir(parents=True, exist_ok=True)
        for split, count in per_set.items():
            for i in range(count):
                name = f"{split}_{i}.jpg"
                Image.fromarray(rng.integers(0, 256, (14 + i, 18, 3), dtype=np.uint8)).save(folder / name)
                rows.append({"filename": name, "set": split, "speciesKey": species, "genusKey": genus})
    path = root / "data.parquet"
    pq.write_table(pa.table({k: [str(row.get(k, 7)) for row in rows] for k in ("filename", "set", *KCOLUMNS)}), path)
    return path


@pytest.mark.parametrize(
    "preset,head,weights",
    [("I", IndependentClassifier, [1, 1, 1]), ("BU", HierarchicalClassifier, [1, 1, 1]), ("F", HierarchicalClassifier, [1, 0, 0])],
)
def test_presets_resolve_to_the_recipe(monkeypatch, preset, head, weights):
    from mini_trainer.train import cli

    config = Path(__file__).resolve().parents[2] / "dev" / "lepinet" / f"lepinet-{preset}.yaml"
    monkeypatch.setattr("sys.argv", ["mt_train", "--config", str(config), "--input", "data.parquet"])
    args = cli()
    assert args["builder"] is LepinetBuilder and args["model_builder_kwargs"]["cls"] is head
    assert args["optimizer_builder_kwargs"]["optimizer_cls"] is torch.optim.AdamW
    assert args["criterion_builder_kwargs"]["weights"] == weights and args["spec_model_dataloader_kwargs"]["min_count"] == 50
    # Reader, model preprocessing and augmentation must agree on the two image sizes.
    assert args["dataloader_builder_kwargs"]["read_size"] == args["model_builder_kwargs"]["read_size"] == LepinetBuilder.READ_SIZE
    assert args["augmentation_builder_kwargs"]["size"] == args["size"]


def test_lepinet_recipe_trains_reloads_and_predicts_on_cpu(tmp_path):
    from mini_trainer.predict import main as predict
    from mini_trainer.train import main as train

    parquet = _dataset(tmp_path / "data")
    precision = torch.get_float32_matmul_precision()
    try:
        train(
            input=str(parquet),
            output=str(tmp_path / "out"),
            name="run",
            epochs=1,
            size=8,
            device="cpu",
            dtype="float32",
            seed=0,
            builder=LepinetBuilder,
            spec_model_dataloader_kwargs={"min_count": 3},
            model_builder_kwargs={"model_type": TinyMockModel(), "pretrained": False, "cls": IndependentClassifier, "read_size": 12},
            dataloader_builder_kwargs={"batch_size": 2, "num_workers": 0, "read_size": 12},
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
    finally:
        torch.set_float32_matmul_precision(precision)
    run = tmp_path / "out" / "run"
    with open(run / "logs" / "summary.csv") as handle:
        rows = list(csv.DictReader(handle))
    evaluation = next(row for row in rows if row["type"] == "eval")
    assert {"f1", "f1/lvl1", "f1/lvl2"} <= set(evaluation) and 0 <= float(evaluation["f1"]) <= 1

    model, preprocess = Classifier.build(weights=str(run / "weights" / "last.pt"))
    head = classification_module(model)
    assert isinstance(head, IndependentClassifier)
    assert list(head.metadata["cls2idx"]["0"]) == ["1", "2"]  # the floor removed species 3
    assert head.float32_head and all(layer.bias.requires_grad for layer in head.layers)
    assert any(layer.bias.abs().sum() > 0 for layer in head.layers)  # biases trained from zero
    assert isinstance(preprocess.transform, ResampleNormalize) and preprocess.transform.read_size == 12

    predict(
        input=str(parquet),
        weights=str(run / "weights" / "last.pt"),
        output=str(tmp_path / "predictions"),
        name="test",
        device="cpu",
        dtype="float32",
        builder=HierarchicalBuilder,
        collector_cls=HierarchicalResultCollector,
        collector_cls_kwargs={"scientific_names": False},  # GBIF keys as labels, no network lookups
        dataloader_builder_kwargs={"batch_size": 2, "num_workers": 0},
    )
    with open(tmp_path / "predictions" / "test" / "mini_metric.csv") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 5 * 3  # five set-0 images, three levels
    unknown = {row["instance_id"] for row in predictions if row["level"] == "0" and row["known_label"] == "0"}
    assert len(unknown) == 3  # species 3 is out of vocabulary; the scorer restricts to the vocabulary
