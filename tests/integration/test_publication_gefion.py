"""Gefion runs become evidence studies with test-only samples, vocabulary taxonomy and recorded settings."""

import json

import pandas as pd
import yaml

from publication.experiments import gefion
from publication.experiments.training_ablations.data import digest


def test_prepare_builds_study_from_recorded_run(tmp_path):
    run = tmp_path / "meta/efficientnet_b0/results/runs/hierarchical_plantnet"
    (run / "logs").mkdir(parents=True)
    config = {
        "epochs": 10,
        "dtype": "float16",
        "model_builder_kwargs": {"model_type": "efficientnet_b0", "normalized": True, "hidden": True, "droprate": 0.1},
        "dataloader_builder_kwargs": {"batch_size": 64},
        "optimizer_builder_kwargs": {"optimizer_cls": "mini_trainer.training.muon.MuonAuxAdamW", "lr": 0.001, "weight_decay": 0.01},
        "criterion_builder_kwargs": {"weighted": True},
        "lr_schedule_builder_kwargs": {"warmup_epochs": 0.25},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    cls2idx = {"0": {"s1": 1, "s0": 0}, "1": {"g0": 0}}
    (run / "class_spec.json").write_text(json.dumps({"cls2idx": cls2idx, "labels": {"s0": ["s0", "g0"], "s1": ["s1", "g0"]}}))
    pd.DataFrame({"epoch": [0], "type": ["train"], "loss": [1.0]}).to_csv(run / "logs/summary.csv", index=False)
    weights = tmp_path / "weights/efficientnet_b0/results/runs/hierarchical_plantnet/weights/last.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"checkpoint")
    index = {
        "path": ["images_gbif/s0/a.jpg", "images_gbif/s1/b.jpg", "images_gbif/s0/c.jpg"],
        "split": ["test", "train", "test"],
        "label": [["s0", "g0"], ["s1", "g0"], ["s0", "g0"]],
    }
    (tmp_path / "index.json").write_text(json.dumps(index))

    gefion.prepare(tmp_path / "cohort", "plantnet", [run], tmp_path / "weights", tmp_path / "index.json")
    study = tmp_path / "cohort/study"
    samples = pd.read_parquet(study / "samples.parquet")
    assert samples.sample_id.tolist() == ["images_gbif/s0/a.jpg", "images_gbif/s0/c.jpg"] and samples.label.tolist() == [0, 0]
    classes = json.loads((study / "classes.json").read_text())
    assert classes["counts"] == [0, 1] and classes["taxonomy"] == {"s0": ["s0", "g0"], "s1": ["s1", "g0"]}
    assert json.loads((study / "data_index.json").read_text())["path"][0] == "/work/plantnet/images_gbif/s0/a.jpg"
    (plan,) = json.loads((study / "plan.json").read_text())
    assert plan["id"] == "efficientnet_b0_hierarchical" and plan["loss"] == "emla" and plan["optimizer"] == "MuonAuxAdamW"
    attempt = study / "runs/efficientnet_b0_hierarchical/attempt-000"
    assert json.loads((attempt / "train.json").read_text())["weights_sha256"] == digest(weights)
    assert json.loads((attempt / "model/logs/learning.jsonl").read_text())["phase"] == "train"
