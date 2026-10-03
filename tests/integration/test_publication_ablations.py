"""Research contracts: paired treatments, preserved partitions, and durable stages."""

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mini_trainer.modeling import Classifier
from mini_trainer.training import EMLACrossEntropy, class_weight_distribution_regularization
from publication.experiments.training_ablations import study, training
from publication.experiments.training_ablations.data import prepare_data, select_species, species_table, write_json


def test_phase_memory_retains_peak_across_batch_resets(tmp_path, monkeypatch):
    logger = training.StudyLogger.__new__(training.StudyLogger)
    logger.output_dir, logger._epoch, logger._type = str(tmp_path), 0, "train"
    logger._phase_peak = 0
    monkeypatch.setattr(logger, "summary", lambda: {})
    monkeypatch.setattr(training.MultiLogger, "log_memory_use", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    peaks = iter([1000, 600, 200])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: next(peaks))
    logger.log_memory_use()
    logger.log_memory_use()
    logger.save()
    record = json.loads((tmp_path / "learning.jsonl").read_text())
    assert record["peak_allocated_bytes"] == 1000


def metadata():
    return pd.DataFrame(
        [
            {
                "speciesKey": str(c),
                "familyKey": str(c // 2),
                "genusKey": str(c),
                "set": split,
                "gbifID": f"{c}-{split}-{i}",
                "filename": f"{split}-{i}.png",
                "scientificName": f"Species {c}",
            }
            for c in range(8)
            for split in ["0", "1", "2"]
            for i in range(c + 2)
        ]
    )


def fixture_campaign(tmp_path):
    frame = metadata()
    parquet = tmp_path / "source.parquet"
    frame.to_parquet(parquet)
    root = tmp_path / "study"
    root.mkdir()
    config = {
        **study.DEFAULTS,
        "parquet": str(parquet),
        "images": str(tmp_path / "images"),
        "species": 4,
        "size": 16,
        "batch_size": 4,
        "workers": 0,
        "device": "cpu",
        "wandb": False,
        "dtype": "float32",
    }
    prepare_data(config, root)
    write_json(root / "config.json", config)
    return root, config


def test_selection_preserves_splits_and_ignores_holdout_abundance(tmp_path):
    frame = metadata()
    table = species_table(frame)
    first = select_species(table, 4, 42)
    assert first == select_species(species_table(frame.sample(frac=1, random_state=5)), 4, 42)
    assert first == select_species(species_table(pd.concat([frame, frame[frame["set"] == "0"]])), 4, 42)
    assert first == select_species(table, 8, 42)[:4]
    root, _ = fixture_campaign(tmp_path)
    samples = pd.read_parquet(root / "samples.parquet")
    originals = frame.set_index(["speciesKey", "filename"])["set"]
    assert samples.set_index(["speciesKey", "filename"])["set"].equals(
        originals.reindex(samples.set_index(["speciesKey", "filename"]).index)
    )
    assert set(samples.groupby("speciesKey").split.nunique()) == {3}


def test_selection_rejects_cross_partition_observations(tmp_path):
    root, config = fixture_campaign(tmp_path)
    frame = metadata()
    frame["gbifID"] = "same-observation"
    frame.to_parquet(config["parquet"])
    with pytest.raises(ValueError, match="Observation crosses"):
        prepare_data(config, root)


def test_losses_and_regularizer_have_expected_behavior():
    logits = torch.randn(12, 4, requires_grad=True)
    targets = torch.arange(12) % 4
    expected = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.25)
    torch.testing.assert_close(EMLACrossEntropy([10] * 4, label_smoothing=0.25)(logits, targets), expected)
    fixed = training.FixedAdjustment([2, 4, 8, 16], label_smoothing=0.25)
    torch.testing.assert_close(
        fixed(logits, targets), torch.nn.functional.cross_entropy(logits + fixed.adjustments, targets, label_smoothing=0.25)
    )
    weights = torch.randn(4, 8, requires_grad=True)
    class_weight_distribution_regularization(weights).backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_shared_projection_and_isolated_random_stream():
    hashes = []
    for normalized in [False, True]:
        head = Classifier(8, 4, hidden=True, normalized=normalized, skip_spherical_init=True)
        training.initialize_head(head, 42)
        hashes.append(training.tensor_hash(head.hidden.state_dict().items()))
    assert hashes[0] == hashes[1]
    stream = training.RandomStream(lambda: torch.rand(3), 42)
    other = training.RandomStream(lambda: torch.rand(3), 42)
    torch.testing.assert_close(stream(), other())
    torch.rand(100)
    torch.testing.assert_close(stream(), other())


def test_tuning_tiebreak_and_main_matrix(tmp_path):
    paths = []
    for index, run in enumerate(study.tuning_runs(study.DEFAULTS)):
        path = tmp_path / str(index)
        path.mkdir()
        write_json(path / "run.json", run)
        write_json(path / "evaluation.json", {"macro_recall": 0.5, "nll": 1})
        paths.append(path)
    chosen = study.select_tuning(paths)
    assert all(v == {"lr": 0.0003, "weight_decay": 0.001} for v in chosen.values())
    runs = study.main_runs(study.DEFAULTS, chosen)
    assert len(runs) == 24
    assert len({r["id"] for r in runs}) == 24
    assert {r["seed"] for r in runs} == {42, 43, 44}


def test_metrics_probability_semantics_and_missing_support():
    result = training.metrics(np.zeros((2, 4)), [0, 1], [1, 2, 3, 4])
    assert result["nll"] == pytest.approx(np.log(4))
    assert result["brier"] == pytest.approx(0.75)
    assert result["recall"] == [1.0, 0.0, None, None]


def test_recovery_reuses_training_and_verifies_completed_artifacts(tmp_path, monkeypatch):
    run = {"id": "example"}
    stages = []

    def child(root, attempt, stage, *args):
        stages.append(stage)
        if stage == "train":
            (attempt / "model/weights").mkdir(parents=True)
            for name in ["model/weights/last.pt", "train.json", "initialization.json", "parameter_groups.json"]:
                (attempt / name).write_text("{}")
        elif stages.count("evaluate") == 1:
            raise RuntimeError("Evaluation interrupted")
        else:
            for name in ["evaluation.json", "predictions.npz"]:
                (attempt / name).write_text("{}")

    monkeypatch.setattr(study, "child", child)
    with pytest.raises(RuntimeError, match="interrupted"):
        study.execute(tmp_path, run, {}, "0", time.monotonic() + 300)
    path = study.execute(tmp_path, run, {}, "0", time.monotonic() + 300, retry=True)
    assert stages == ["train", "evaluate", "evaluate"]
    assert study.execute(tmp_path, run, {}, "0", time.monotonic() + 300) == path
    (path / "model/weights/last.pt").write_text("modified")
    with pytest.raises(ValueError, match="artifact changed"):
        study.completed(path, run)


def test_queue_does_not_hide_data_failure_behind_oom(tmp_path, monkeypatch):
    def execute(root, run, *args):
        if run["id"] == "oom":
            raise study.CUDAOutOfMemory("memory")
        raise ValueError("corrupt image")

    monkeypatch.setattr(study, "execute", execute)
    with pytest.raises(ValueError, match="corrupt image"):
        study.queue(tmp_path, [{"id": "oom"}, {"id": "data"}], {}, ["0", "1"], time.monotonic() + 100, False)


def test_deadline_prevents_new_child(tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        study.child(tmp_path, tmp_path, "train", "0", {}, time.monotonic())


def test_deadline_terminates_running_process(tmp_path, monkeypatch):
    popen = subprocess.Popen
    processes = []

    def sleeping_child(command, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(study.subprocess, "Popen", sleeping_child)
    with pytest.raises(TimeoutError, match="deadline"):
        study.child(tmp_path, tmp_path, "train", "0", {"threads": 1}, time.monotonic() + 60.2)
    assert processes and processes[0].poll() is not None


@pytest.mark.parametrize(
    "hidden,optimizer,normalized,loss",
    [
        (False, "adamw", True, "emla"),
        (True, "adamw", True, "emla"),
        (False, "muon", True, "emla"),
        (True, "muon", True, "emla"),
        (True, "muon", False, "fixed"),
        (False, "adamw", False, "ce"),
    ],
)
def test_tiny_train_reload_evaluate(tmp_path, monkeypatch, hidden, optimizer, normalized, loss):
    from mini_trainer.modeling import classifier
    from tests.integration.test_integration_train import TinyMockModel

    root, config = fixture_campaign(tmp_path)
    for row in pd.read_parquet(root / "samples.parquet").itertuples():
        path = tmp_path / "images" / row.sample_id
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 30 + 20 * row.label, dtype=np.uint8)).save(path)
    original = classifier.get_model
    monkeypatch.setattr(
        classifier,
        "get_model",
        lambda *args, **kwargs: original(
            TinyMockModel(), model_args={"pretrained": False}, transform=training.transforms.ConvertImageDtype(torch.float32)
        ),
    )
    torch.save(TinyMockModel().state_dict(), root / "pretrained.pt")
    # Test the real training/evaluation orchestration without compiling augmentation kernels.
    monkeypatch.setattr(training.StudyBuilder, "build_augmentation", lambda **kwargs: training.transforms.Compose([]))
    built = []
    build_model = training.StudyBuilder.build_model

    def capture(cls, **kwargs):
        result = build_model(**kwargs)
        built.append(result)
        return result

    monkeypatch.setattr(training.StudyBuilder, "build_model", classmethod(capture))
    attempt = root / "attempt"
    attempt.mkdir()
    run = {
        **study.FULL,
        "hidden": hidden,
        "optimizer": optimizer,
        "normalized": normalized,
        "loss": loss,
        "seed": 42,
        "lr": 0.001,
        "weight_decay": 0.01,
        "epochs": 2,
        "id": "tiny",
        "tuning": True,
    }
    training.train(root, attempt, config, run)
    training.evaluate(root, attempt, config, run)
    result = json.loads((attempt / "evaluation.json").read_text())
    assert result["split"] == "validation"
    assert result["backbone_parameters_changed"]
    assert sum(result["support"]) > 0
    assert np.isfinite(result["nll"])
    live, preprocess = built[0]
    loaded, loaded_preprocess = Classifier.build(weights=str(attempt / "model/weights/last.pt"), device="cpu")
    live.eval()
    loaded.eval()
    images = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
    with torch.inference_mode():
        torch.testing.assert_close(live(preprocess(images)), loaded(loaded_preprocess(images)))
