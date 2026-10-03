"""Operational sampling and actual controller-signal cleanup contracts."""

import json
import os
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest

from publication.experiments.training_ablations.operational import sample_frame


def test_profile_sample_is_random_reproducible_and_order_independent():
    frame = pd.DataFrame({"sample_id": [f"image-{i:04d}" for i in range(1000)], "label": [i // 10 for i in range(1000)]})
    sampled = sample_frame(frame, 100, 39)
    pd.testing.assert_frame_equal(sampled, sample_frame(frame.sample(frac=1, random_state=8), 100, 39))
    assert sampled.sample_id.is_unique
    assert sampled.label.max() > 90
    assert sampled.label.min() < 10
    assert len(sample_frame(frame, 2000, 39)) == 1000


def test_sigterm_stops_worker_and_retry_retains_failed_attempt(tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text("""
import json, signal, subprocess, sys, time
from pathlib import Path
from publication.experiments.training_ablations import study
root = Path(sys.argv[1])
run = {"id": "signal-probe"}
signal.signal(signal.SIGTERM, lambda *_: study.STOP.set())
if sys.argv[2] == "interrupt":
    popen = subprocess.Popen
    def worker(command, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        (root / "worker.pid").write_text(str(process.pid))
        return process
    study.subprocess.Popen = worker
else:
    def child(root, attempt, stage, *args):
        (attempt / "model/weights").mkdir(parents=True, exist_ok=True)
        names = (["model/weights/last.pt", "train.json", "initialization.json", "parameter_groups.json"]
                 if stage == "train" else ["evaluation.json", "predictions.npz"])
        for name in names:
            (attempt / name).write_text("{}")
    study.child = child
study.execute(root, run, {"threads": 1}, "0", time.monotonic() + 120, retry=sys.argv[2] == "retry")
""")
    with (tmp_path / "controller.log").open("w") as log:
        controller = subprocess.Popen([sys.executable, str(driver), str(tmp_path), "interrupt"], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / "worker.pid").exists():
                if controller.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("Controller failed to start its worker")
                time.sleep(0.02)
            worker_pid = int((tmp_path / "worker.pid").read_text())
            controller.send_signal(signal.SIGTERM)
            assert controller.wait(timeout=15) != 0
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait()
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)
    runs = tmp_path / "runs/signal-probe"
    failure = json.loads((runs / "attempt-000/failure.json").read_text())
    assert failure["type"] == "InterruptedError"
    subprocess.run([sys.executable, str(driver), str(tmp_path), "retry"], check=True, timeout=15)
    assert (runs / "attempt-000/failure.json").exists()
    assert not (runs / "attempt-000/complete.json").exists()
    assert (runs / "attempt-001/complete.json").exists()


def test_profile_uses_real_trainer_and_reload_on_random_subset(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from PIL import Image

    from mini_trainer.modeling import classifier
    from publication.experiments.training_ablations import operational, study, training
    from tests.integration.test_integration_train import TinyMockModel
    from tests.integration.test_publication_ablations import fixture_campaign

    root, config = fixture_campaign(tmp_path)
    frame = pd.read_parquet(root / "samples.parquet")
    train = frame[frame.split == "train"].sample(n=48, replace=True, random_state=39).reset_index(drop=True)
    train["sample_id"] = [f"train-{i}.png" for i in range(len(train))]
    frame = pd.concat([train, frame[frame.split != "train"]])
    frame.to_parquet(root / "samples.parquet", index=False)
    for row in frame.itertuples():
        path = tmp_path / "images" / row.sample_id
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 30 + 20 * row.label, dtype=np.uint8)).save(path)
    (root / "prepared.json").write_text("{}")
    torch.save(TinyMockModel().state_dict(), root / "pretrained.pt")
    original = classifier.get_model
    monkeypatch.setattr(
        classifier,
        "get_model",
        lambda *args, **kwargs: original(
            TinyMockModel(), model_args={"pretrained": False}, transform=training.transforms.ConvertImageDtype(torch.float32)
        ),
    )
    monkeypatch.setattr(study, "verify", lambda _: config)
    monkeypatch.setattr(operational.ProfileBuilder, "build_augmentation", lambda **kwargs: training.transforms.Compose([]))
    # Restore the builder after the dedicated-process helper is exercised in pytest.
    monkeypatch.setattr(training, "StudyBuilder", training.StudyBuilder)
    output = tmp_path / "profile"
    config["batch_size"] = 8
    (root / "qualified.json").write_text(json.dumps({"batch_size": 8}))
    frozen_bytes = (root / "config.json").read_bytes()
    operational.profile(root, output, 48, batch_size=4, workers=0)
    resolved = json.loads((output / "resolved.json").read_text())
    assert resolved["batch_size"] == 4 and resolved["workers"] == 0
    assert config["batch_size"] == 8
    assert (root / "config.json").read_bytes() == frozen_bytes
    result = json.loads((output / "profile.json").read_text())
    assert result["training_batches_measured"] == 8
    assert result["training_images_per_second"] > 0
    assert result["validation_images_per_second"] > 0
    assert result["sample_images"] == 48
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["split"] == "validation"
    assert evaluation["backbone_parameters_changed"]
    assert not (root / "runs").exists()


@pytest.mark.parametrize("oom", [False, True])
def test_profile_cli_retains_oom_without_disguising_other_failures(tmp_path, monkeypatch, oom):
    import torch

    from publication.experiments.training_ablations import operational

    output = tmp_path / "profile"
    error = torch.cuda.OutOfMemoryError("capacity exhausted") if oom else ValueError("invalid image")

    def fail(root, output, samples, **kwargs):
        output.mkdir()
        raise error

    monkeypatch.setattr(operational, "profile", fail)
    monkeypatch.setattr(sys, "argv", ["operational", str(tmp_path), str(output), "--batch-size", "768", "--workers", "48"])
    with pytest.raises(type(error), match=str(error)):
        operational.main()
    failure = json.loads((output / "failure.json").read_text())
    assert failure["cuda_oom"] is oom
    assert failure["type"] == type(error).__name__
