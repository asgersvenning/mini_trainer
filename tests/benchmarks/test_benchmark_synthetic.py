import csv

import numpy as np
from PIL import Image

from dev.benchmarks.synthetic import generate, oracle


def test_synthetic_reproducibility_and_oracle(tmp_path):
    first = generate(tmp_path / "first", train_per_class=3, val_per_class=2, test_per_class=2)
    second = generate(tmp_path / "second", train_per_class=3, val_per_class=2, test_per_class=2)
    assert first == second
    records = first["records"]
    images = np.stack([np.asarray(Image.open(tmp_path / "first" / record["path"])) for record in records])
    np.testing.assert_array_equal(oracle(images), [record["label"] for record in records])
    assert len({record["sha256"] for record in records}) == len(records)
    assert {record["split"] for record in records} == {"train", "val", "test"}
    changed = generate(tmp_path / "changed", seed=43, train_per_class=3, val_per_class=2, test_per_class=2)
    assert {record["sha256"] for record in records}.isdisjoint(record["sha256"] for record in changed["records"])


def test_synthetic_training_matches_oracle_and_repeats(tmp_path):
    import json

    import torch

    from dev.benchmarks.run import run

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        first = run(tmp_path / "first", cache="CPU", cache_workers=0)
        second = run(tmp_path / "second", cache="RAM", cache_workers=0)
    finally:
        torch.set_num_threads(threads)
    assert first["cache"] == second["cache"] == "CPU"
    assert len(first["phase_measurements"]) == 24
    assert [phase["phase"] for phase in first["phase_measurements"]] == ["train", "eval"] * 12
    assert all(phase["seconds"] >= 0 and phase["peak_cuda_allocated_bytes"] is None for phase in first["phase_measurements"])
    assert first["peak_cuda_allocated_bytes"] is None
    assert first["test_accuracy"] == second["test_accuracy"] == 1.0
    for name in ("first", "second"):
        with (tmp_path / name / "training/logs/summary.csv").open() as stream:
            summaries = list(csv.DictReader(stream))
        assert [(int(row["epoch"]), row["type"]) for row in summaries] == [
            (epoch, phase) for epoch in range(12) for phase in ("train", "eval")
        ]
        assert all(np.isfinite(float(row["loss"])) and float(row["loss"]) > 0 for row in summaries)
        assert float(summaries[-1]["acc1"]) == 100.0
    assert first["dataset_manifest_sha256"] == second["dataset_manifest_sha256"]
    with np.load(tmp_path / "first/predictions.npz") as a, np.load(tmp_path / "second/predictions.npz") as b:
        np.testing.assert_array_equal(a["scores"], b["scores"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        index = json.loads((tmp_path / "first/train_index.json").read_text())
        assert set(index["split"]) == {"train", "val"}
        assert all("/test/" not in path for path in index["path"])


def test_requested_gpu_profile_does_not_fall_back_to_cpu(tmp_path, monkeypatch):
    import pytest
    import torch

    from dev.benchmarks.run import run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no accessible CUDA device"):
        run(tmp_path / "gpu", device="cuda:0")
    assert not (tmp_path / "gpu").exists()


def test_cli_retains_failure_report(tmp_path, monkeypatch):
    import json
    import sys

    import pytest
    import torch

    from dev.benchmarks.run import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--output",
            str(tmp_path / "failed"),
            "--device",
            "cuda:0",
            "--compile",
            "--compile-mode",
            "reduce-overhead",
            "--compile-optimizer",
            "--optimizer-cudagraphs",
            "--fine-tune",
        ],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: None)
    with pytest.raises(RuntimeError, match="no accessible CUDA device"):
        main()
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed"
    assert report["device"] == "cuda:0"
    assert report["compile_mode"] == "reduce-overhead"
    assert report["optimizer_cudagraphs"] is True
    assert report["fine_tune"] is True
    assert report["error"]["type"] == "RuntimeError"
    assert "test_accuracy" not in report


def test_qt_profile_requires_cuda_before_creating_output(tmp_path):
    import pytest

    from dev.benchmarks.run import run

    with pytest.raises(ValueError, match="require CUDA"):
        run(tmp_path / "qt", quantized_training=True)
    assert not (tmp_path / "qt").exists()


def test_compile_mode_requires_compilation_before_creating_outputs(tmp_path):
    import pytest

    from dev.benchmarks.run import run
    from mini_trainer.train import main
    from mini_trainer.training.compilation import model_compile_options

    for mode in ("reduce-overhead", "invalid"):
        with pytest.raises(ValueError, match="requires compile=True"):
            run(tmp_path / "benchmark", compile_mode=mode)
        with pytest.raises(ValueError, match="requires compile=True"):
            main(input=str(tmp_path / "missing"), output=str(tmp_path / "train"), compile_mode=mode)
    with pytest.raises(ValueError, match="Unknown model compile mode"):
        model_compile_options(True, "invalid")
    assert not list(tmp_path.iterdir())


def test_optimizer_graphs_require_compilation_before_output(tmp_path):
    import pytest

    from dev.benchmarks.run import run
    from mini_trainer.train import main

    with pytest.raises(ValueError, match="requires compile_optimizer=True"):
        run(tmp_path / "benchmark", optimizer_cudagraphs=True)
    with pytest.raises(ValueError, match="requires compile_optimizer=True"):
        main(input=str(tmp_path / "missing"), output=str(tmp_path / "train"), optimizer_cudagraphs=True)
    assert not list(tmp_path.iterdir())


def test_optimizer_graphs_reject_cpu_before_output(tmp_path):
    import pytest

    from dev.benchmarks.run import run
    from mini_trainer.train import main

    with pytest.raises(ValueError, match="require CUDA"):
        run(tmp_path / "benchmark", device="cpu", compile_optimizer=True, optimizer_cudagraphs=True)
    with pytest.raises(ValueError, match="require CUDA"):
        main(
            input=str(tmp_path / "missing"), output=str(tmp_path / "train"), device="cpu", compile_optimizer=True, optimizer_cudagraphs=True
        )
    assert not list(tmp_path.iterdir())
