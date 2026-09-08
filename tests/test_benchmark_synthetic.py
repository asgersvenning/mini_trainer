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
        first = run(tmp_path / "first")
        second = run(tmp_path / "second")
    finally:
        torch.set_num_threads(threads)
    assert first["test_accuracy"] == second["test_accuracy"] == 1.0
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

    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(tmp_path / "failed"), "--device", "cuda:0"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: None)
    with pytest.raises(RuntimeError, match="no accessible CUDA device"):
        main()
    report = json.loads((tmp_path / "failed/report.json").read_text())
    assert report["status"] == "failed"
    assert report["device"] == "cuda:0"
    assert report["error"]["type"] == "RuntimeError"
    assert "test_accuracy" not in report
