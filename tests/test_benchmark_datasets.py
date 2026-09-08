import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from dev.benchmarks.datasets import prepare_real
from dev.benchmarks.summarize import summarize


def make_dataset(root):
    for label_id, label in enumerate(("a", "b")):
        for index in range(6):
            split = "train" if index < 5 else "test"
            path = root / split / label / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (index + 10 * label_id, 0, 0)).save(path)


def test_real_split_is_reproducible_and_groups_duplicates(tmp_path):
    root = tmp_path / "images"
    make_dataset(root)
    shutil.copy(root / "train/a/0.png", root / "train/a/duplicate.png")
    shutil.copy(root / "test/b/5.png", root / "train/b/leaked.png")
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        output.mkdir()
    first, spec = prepare_real(root, outputs[0], name="mnist", seed=42)
    second, _ = prepare_real(root, outputs[1], name="mnist", seed=42)
    assert first == second
    assert spec["cls2idx"] == {"a": 0, "b": 1}
    assert first["excluded_train_test_duplicates"] == ["train/b/leaked.png"]
    records = first["records"]
    hashes = {split: {r["sha256"] for r in records if r["split"] == split} for split in ("train", "val", "test")}
    assert hashes["train"].isdisjoint(hashes["val"] | hashes["test"])
    assert hashes["val"].isdisjoint(hashes["test"])
    assert sum(r["split"] == "test" for r in records) == 2
    assert (root / "train/b/leaked.png").exists()  # Inventory filtering never modifies source data.


def test_blair_requires_explicit_covering_taxonomy(tmp_path):
    root = tmp_path / "images"
    make_dataset(root)
    with pytest.raises(ValueError, match="explicit reviewed"):
        prepare_real(root, tmp_path, name="blair", seed=42)
    spec = {
        "labels": {"a": ["species_a", "parent"], "b": ["species_b", "parent"]},
        "num_classes": [2, 1],
        "cls2idx": {"0": {"species_a": 0, "species_b": 1}, "1": {"parent": 0}},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    manifest, restored = prepare_real(root, tmp_path, name="blair", seed=42, class_spec=path)
    assert restored == spec
    assert {tuple(record["targets"]) for record in manifest["records"]} == {(0, 0), (1, 0)}
    del spec["labels"]["b"]
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="exactly cover"):
        prepare_real(root, tmp_path, name="blair", seed=42, class_spec=path)


def test_summary_preserves_failures_and_unmeasured_fields(tmp_path):
    path = tmp_path / "failed"
    path.mkdir()
    (path / "report.json").write_text(json.dumps({"status": "failed", "device": "cuda:0", "dtype": "float16", "quantized_training": True}))
    summary = summarize(tmp_path)
    assert "| failed | failed | cuda:0 / float16 | requested | — | — | — | — |" in summary
    assert "CPU results do not validate GPU" in summary
    assert "No reports produced" in summarize(Path(tmp_path / "missing"))


@pytest.mark.parametrize("mode", ["qt", "qt-large-batch"])
def test_shared_harness_records_process_failures(tmp_path, mode):
    import os
    import shlex
    import subprocess
    import sys

    runner = tmp_path / "python-wrapper"
    runner.write_text(
        '#!/usr/bin/env bash\nif [[ "$1" == "-m" && "$2" == "dev.benchmarks.run" ]]; then exit 134; fi\n'
        + f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    runner.chmod(0o755)
    output = tmp_path / "reports"
    result = subprocess.run(
        ["bash", "dev/check-benchmarks.sh", mode, str(output)],
        env={**os.environ, "BENCHMARK_PYTHON": str(runner), "BENCHMARK_DATA_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    profiles = (
        ("synthetic-float", "synthetic-int8")
        if mode == "qt"
        else tuple(f"mnist-large-batch-{precision}-seed{seed}" for seed in (42, 43, 44) for precision in ("float", "int8"))
    )
    for profile in profiles:
        report = json.loads((output / profile / "report.json").read_text())
        assert report["status"] == "failed"
        assert report["error"]["exit_code"] == 134
        assert report["quantized_training"] == ("int8" in profile)
        assert "test_accuracy" not in report
    assert "requested" in (output / "summary.md").read_text()


def test_benchmark_retains_cuda_peaks_across_phase_resets(tmp_path):
    import os

    import torch
    from torch.utils.data import DataLoader

    from dev.benchmarks.performance import BenchmarkLogger

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to validate cross-phase CUDA peaks")
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    loader = DataLoader(torch.arange(4), batch_size=2)
    logger = BenchmarkLogger(
        train_loader=loader,
        val_loader=loader,
        epochs=1,
        output=str(tmp_path),
        name="measure",
        logger_cls=[],
        measurement_device="cuda:0",
    )
    large = torch.empty(8 * 1024 * 1024, device="cuda:0")
    earlier_peak = torch.cuda.max_memory_allocated()
    logger.update(epoch=0, type="train")
    del large
    logger.start_timing()
    later = torch.empty(4 * 1024 * 1024, device="cuda:0")
    phase_peak = torch.cuda.max_memory_allocated()
    logger.step()
    del later
    logger.step()
    logger.stop_timing()
    logger.update(epoch=0, type="eval")
    logger.start_timing()
    logger.stop_timing()
    logger.finish()
    report = json.loads((tmp_path / "measure/logs/performance.json").read_text())
    assert report["peak_cuda_allocated_bytes"] >= earlier_peak
    assert report["peak_cuda_allocated_bytes"] > torch.cuda.max_memory_allocated()
    assert [phase["phase"] for phase in report["phases"]] == ["train", "eval"]
    assert report["phases"][0]["peak_cuda_allocated_bytes"] >= phase_peak
    assert all(phase["seconds"] >= 0 for phase in report["phases"])


def test_summary_marks_legacy_cuda_readings_unverified(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    report = {"status": "completed", "peak_cuda_allocated_bytes": 64 * 2**20}
    (path / "report.json").write_text(json.dumps(report))
    assert "unverified" in summarize(tmp_path).splitlines()[4]
    report["peak_cuda_memory_scope"] = "maximum across logger resets"
    (path / "report.json").write_text(json.dumps(report))
    assert "64.00" in summarize(tmp_path).splitlines()[4]


def test_summary_later_epoch_median_requires_timing_scope(tmp_path):
    report = {
        "status": "completed",
        "training_wall_seconds": 123,
        "phase_measurements": [
            {"epoch": 0, "phase": "train", "seconds": 80},
            {"epoch": 1, "phase": "train", "seconds": 20},
            {"epoch": 2, "phase": "train", "seconds": 2},
            {"epoch": 3, "phase": "train", "seconds": 4},
            {"epoch": 3, "phase": "eval", "seconds": 10},
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert "| unverified | 123.00s |" in summarize(tmp_path)
    report["phase_measurement_scope"] = "synchronized batch loop"
    path.write_text(json.dumps(report))
    assert "| 3.000s | 123.00s |" in summarize(tmp_path)
    report["phase_measurements"] = report["phase_measurements"][:2]
    path.write_text(json.dumps(report))
    assert "| — | 123.00s |" in summarize(tmp_path)
