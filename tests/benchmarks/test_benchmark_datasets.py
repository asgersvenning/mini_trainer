import csv
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


@pytest.mark.parametrize("fine_tune", [False, True])
def test_summary_preserves_failures_and_unmeasured_fields(tmp_path, fine_tune):
    path = tmp_path / "failed"
    path.mkdir()
    (path / "report.json").write_text(
        json.dumps({"status": "failed", "device": "cuda:0", "dtype": "float16", "quantized_training": True, "fine_tune": fine_tune})
    )
    summary = summarize(tmp_path)
    assert "| failed | failed | cuda:0 / float16 | requested | — | — | — | — |" in summary
    assert "CPU results do not validate GPU" in summary
    assert (" | frozen |" if fine_tune else " | trainable |") in summary
    assert "No reports produced" in summarize(Path(tmp_path / "missing"))


@pytest.mark.parametrize("mode", ["qt", "qt-large-batch", "qt-cudagraphs", "qt-optimizer-cudagraphs"])
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
    prefix = {"qt-cudagraphs": "mnist-cudagraphs", "qt-optimizer-cudagraphs": "mnist-optimizer-cudagraphs"}.get(mode, "mnist-large-batch")
    profiles = (
        ("synthetic-float", "synthetic-int8")
        if mode == "qt"
        else tuple(f"{prefix}-{precision}-seed{seed}" for seed in (42, 43, 44) for precision in ("float", "int8"))
    )
    for profile in profiles:
        report = json.loads((output / profile / "report.json").read_text())
        assert report["status"] == "failed"
        assert report["error"]["exit_code"] == 134
        assert report["quantized_training"] == ("int8" in profile)
        assert "test_accuracy" not in report
        arguments = report["arguments"]
        if mode in ("qt-cudagraphs", "qt-optimizer-cudagraphs"):
            assert arguments[arguments.index("--compile-mode") + 1] == "reduce-overhead"
            assert "--compile" in arguments and "--compile-optimizer" in arguments
        else:
            assert "--compile-mode" not in arguments
        assert ("--optimizer-cudagraphs" in arguments) == (mode == "qt-optimizer-cudagraphs")
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


@pytest.mark.parametrize("fine_tune", [False, True])
def test_efficientnet_flat_and_hierarchical_share_blair_splits(tmp_path, monkeypatch, fine_tune):
    import numpy as np
    import torch

    from dev.benchmarks.run import run
    from mini_trainer.builders import BaseBuilder
    from mini_trainer.modeling import classification_module

    captured = {}
    build_optimizer = BaseBuilder.build_optimizer

    def inspect_optimizer(model, *args, **kwargs):
        head_ids = {id(p) for p in classification_module(model).parameters()}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        assert backbone and all(p.requires_grad != fine_tune for p in backbone)
        if fine_tune:
            captured["backbone"] = [(p, p.detach().clone()) for p in backbone]
            captured["head"] = [(p, p.detach().clone()) for p in classification_module(model).parameters() if p.requires_grad]
        return build_optimizer(model, *args, **kwargs)

    monkeypatch.setattr(BaseBuilder, "build_optimizer", inspect_optimizer)

    root = tmp_path / "images"
    make_dataset(root)
    spec = {
        "labels": {"a": ["species_a", "parent_a"], "b": ["species_b", "parent_b"]},
        "num_classes": [2, 2],
        "cls2idx": {"0": {"species_b": 0, "species_a": 1}, "1": {"parent_a": 0, "parent_b": 1}},
    }
    spec_path = tmp_path / "taxonomy.json"
    spec_path.write_text(json.dumps(spec))
    reports = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for head in ("flat", "hierarchical"):
            reports.append(
                run(
                    tmp_path / head,
                    epochs=1,
                    dataset="blair",
                    data_root=root,
                    class_spec=spec_path,
                    backbone="efficientnet_v2_s",
                    head=head,
                    hidden=True,
                    normalized=True,
                    image_size=32,
                    batch_size=4,
                    cache_workers=0,
                    fine_tune=fine_tune,
                )
            )
            if fine_tune:
                assert all(torch.equal(p, initial) and p.grad is None for p, initial in captured["backbone"])
                assert any(not torch.equal(p, initial) for p, initial in captured["head"])
                captured.clear()
    finally:
        torch.set_num_threads(previous_threads)
    with (tmp_path / "hierarchical/training/logs/summary.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    assert [row["type"] for row in summaries] == ["train", "eval"]
    assert all(float(row[level_loss]) > 0 for row in summaries for level_loss in ("loss/lvl0", "loss/lvl1"))
    flat, hierarchical = reports
    assert flat["dataset_manifest_sha256"] == hierarchical["dataset_manifest_sha256"]
    assert flat["class_mapping"] == spec["cls2idx"]["0"]
    assert hierarchical["class_mapping"]["0"] == flat["class_mapping"]
    assert flat["hidden_width"] == hierarchical["hidden_width"] == 1280
    assert all(report["normalized"] and not report["pretrained"] for report in reports)
    assert all(report["fine_tune"] == fine_tune and report["backbone_training_mode"] == "train" for report in reports)
    with np.load(tmp_path / "flat/predictions.npz") as a, np.load(tmp_path / "hierarchical/predictions.npz") as b:
        np.testing.assert_array_equal(a["paths"], b["paths"])
        np.testing.assert_array_equal(a["labels"], b["labels"])
        assert a["scores"].shape == b["scores"].shape == (2, 2)
        assert b["scores_1"].shape == (2, 2)
    indices = [json.loads((tmp_path / head / "train_index.json").read_text()) for head in ("flat", "hierarchical")]
    assert indices[0]["path"] == indices[1]["path"]
    assert indices[0]["split"] == indices[1]["split"]
    assert indices[0]["class"] == [labels[0] for labels in indices[1]["class"]]


def test_representative_profile_retains_training_and_quality_failures(tmp_path):
    import os
    import shlex
    import subprocess
    import sys

    runner = tmp_path / "python-wrapper"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
        'if [[ "$1" == "-m" && "$2" == "dev.benchmarks.run" ]]; then exit 134; fi\n' + f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    runner.chmod(0o755)
    output = tmp_path / "reports"
    result = subprocess.run(
        ["bash", "dev/check-benchmarks.sh", "qt-efficientnet", str(output)],
        env={
            **os.environ,
            "BENCHMARK_PYTHON": str(runner),
            "BENCHMARK_METRICS_PYTHON": str(runner),
            "BENCHMARK_DATA_ROOT": str(tmp_path),
            "BLAIR_CLASS_SPEC": str(tmp_path / "spec.json"),
            "BENCHMARK_HEAD": "hierarchical",
            "BENCHMARK_TRAINING_MODE": "frozen",
            "BENCHMARK_SEEDS": "43",
            "BENCHMARK_EPOCHS": "20",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    for precision in ("float", "int8"):
        report = json.loads((output / f"blair-hierarchical-frozen-{precision}-seed43/report.json").read_text())
        assert report["status"] == "failed" and report["error"]["exit_code"] == 134
        args = report["arguments"]
        for option, value in (
            ("--backbone", "efficientnet_v2_s"),
            ("--head", "hierarchical"),
            ("--hidden", "symmetric"),
            ("--epochs", "20"),
            ("--seed", "43"),
            ("--dtype", "bfloat16"),
        ):
            assert args[args.index(option) + 1] == value
        assert "--fine-tune" in args and "--pretrained" in args and "--normalized" in args
        assert ("--quantized-training" in args) == (precision == "int8")
    quality = json.loads((output / "blair-hierarchical-frozen-seed43-quality/report.json").read_text())
    assert quality["status"] == "failed"
    summary = (output / "summary.md").read_text()
    assert "Paired held-out quality" in summary
    assert "blair-hierarchical-frozen-seed43-quality | failed" in summary


def test_summary_renders_paired_metrics_without_treating_them_as_training(tmp_path):
    output = tmp_path / "pair"
    output.mkdir()
    (output / "report.json").write_text(
        json.dumps(
            {
                "status": "evaluated",
                "models": {},
                "levels": [
                    {
                        "name": "parent",
                        "candidate_minus_baseline": {
                            "f1": -0.051,
                            "recall": 0.02,
                            "precision": None,
                            "coverage": 0,
                            "theilU": -0.01,
                        },
                    }
                ],
            }
        )
    )
    summary = summarize(tmp_path)
    assert "| pair / parent | evaluated | -5.100 | +2.000 | undefined | +0.000 | -1.000 |" in summary
    assert "| pair | evaluated | ? / ?" not in summary
    assert "not acceptance gates" in summary


@pytest.mark.parametrize("seeds", [" ", "42 42", "-1"])
def test_representative_profile_rejects_invalid_seeds_before_output(tmp_path, seeds):
    import os
    import subprocess
    import sys

    output = tmp_path / "reports"
    result = subprocess.run(
        ["bash", "dev/check-benchmarks.sh", "qt-efficientnet", str(output)],
        env={
            **os.environ,
            "BENCHMARK_PYTHON": sys.executable,
            "BENCHMARK_METRICS_PYTHON": sys.executable,
            "BENCHMARK_DATA_ROOT": str(tmp_path),
            "BLAIR_CLASS_SPEC": str(tmp_path / "spec.json"),
            "BENCHMARK_SEEDS": seeds,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and "BENCHMARK_SEEDS" in result.stderr
    assert not output.exists()
