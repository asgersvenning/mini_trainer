"""Safety and real CPU round-trip checks for the external UCloud launch harness."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "dev" / "ucloud"))
    return importlib.import_module("compare"), importlib.import_module("worker")


@pytest.fixture
def config(tmp_path):
    source = Path(__file__).resolve().parents[2] / "dev" / "ucloud" / "comparison.json"
    value = json.loads(source.read_text())
    value.update(output=str(tmp_path / "output with spaces"), parquet=str(tmp_path / "data.parquet"))
    return value


def test_launch_topology_and_paired_plan(harness, config):
    compare, _ = harness
    compare.validate(config)
    runs = compare.plan(config)
    assert runs == compare.plan(config)
    assert len(runs) == 18
    for seed in config["seeds"]:
        assert sum(r["seed"] == seed for r in runs) == 6
    argv = compare.command(config, runs[0], Path(config["output"]) / "comparison.json")
    assert "--nproc-per-node=4" in argv
    assert argv[-3] == str(Path(config["output"]) / "comparison.json")
    config["gpus"] = 1
    assert "torch.distributed.run" not in compare.command(config, runs[0], "config.json")
    config["variants"].append("quant_int8")
    compare.validate(config)
    config["gpus"] = 2
    with pytest.raises(ValueError, match="DDP is unsupported"):
        compare.validate(config)


def test_invalid_batch_and_duplicate_seeds(harness, config):
    compare, _ = harness
    config["global_batch_size"] = 63
    with pytest.raises(ValueError, match="divide evenly"):
        compare.validate(config)
    config["global_batch_size"] = 64
    config["seeds"] = [42, 42]
    with pytest.raises(ValueError, match="Duplicate seeds"):
        compare.validate(config)


def test_summary_preserves_failure_and_pairs_seed(harness, config):
    compare, _ = harness
    root = Path(config["output"])
    for name, seconds, status in [
        ("master_eager_seed42", 10, "completed"),
        ("quant_eager_seed42", 5, "completed"),
        ("quant_combined_seed42", 1, "failed"),
    ]:
        directory = root / "runs" / name
        directory.mkdir(parents=True)
        compare.write_json(
            directory / "result.json",
            {
                "name": name,
                "seed": 42,
                "status": status,
                "wall_seconds": seconds,
            },
        )
    compare.summary(root)
    assert "failed" in (root / "comparison.csv").read_text()
    paired = json.loads((root / "paired.json").read_text())
    assert len(paired) == 2
    assert next(row for row in paired if row["name"] == "quant_eager_seed42")["wall_speedup_vs_master"] == 2


def test_controller_freezes_inputs_and_never_retrains_completed_runs(harness, config, monkeypatch, tmp_path):
    compare, _ = harness
    config.update(seeds=[42], variants=["master_eager", "quant_eager"])
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps(config))
    Path(config["parquet"]).write_bytes(b"fixture parquet identity")
    root = Path(config["output"])
    launched = []

    def execute(argv, log, cwd, timeout):
        if argv[2] == "preflight":
            compare.write_json(root / f"environment-{argv[-1]}.json", {"dependencies": {}, "python_version": "3.12"})
        elif argv[2] == "prepare":
            for filename in ("class_spec.json", "data_index.json", "preprocessing.txt", "initial_seed42.pt"):
                (root / filename).write_text("fixture")
            compare.write_json(root / "dataset.json", {"parquet_sha256": compare.digest(config["parquet"])})
        else:
            name = argv[-1]
            launched.append(name)
            weights = root / "runs" / name / "model" / "weights"
            weights.mkdir(parents=True)
            (weights / "best.pt").write_bytes(b"fixture checkpoint")
        return 0

    monkeypatch.setattr(compare, "execute", execute)
    for stage in ("prepare", "train", "train"):
        monkeypatch.setattr("sys.argv", ["compare.py", str(source), "--stage", stage])
        compare.main()
    assert len(launched) == 2
    assert len(json.loads((root / "paired.json").read_text())) == 2
    (root / "data_index.json").write_text("changed")
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        compare.main()


def test_parquet_preparation_training_and_reload_cpu(harness, config, monkeypatch):
    """Actual hierarchy, frozen Parquet splits and EfficientNet; no downloads/GPU."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import torch
    from PIL import Image

    import mini_trainer.train as training
    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.modeling import Classifier

    compare, worker = harness
    config.update(gpus=1, global_batch_size=4, epochs=1, size=32, seeds=[42], num_workers_per_rank=0)
    root = Path(config["output"])
    root.mkdir()
    rows = []
    for species in ("10", "11"):
        for i, split in enumerate(("2", "2", "2", "2", "1", "1", "0", "0")):
            filename = f"{i}.png"
            path = Path(config["parquet"]).parent / "images" / species / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 40), (i * 20, 90, 100)).save(path)
            rows.append(
                dict(
                    filename=filename,
                    set=split,
                    speciesKey=species,
                    genusKey="20",
                    familyKey="30",
                    orderKey="40",
                    classKey="50",
                    phylumKey="60",
                    kingdomKey="70",
                )
            )
    pq.write_table(pa.Table.from_pylist(rows), config["parquet"])
    build_model = HierarchicalBuilder.build_model

    def offline_build(**kwargs):
        return build_model(**kwargs, model_args={"pretrained": False})

    old_threads = torch.get_num_threads()
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    torch.set_num_threads(1)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(HierarchicalBuilder, "build_model", staticmethod(offline_build))
            worker.prepare(config)
        dataset = json.loads((root / "dataset.json").read_text())
        assert dataset["counts"] == {"train": 8, "validation": 4, "test": 4}
        assert dataset["num_classes"] == [2, 1, 1]
        initial, _ = Classifier.build(weights=str(root / "initial_seed42.pt"), model_args={"pretrained": False})
        initial_state = initial.state_dict()
        del initial
        captured = {}
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, "is_available", lambda: True)
            patch.setattr(torch.cuda, "device_count", lambda: 1)
            patch.setattr(worker, "instrument", lambda _: None)
            patch.setattr(worker, "preflight", lambda *args, **kwargs: None)
            patch.setattr(training, "main", lambda **kwargs: captured.update(kwargs))
            worker.train(config, "quant_eager_seed42")
        assert captured["name"] == "model"
        assert captured["ema"] is False
        captured.update(device="cpu", dtype="float32")
        training.main(**captured)
        directory = root / "runs" / "quant_eager_seed42" / "model"
        assert (directory / "weights" / "checkpoint_last.pth").is_file()
        reloaded, preprocess = Classifier.build(weights=str(directory / "weights" / "best.pt"), model_args={"pretrained": False})
        reloaded.eval()
        with torch.inference_mode():
            outputs = reloaded(preprocess(torch.zeros(2, 3, 32, 32, dtype=torch.uint8)))
        assert [tuple(output.shape) for output in outputs] == [(2, 2), (2, 1), (2, 1)]
        assert all(torch.isfinite(output).all() for output in outputs)
        assert any(
            not torch.equal(initial_state[key], value)
            for key, value in reloaded.state_dict().items()
            if key in initial_state and isinstance(value, torch.Tensor) and value.is_floating_point()
        )
    finally:
        torch.set_num_threads(old_threads)
