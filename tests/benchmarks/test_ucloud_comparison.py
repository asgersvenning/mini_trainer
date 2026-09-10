"""Safety and real CPU round-trip checks for the external UCloud launch harness."""

import importlib
import json
import os
import subprocess
import sys
import time
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


@pytest.mark.parametrize("qualification", [False, True])
def test_parquet_preparation_training_and_reload_cpu(harness, config, monkeypatch, qualification):
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
    if qualification:
        config["qualification"] = {"seed": 42, "train": 4, "validation": 2, "test": 2}
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
        assert dataset["counts"] == (
            {"train": 4, "validation": 2, "test": 2} if qualification else {"train": 8, "validation": 4, "test": 4}
        )
        assert dataset["num_classes"] == [2, 1, 1]
        assert dataset["scope"] == ("qualification_subset" if qualification else "full_dataset")
        index = json.loads((root / "data_index.json").read_text())
        assert all(Path(path).is_file() for path in index["path"])
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


def test_qualification_sampling_is_bounded_paired_and_preserves_splits(harness, config):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    compare, worker = harness
    config.update(gpus=1, global_batch_size=4, qualification={"seed": 42, "train": 8, "validation": 4, "test": 2})
    compare.validate(config)
    Path(config["output"]).mkdir()
    rows = []
    for split in ("0", "1", "2", "3", "excluded"):
        for i in range(30):
            rows.append(
                dict(
                    filename=f"{split}-{i}.png",
                    set=split,
                    **{
                        key: str(i % 3)
                        for key in (
                            "speciesKey",
                            "genusKey",
                            "familyKey",
                            "orderKey",
                            "classKey",
                            "phylumKey",
                            "kingdomKey",
                        )
                    },
                )
            )
    pq.write_table(pa.Table.from_pylist(rows), config["parquet"], row_group_size=7)
    source_hash = compare.digest(config["parquet"])
    sample = worker.qualification_parquet(config)
    first = pq.read_table(sample).to_pylist()
    assert len(first) == 14
    assert sum(row["set"] == "0" for row in first) == 2
    assert sum(row["set"] == "1" for row in first) == 4
    assert sum(row["set"] in ("2", "3") for row in first) == 8
    assert all(row in rows for row in first)
    assert len({row["filename"] for row in first}) == 14
    worker.qualification_parquet(config)
    assert pq.read_table(sample).to_pylist() == first
    config["qualification"]["seed"] = 43
    worker.qualification_parquet(config)
    assert pq.read_table(sample).to_pylist() != first
    assert compare.digest(config["parquet"]) == source_hash
    config["qualification"]["test"] = 31
    with pytest.raises(ValueError, match="Not enough rows"):
        worker.qualification_parquet(config)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"seed": 42, "train": 8, "validation": 4, "test": 0},
        {"seed": 42, "train": 3, "validation": 4, "test": 2},
        {"seed": True, "train": 8, "validation": 4, "test": 2},
    ],
)
def test_invalid_qualification_rejected_before_preparation(harness, config, settings):
    compare, _ = harness
    config.update(gpus=1, global_batch_size=4, qualification=settings)
    with pytest.raises(ValueError, match="qualification"):
        compare.validate(config)


def test_incomplete_preparation_has_actionable_error(harness, config, monkeypatch):
    compare, _ = harness
    source = Path(config["output"]).parent / "comparison.json"
    source.write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", ["compare.py", str(source), "--stage", "train"])
    with pytest.raises(ValueError, match="Preparation is incomplete"):
        compare.main()


def test_interruption_retains_failed_run_and_has_clean_exit(harness, config, monkeypatch):
    compare, _ = harness
    root = Path(config["output"])
    (root / "runs").mkdir(parents=True)
    config.update(seeds=[42], variants=["master_eager", "quant_eager"])
    compare.validate(config)
    source = root / "comparison.json"
    source.write_text(json.dumps(config))
    Path(config["parquet"]).write_bytes(b"fixture")
    compare.write_json(root / "dataset.json", {"parquet_sha256": compare.digest(config["parquet"])})
    compare.write_json(root / "prepared.json", {"dataset.json": compare.digest(root / "dataset.json")})

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(compare, "execute", interrupt)
    monkeypatch.setattr("sys.argv", ["compare.py", str(source), "--stage", "train"])
    with pytest.raises(SystemExit) as exc:
        compare.main()
    assert exc.value.code == 130
    result = json.loads(next((root / "runs").glob("*/result.json")).read_text())
    assert result["status"] == "interrupted"
    assert result["wall_seconds"] >= 0
    assert json.loads((root / "paired.json").read_text()) == []
    with pytest.raises(RuntimeError, match="Incomplete/failed run exists"):
        compare.main()


def test_budget_stops_real_worker_and_persists_across_stages(harness, config):
    compare, _ = harness
    config["budget_seconds"] = 1
    root = Path(config["output"])
    root.mkdir()
    log = root / "worker.log"
    with pytest.raises(compare.BudgetExceeded):
        with compare.execution_budget(config, "prepare") as deadline:
            compare.write_json(root / "budget.json", {"deadline": deadline})
            compare.execute(
                [sys.executable, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(30)"],
                log,
                root,
                30,
            )
    pid = int(log.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert json.loads((root / "budget.json").read_text())["deadline"] <= time.time()
    with pytest.raises(compare.BudgetExceeded):
        with compare.execution_budget(config, "train"):
            pytest.fail("Expired budget must not launch more training")
    with compare.execution_budget(config, "summary"):
        pass


@pytest.mark.parametrize("fail_sync", [False, True])
def test_fresh_job_setup_uses_pins_and_stops_on_install_failure(tmp_path, fail_sync):
    """Run the actual shell flow with a fake installer; no package/network mutation."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['SETUP_TEST_LOG'], 'a') as f: f.write(json.dumps(args)+'\\n')\n"
        "if args[0]=='export':\n"
        " root=pathlib.Path(args[args.index('--directory')+1])\n"
        " assert (root/'uv.lock').is_file() and (root/'pyproject.toml').is_file()\n"
        " pathlib.Path(args[args.index('--output-file')+1]).write_text('fixture==1\\n')\n"
        "if args[0]=='venv':\n"
        " target=pathlib.Path(args[-1])/'bin'/'python'\n"
        " target.parent.mkdir(parents=True)\n"
        " target.symlink_to(sys.executable)\n"
        "if args[:2]==['pip','sync'] and os.environ['SETUP_TEST_FAIL']=='1': sys.exit(7)\n"
    )
    uv.chmod(0o755)
    parquet = tmp_path / "mounted dataset.parquet"
    parquet.write_bytes(b"fixture")
    work = tmp_path / "work with spaces"
    log = tmp_path / "commands.jsonl"
    setup = Path(__file__).resolve().parents[2] / "dev/ucloud/setup.sh"
    result = subprocess.run(
        ["bash", str(setup), str(parquet)],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "MT_WORK_ROOT": str(work),
            "MT_CONFIG": str(work / "qualification.json"),
            "MT_TORCH_BACKEND": "cu130",
            "MT_REPO_URL": "https://github.com/asgersvenning/mini_trainer.git",
            "SETUP_TEST_LOG": str(log),
            "SETUP_TEST_FAIL": str(int(fail_sync)),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    if fail_sync:
        assert result.returncode == 7
        assert not (work / "qualification.json").exists()
        assert not any(command[:2] == ["pip", "install"] for command in commands)
        return
    assert result.returncode == 0, result.stderr
    config = json.loads((work / "qualification.json").read_text())
    assert config["parquet"] == str(parquet)
    assert config["gpus"] == 1 and config["budget_seconds"] == 1800
    syncs = [command for command in commands if command[:2] == ["pip", "sync"]]
    installs = [command for command in commands if command[:2] == ["pip", "install"]]
    assert len(syncs) == len(installs) == 2
    assert all("--require-hashes" in command and "--index-strategy" in command for command in syncs)
    assert all("--no-deps" in command for command in installs)
    for command, branch in zip(installs, ("master", "quant"), strict=True):
        assert command[-1].endswith("@" + config["environments"][branch]["commit"])
        assert config["environments"][branch]["python"] == str(work / "venvs" / f"mt-{branch}" / "bin/python")
