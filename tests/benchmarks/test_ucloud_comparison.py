"""Safety and real CPU round-trip checks for the external UCloud launch harness."""

import csv
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


@pytest.mark.parametrize("int8_variant", ["quant_int8", "quant_int8_combined"])
def test_launch_topology_and_paired_plan(harness, config, int8_variant):
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
    config["variants"].append(int8_variant)
    compare.validate(config)
    config["gpus"] = 2
    with pytest.raises(ValueError, match="DDP is unsupported"):
        compare.validate(config)


def test_combined_int8_plan_isolates_each_added_option(harness):
    compare, _ = harness
    source = Path(__file__).resolve().parents[2] / "dev" / "ucloud" / "combined-int8.json"
    config = compare.validate(json.loads(source.read_text()))
    runs = compare.plan(config)
    assert [run["name"] for run in runs] == [
        f"{variant}_seed42"
        for variant in (
            "quant_eager",
            "master_eager",
            "quant_compile_model",
            "quant_compile_both",
            "quant_float_combined",
            "quant_int8_combined",
        )
    ]
    assert runs[3]["options"] == {**runs[2]["options"], "compile_optimizer": True}
    assert runs[4]["options"] == {**runs[3]["options"], "cuda_prefetch": True}
    assert runs[5]["options"] == {**runs[4]["options"], "quantized_training": True}
    assert all(run["branch"] == "quant" for run in runs[2:])
    assert compare.uses_quantized_training(config)
    config["variants"].remove("quant_int8_combined")
    assert not compare.uses_quantized_training(config)


@pytest.mark.parametrize("variant", ["quant_compile_both", "quant_float_combined", "quant_int8_combined"])
def test_combined_worker_routes_training_and_loader_options(harness, config, monkeypatch, variant):
    import torch

    import mini_trainer.train as training

    compare, worker = harness
    config.update(gpus=1, seeds=[42], variants=["master_eager", "quant_eager", variant])
    captured = {}
    monkeypatch.setattr(worker, "preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "instrument", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(training, "main", lambda **kwargs: captured.update(kwargs))
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    worker.train(config, f"{variant}_seed42")
    expected = compare.VARIANTS[variant][1]
    assert captured["compile"] and captured["compile_optimizer"]
    assert captured.get("quantized_training", False) == expected.get("quantized_training", False)
    assert captured["dataloader_builder_kwargs"].get("cuda_prefetch", False) == expected.get("cuda_prefetch", False)
    assert "cuda_prefetch" not in captured
    assert captured["ema"] is False


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


@pytest.mark.parametrize("qualification", [False, True, "full_taxonomy"])
def test_parquet_preparation_training_and_reload_cpu(harness, config, monkeypatch, qualification):
    """Actual hierarchy, frozen Parquet splits and EfficientNet; no downloads/GPU."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import torch
    from PIL import Image

    import mini_trainer.train as training
    from mini_trainer.hierarchical.integration import HierarchicalBuilder
    from mini_trainer.logging import MultiLogger
    from mini_trainer.modeling import Classifier

    compare, worker = harness
    config.update(gpus=1, global_batch_size=4, epochs=1, size=32, seeds=[42], num_workers_per_rank=0)
    if qualification:
        config["qualification"] = {"seed": 42, "train": 4, "validation": 2, "test": 2}
        config["figures"] = False

        def reject_figures(*args, **kwargs):
            pytest.fail("Figure generation must be bypassed for this experiment")

        # Register restoration before the worker replaces the process-local hook.
        monkeypatch.setattr(MultiLogger, "figures", MultiLogger.figures)
        monkeypatch.setattr(MultiLogger, "confusion_matrix", reject_figures)
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
    if qualification == "full_taxonomy":
        config["full_taxonomy"] = True

        def single_species_sample(settings):
            path = root / "qualification.parquet"
            pq.write_table(pa.Table.from_pylist(rows[:8]), path)
            return str(path)

        monkeypatch.setattr(worker, "qualification_parquet", single_species_sample)
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
            patch.setattr(worker, "instrument", lambda *args: None)
            patch.setattr(worker, "preflight", lambda *args, **kwargs: None)
            patch.setattr(training, "main", lambda **kwargs: captured.update(kwargs))
            worker.train(config, "quant_eager_seed42")
        assert captured["name"] == "model"
        assert captured["ema"] is False
        captured.update(device="cpu", dtype="float32")
        training.main(**captured)
        directory = root / "runs" / "quant_eager_seed42" / "model"
        assert (directory / "weights" / "checkpoint_last.pth").is_file()
        with (directory / "logs" / "summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["type"] for row in rows] == ["train", "eval"]
        assert all("loss/lvl2" in row for row in rows)
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
        replay = importlib.import_module("replay_validation").replay
        destination = root / "replay"
        destination.mkdir()
        report = replay(
            config,
            index,
            json.loads((root / "class_spec.json").read_text()),
            str(directory / "weights" / "best.pt"),
            precision="fp32",
            prefetch=False,
            output=destination,
            device="cpu",
        )
        assert report["parameters_finite"]
        assert report["images"] == dataset["counts"]["validation"]
        batches = [json.loads(line) for line in (destination / "batches.jsonl").read_text().splitlines()]
        assert all(batch["finite"]["input"] and all(batch["finite"]["outputs"]) for batch in batches)
        assert len(batches[0]["losses"]) == 3
    finally:
        torch.set_num_threads(old_threads)


def test_loss_audit_distinguishes_validation_nan_and_incomplete_logs(harness, tmp_path):
    compare, _ = harness
    path = tmp_path / "summary.csv"
    header = "epoch,type,loss,loss/lvl0,loss/lvl1,loss/lvl2\n"
    rows = ["0,train,3,1,1,1\n", "0,eval,3,1,1,1\n", "1,train,3,1,1,1\n", "1,eval,3,1,1,1\n"]
    path.write_text(header + "".join(rows))
    assert compare.audit_losses(path, 2) == []
    rows[1] = "0,eval,nan,nan,nan,nan\n"
    path.write_text(header + "".join(rows))
    issues = compare.audit_losses(path, 2)
    assert len(issues) == 4
    assert all("epoch=0 phase=eval" in issue for issue in issues)
    path.write_text(header + rows[0])
    assert len(compare.audit_losses(path, 2)) == 3
    path.unlink()
    assert compare.audit_losses(path, 2)


def test_validation_nan_excluded_from_successful_pairs(harness, config, monkeypatch):
    compare, _ = harness
    config.update(seeds=[42], variants=["master_eager", "quant_eager"], epochs=1, require_finite_losses=True)
    compare.validate(config)
    root = Path(config["output"])
    (root / "runs").mkdir(parents=True)
    source = root / "comparison.json"
    compare.write_json(source, config)
    Path(config["parquet"]).write_bytes(b"fixture")
    compare.write_json(root / "dataset.json", {"parquet_sha256": compare.digest(config["parquet"])})
    compare.write_json(root / "prepared.json", {"dataset.json": compare.digest(root / "dataset.json")})

    def execute(argv, *args):
        model = root / "runs" / argv[-1] / "model"
        (model / "weights").mkdir(parents=True)
        (model / "weights" / "best.pt").write_bytes(b"checkpoint")
        (model / "logs").mkdir()
        loss = "nan" if argv[-1].startswith("quant") else "3"
        (model / "logs" / "summary.csv").write_text(
            f"epoch,type,loss,loss/lvl0,loss/lvl1,loss/lvl2\n0,train,3,1,1,1\n0,eval,{loss},1,1,1\n"
        )
        return 0

    monkeypatch.setattr(compare, "execute", execute)
    monkeypatch.setattr("sys.argv", ["compare.py", str(source), "--stage", "train"])
    with pytest.raises(SystemExit, match="Some runs failed"):
        compare.main()
    result = json.loads((root / "runs" / "quant_eager_seed42" / "result.json").read_text())
    assert result["returncode"] == 0
    assert result["status"] == "invalid_metrics"
    assert result["loss_issues"] == ["epoch=0 phase=eval loss='nan'"]
    assert Path(result["checkpoint"]).is_file()
    assert [row["name"] for row in json.loads((root / "paired.json").read_text())] == ["master_eager_seed42"]


def test_replay_accepts_original_manifest_and_rejects_changed_index(harness, tmp_path, monkeypatch):
    compare, _ = harness
    module = importlib.import_module("replay_validation")
    root = tmp_path / "old"
    run = root / "runs" / "quant_prefetch_seed42"
    run.mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    compare.write_json(run / "result.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": compare.digest(checkpoint)})
    compare.write_json(root / "comparison.json", {"gpus": 1})
    for name in ("data_index.json", "class_spec.json"):
        compare.write_json(root / name, {})
    # Original manifests hash these artifacts, but do not hash comparison.json.
    compare.write_json(root / "prepared.json", {name: compare.digest(root / name) for name in ("data_index.json", "class_spec.json")})
    captured = []
    monkeypatch.setattr(module, "replay", lambda *args, **kwargs: captured.append((args, kwargs)))
    destination = tmp_path / "replay"
    monkeypatch.setattr("sys.argv", ["replay_validation.py", str(root), run.name, "--precision", "fp16", "--output", str(destination)])
    module.main()
    assert captured[0][1]["precision"] == "fp16"
    assert captured[0][1]["prefetch"] is False
    assert (destination / "source.json").is_file()
    (root / "data_index.json").write_text("changed")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        module.main()


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
    excluded = Path(config["output"]) / "excluded.parquet"
    excluded.write_bytes(Path(sample).read_bytes())
    excluded_names = {row["filename"] for row in pq.read_table(excluded).to_pylist()}
    config.update(exclude_qualification=str(excluded), exclude_qualification_sha256=compare.digest(excluded))
    worker.qualification_parquet(config)
    assert not excluded_names.intersection(row["filename"] for row in pq.read_table(sample).to_pylist())
    assert json.loads((Path(config["output"]) / "selection.json").read_text())["sha256"] == compare.digest(sample)
    expected_hash = config["exclude_qualification_sha256"]
    config["exclude_qualification_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Exclusion sample changed"):
        worker.qualification_parquet(config)
    config["exclude_qualification_sha256"] = expected_hash
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


def test_scaling_plan_and_trial_generation(harness, config, tmp_path):
    compare, worker = harness
    scaling = importlib.import_module("scaling")
    config.update(
        mode="scaling", gpus=8, global_batch_size=256, seeds=[42], variants=["quant_compile_model"], full_taxonomy=True, wandb=True
    )
    config["environments"].pop("master")
    compare.validate(config)
    assert compare.branches(config) == ["quant"]
    run = compare.plan(config)[0]
    assert run["options"] == {"compile": True}
    assert "--nproc-per-node=8" in compare.command(config, run, "config.json")
    base = tmp_path / "base.json"
    compare.write_json(base, config)
    derived = scaling.trial(base, tmp_path / "next.json", tmp_path / "next", 64, 3)
    assert derived["global_batch_size"] == 512
    assert derived["reuse_preparation"] == config["output"]
    logging = worker.configure_wandb(config, run["name"])
    other = worker.configure_wandb(derived, run["name"])
    assert len(logging["logger_cls_extra_kwargs"]) == len(logging["logger_cls"]) == 2
    assert logging["logger_cls_extra_kwargs"][1]["run_id"] != other["logger_cls_extra_kwargs"][1]["run_id"]


def test_reused_preparation_integrity_and_shared_deadline(harness, config, tmp_path):
    compare, _ = harness
    source = Path(config["output"])
    source.mkdir()
    Path(config["parquet"]).write_bytes(b"source data")
    compare.write_json(source / "comparison.json", config)
    compare.write_json(source / "dataset.json", {"parquet_sha256": compare.digest(config["parquet"])})
    (source / "class_spec.json").write_text("{}")
    compare.write_json(source / "prepared.json", {name: compare.digest(source / name) for name in ("dataset.json", "class_spec.json")})
    deadline = time.time() + 50
    compare.write_json(source / "budget.json", {"deadline": deadline})
    derived = dict(config, mode="scaling", reuse_preparation=str(source), output=str(tmp_path / "next"), budget_seconds=1800)
    target = Path(derived["output"])
    target.mkdir()
    with compare.execution_budget(derived, "prepare") as inherited:
        assert inherited == deadline
        compare.reuse_preparation(derived, target)
    assert (target / "class_spec.json").read_text() == "{}"
    assert not (target / "budget.json").exists()
    (source / "class_spec.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        compare.reuse_preparation(derived, target)


def test_timed_loader_and_rank_aggregation(harness, tmp_path):
    compare, worker = harness
    scaling = importlib.import_module("scaling")
    batches = [([1, 2], [0, 1]), ([3], [0])]
    loader = worker.TimedLoader(batches)
    assert list(loader) == batches
    assert loader.samples == 3 and loader.wait_seconds >= 0 and len(loader) == 2
    compare.write_json(tmp_path / "comparison.json", {"gpus": 2, "global_batch_size": 64})
    compare.write_json(tmp_path / "environment-quant.json", {"gpu_memory_bytes": [1000, 1000]})
    run = tmp_path / "runs" / "example"
    run.mkdir(parents=True)
    compare.write_json(run / "result.json", {"status": "completed"})
    for rank in range(2):
        phases = [
            dict(phase="train", epoch=e, samples=128, seconds=s, loader_wait_seconds=1, max_memory_reserved=700)
            for e, s in enumerate([20, 4 + rank, 4 + rank])
        ]
        (run / f"phases-rank{rank}.jsonl").write_text("\n".join(map(json.dumps, phases)))
    result = scaling.report(tmp_path)[0]
    assert result["warm_images_per_second"] == 256 / 5
    assert result["peak_reserved_fraction"] == 0.7
    assert result["all_ranks_recorded"]


def test_resume_hook_verifies_restored_state_before_updates(harness, monkeypatch, tmp_path):
    import torch

    import mini_trainer.train as training

    _, worker = harness
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    scaler = torch.amp.GradScaler("cpu")
    objects = dict(model=model, optimizer=optimizer, lr_scheduler=scheduler, scaler=scaler)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(dict(epoch=1, **{k: v.state_dict() for k, v in objects.items()}), checkpoint)
    monkeypatch.setattr(training, "train", lambda **kwargs: kwargs["weight_store_rate"])
    worker.configure_training_checks({"checkpoint": str(checkpoint), "resume_epoch": 2}, tmp_path)
    assert training.train(start_epoch=2, **objects) == 1
    assert json.loads((tmp_path / "restore-rank0.json").read_text())["model_optimizer_scheduler_scaler_restored"]
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(AssertionError):
        training.train(start_epoch=2, **objects)


def test_scaling_workers_warm_steps_and_separate_storage(harness, tmp_path):
    compare, _ = harness
    scaling = importlib.import_module("scaling")
    cfg = json.loads((Path(__file__).resolve().parents[2] / "dev/ucloud/ddp.json").read_text())
    cfg["output"] = str(tmp_path / "baseline")
    Path(cfg["output"]).mkdir()
    compare.write_json(Path(cfg["output"]) / "prepared.json", {"qualification.parquet": "a" * 64})
    base = tmp_path / "base.json"
    compare.write_json(base, cfg)
    derived = scaling.trial(base, tmp_path / "large.json", tmp_path / "large", 256, 3, workers=8)
    assert derived["epochs"] == 5
    assert derived["num_workers_per_rank"] == 8
    storage = scaling.trial(base, tmp_path / "storage.json", tmp_path / "storage", 128, 3, workers=16, storage=True)
    assert "reuse_preparation" not in storage
    assert storage["epochs"] == 1 and storage["timeout_seconds"] == 600
    assert storage["qualification"]["train"] == 262144
    assert storage["exclude_qualification_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="num_workers"):
        scaling.trial(base, tmp_path / "invalid.json", tmp_path / "invalid", 64, 3, workers=-1)


def test_timing_windows_survive_completed_chunks(harness):
    _, worker = harness
    saved, synchronized = [], []
    loader = worker.TimedLoader([([1, 2], [0, 1])] * 35, synchronize=lambda: synchronized.append(True), on_window=saved.append)
    assert len(list(loader)) == 35
    assert [w["steps"] for w in saved] == [32, 3]
    assert [w["samples"] for w in saved] == [64, 6]
    assert len(synchronized) == 2
    assert sum(w["loader_wait_seconds"] for w in saved) <= loader.wait_seconds
