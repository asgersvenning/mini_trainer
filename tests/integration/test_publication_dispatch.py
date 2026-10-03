"""Shared-root dispatch must neither duplicate training nor select partial tuning."""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from publication.experiments.training_ablations import study
from publication.experiments.training_ablations.data import write_json


@pytest.mark.parametrize("count", [1, 2, 4, 8, 11])
def test_shards_cover_plans_without_overlap(count):
    selected = {name: {"lr": 0.001, "weight_decay": 0.01} for name in ["muon", "adamw"]}
    for runs in [study.tuning_runs(study.DEFAULTS), study.main_runs(study.DEFAULTS, selected)]:
        shards = [study.shard_runs(runs, study.parse_shard(f"{index}/{count}")) for index in range(count)]
        ids = [run["id"] for shard in shards for run in shard]
        assert len(ids) == len(set(ids)) == len(runs)
        assert set(ids) == {run["id"] for run in runs}


@pytest.mark.parametrize("value", ["-1/4", "4/4", "0/0", "0", "a/2", "0/2/3"])
def test_invalid_shard(value):
    with pytest.raises(argparse.ArgumentTypeError):
        study.parse_shard(value)


def test_overlapping_launch_cannot_create_another_attempt(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def child(*args):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("deliberate interrupted worker")

    monkeypatch.setattr(study, "child", child)
    run = {"id": "shared"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(study.execute, tmp_path, run, {}, "0", 100)
        try:
            assert entered.wait(5)
            with pytest.raises(BlockingIOError):
                study.execute(tmp_path, run, {}, "0", 100, retry=True)
            assert len(list((tmp_path / "runs/shared").glob("attempt-*"))) == 1
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="deliberate"):
            first.result()
    # The failed worker releases the lock; retry creates its own retained attempt.
    with pytest.raises(RuntimeError, match="deliberate"):
        study.execute(tmp_path, run, {}, "0", 100, retry=True)
    assert len(list((tmp_path / "runs/shared").glob("attempt-*"))) == 2


def test_tuning_finalizes_only_complete_verified_plan(tmp_path):
    runs = study.tuning_runs(study.DEFAULTS)
    for index, run in enumerate(runs):
        attempt = tmp_path / "runs" / run["id"] / "attempt-000"
        attempt.mkdir(parents=True)
        write_json(attempt / "run.json", run)
        write_json(attempt / "evaluation.json", {"macro_recall": index / 10, "nll": 1})
        if index < len(runs) - 1:
            write_json(attempt / "complete.json", {name: study.digest(attempt / name) for name in ["run.json", "evaluation.json"]})
    assert not study.finalize_tuning(tmp_path, study.DEFAULTS)
    assert not (tmp_path / "tuning.json").exists()
    assert not (tmp_path / "plan.json").exists()
    write_json(attempt / "complete.json", {name: study.digest(attempt / name) for name in ["run.json", "evaluation.json"]})
    assert study.finalize_tuning(tmp_path, study.DEFAULTS)
    selected = json.loads((tmp_path / "tuning.json").read_text())
    assert json.loads((tmp_path / "plan.json").read_text()) == study.main_runs(study.DEFAULTS, selected)
    write_json(attempt / "evaluation.json", {"macro_recall": 1, "nll": 0})
    with pytest.raises(ValueError, match="Completed artifact changed"):
        study.finalize_tuning(tmp_path, study.DEFAULTS)


def test_qualification_descends_extended_batch_ladder(tmp_path, monkeypatch):
    seen = []

    def oom(root, runs, config, *args):
        seen.append(config["batch_size"])
        raise study.CUDAOutOfMemory("synthetic capacity failure")

    monkeypatch.setattr(study, "queue", oom)
    with pytest.raises(study.CUDAOutOfMemory):
        study.qualify(tmp_path, {**study.DEFAULTS, "batch_size": 512}, ["0"], 100, False)
    assert seen == [512, 256, 128, 64, 32]


def test_cli_shards_finalize_after_last_tuning_and_main_shards(tmp_path, monkeypatch):
    config = dict(study.DEFAULTS)
    write_json(tmp_path / "qualified.json", {"batch_size": 128, "attempts": []})
    monkeypatch.setattr(study, "verify", lambda root: dict(config))
    monkeypatch.setattr(study.signal, "signal", lambda *args: None)
    launched, summaries = [], []
    monkeypatch.setattr(study, "summarize", lambda root: summaries.append(root))

    def queue(root, runs, *args):
        launched.append([run["id"] for run in runs])
        for run in runs:
            attempt = root / "runs" / run["id"] / "attempt-000"
            attempt.mkdir(parents=True)
            write_json(attempt / "run.json", run)
            write_json(attempt / "evaluation.json", {"macro_recall": 0.5, "nll": 1})
            write_json(attempt / "complete.json", {name: study.digest(attempt / name) for name in ["run.json", "evaluation.json"]})

    monkeypatch.setattr(study, "queue", queue)
    for command in ["tune", "run"]:
        for index in range(2):
            monkeypatch.setattr(study.sys, "argv", ["study", command, str(tmp_path), "--shard", f"{index}/2"])
            study.main()
            if command == "tune":
                assert (tmp_path / "plan.json").exists() == bool(index)
            if command == "tune" or index == 0:
                assert not summaries
    assert len(summaries) == 1
    assert set(launched[0]).isdisjoint(launched[1])
    assert set(launched[2]).isdisjoint(launched[3])
    assert sum(map(len, launched[:2])) == 8
    assert sum(map(len, launched[2:])) == len(study.VARIANTS) * len(config["seeds"])
