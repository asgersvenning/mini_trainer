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


def test_screening_runs_without_tuning_and_rejects_changed_plan(tmp_path, monkeypatch):
    config = {**study.DEFAULTS, "screening": True, "seeds": [42], "epochs": 10}
    write_json(tmp_path / "qualified.json", {"batch_size": 512, "attempts": []})
    monkeypatch.setattr(study, "verify", lambda root: dict(config))
    monkeypatch.setattr(study.signal, "signal", lambda *args: None)
    monkeypatch.setattr(study, "summarize", lambda root: None)
    launched = []
    monkeypatch.setattr(study, "queue", lambda root, runs, *args: launched.extend(runs))
    monkeypatch.setattr(study.sys, "argv", ["study", "run", str(tmp_path)])
    study.main()
    assert {run["variant"] for run in launched} == set(study.VARIANTS)
    assert all(run.get("screening") and run["seed"] == 42 and run["epochs"] == 10 for run in launched)
    assert len({(run["lr"], run["weight_decay"]) for run in launched}) == 1
    assert not (tmp_path / "tuning.json").exists()
    write_json(tmp_path / "plan.json", [])
    with pytest.raises(ValueError, match="Screening plan differs"):
        study.main()
    monkeypatch.setattr(study.sys, "argv", ["study", "tune", str(tmp_path)])
    with pytest.raises(ValueError, match="fixed shared hyperparameters"):
        study.main()


def test_factorial_detects_pure_interaction_with_zero_marginal_effects():
    selected = {"muon": {"lr": 0.003, "weight_decay": 0.001}}
    rows = study.main_runs({**study.DEFAULTS, "seeds": [42]}, selected)
    for row in rows:
        n, r, e = int(row["normalized"]), int(row["regularization"]), int(row["loss"] == "emla")
        value = 0.5 + 0.08 * (n - 0.5) * (r - 0.5) * (e - 0.5)
        row.update(split="validation", macro_recall=value, tail_recall=value, nll=1 - value)
    effects = study.factorial_contrasts(rows)
    marginal = [x for x in effects if not x["condition"] and x["metric"] == "macro_recall"]
    assert len(marginal) == 7
    assert all(x["difference"] == pytest.approx(0) for x in marginal if len(x["factors"]) < 3)
    assert next(x["difference"] for x in marginal if len(x["factors"]) == 3) == pytest.approx(0.08)
    nr = [x for x in effects if x["factors"] == ["normalization", "regularization"] and x["condition"]]
    assert next(x["difference"] for x in nr if x["metric"] == "macro_recall" and x["condition"] == {"emla": 1}) == pytest.approx(0.04)
    assert next(x["difference"] for x in nr if x["metric"] == "macro_recall" and x["condition"] == {"emla": 0}) == pytest.approx(-0.04)
    assert study.factorial_contrasts([r for r in rows if r["variant"] != "full"]) == []
    next(r for r in rows if r["variant"] == "full")["split"] = "test"
    with pytest.raises(ValueError, match="differ in split"):
        study.factorial_contrasts(rows)
