"""Shared-root dispatch must neither duplicate training nor select partial tuning."""

import argparse
import fcntl
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from publication.experiments.training_ablations import campaign, study, support_sensitivity
from publication.experiments.training_ablations.data import write_json


def finish(attempt, stage, wall_seconds=1):
    """Write the artifacts a successful training or evaluation stage leaves behind."""
    if stage == "train":
        (attempt / "model/weights").mkdir(parents=True)
        for name in ["model/weights/last.pt", "initialization.json", "parameter_groups.json"]:
            (attempt / name).write_text("{}")
        write_json(attempt / "train.json", {"wall_seconds": wall_seconds})
    else:
        for name in ["evaluation.json", "predictions.npz"]:
            (attempt / name).write_text("{}")


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
    assert study.factorial_contrasts(rows, metrics=("tail_recall",)) == [x for x in effects if x["metric"] == "tail_recall"]
    assert study.factorial_contrasts([r for r in rows if r["variant"] != "full"]) == []
    next(r for r in rows if r["variant"] == "full")["split"] = "test"
    with pytest.raises(ValueError, match="differ in split"):
        study.factorial_contrasts(rows)


def test_shared_queue_skips_busy_work_and_does_not_duplicate(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    trained = []

    def child(root, attempt, stage, *args):
        if stage == "train":
            run = json.loads((attempt / "run.json").read_text())
            trained.append(run["id"])
            if run["id"] == "long":
                entered.set()
                assert release.wait(10)
        finish(attempt, stage)

    monkeypatch.setattr(study, "child", child)
    runs = [{"id": name} for name in ["long", "short", "next"]]
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(study.queue, tmp_path, runs, {}, ["0"], time.monotonic() + 100, False, True)
        try:
            assert entered.wait(5)
            study.queue(tmp_path, runs, {}, ["0"], time.monotonic() + 100, False, shared=True)
            assert trained == ["long", "short", "next"]
        finally:
            release.set()
        first.result()
    assert sorted(trained) == ["long", "next", "short"]
    assert all(study.latest_complete(tmp_path, run) is not None for run in runs)


def test_hierarchy_factorial_preserves_conditional_effects_and_rejects_mismatches():
    rows = study.main_runs({**study.DEFAULTS, "hierarchy": True, "seeds": [42]}, {"muon": {"lr": 0.003, "weight_decay": 0.001}})
    for row in rows:
        h, r = int(any(row["rank_weights"][1:])), int(row["regularization"])
        row.update(split="validation", score=0.5 + 0.02 * h + 0.01 * r + 0.04 * h * r)
    effects = study.factorial_contrasts(rows, metrics=("score",))
    assert len(effects) == 7
    interaction = next(e for e in effects if e["factors"] == ["hierarchy", "regularization"])
    assert interaction["difference"] == pytest.approx(0.04)
    for value in [0, 1]:
        effect = next(e for e in effects if e["factors"] == ["hierarchy"] and e["condition"] == {"regularization": value})
        assert effect["difference"] == pytest.approx(0.02 + 0.04 * value)
    assert study.factorial_contrasts(rows[:-1], metrics=("score",)) == []
    with pytest.raises(ValueError, match="Duplicate factorial"):
        study.factorial_contrasts([*rows, rows[0]], metrics=("score",))
    row = next(r for r in rows if any(r["rank_weights"][1:]))
    row["rank_weights"] = [0.5, 0.25, 0.25]
    with pytest.raises(ValueError, match="rank weights"):
        study.factorial_contrasts(rows, metrics=("score",))


def test_campaign_summary_uses_frozen_subset_and_summarizes_each_metric(tmp_path):
    config = {**study.DEFAULTS, "hierarchy": True, "seeds": [42, 43]}
    write_json(tmp_path / "config.json", config)
    runs = []
    for seed, delta in [(42, 0.01), (43, -0.02)]:
        for variant, hierarchy in [("species_regularized", False), ("hierarchy_regularized", True)]:
            run = {
                "id": f"{variant}_seed{seed}",
                "variant": variant,
                "seed": seed,
                "rank_weights": [1.0, 0.0, 0.0] if not hierarchy else [1 / 3] * 3,
                "normalized": True,
                "regularization": True,
                "loss": "emla",
                "split": "validation",
                "optimizer": "adamw",
                "hidden": 0,
                "lr": 0.001,
                "weight_decay": 0.01,
                "epochs": 10,
            }
            runs.append(run)
            attempt = tmp_path / "runs" / run["id"] / "attempt-000"
            attempt.mkdir(parents=True)
            write_json(attempt / "run.json", run)
            write_json(
                attempt / "evaluation.json",
                {"macro_recall": 0.5 + delta * hierarchy, "tail_recall": 0.3 + delta * hierarchy, "nll": 1.0 - delta * hierarchy},
            )
            write_json(attempt / "train.json", {"wall_seconds": 1})
            write_json(attempt / "complete.json", {})
    write_json(tmp_path / "plan.json", runs)

    study.summarize(tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed_main_runs"] == summary["expected_main_runs"] == 4
    paired = json.loads((tmp_path / "paired-summary.json").read_text())
    macro = next(row for row in paired if row["contrast"] == "hierarchy - species (regularized)" and row["metric"] == "macro_recall")
    tail = next(row for row in paired if row["contrast"] == "hierarchy - species (regularized)" and row["metric"] == "tail_recall")
    nll = next(row for row in paired if row["contrast"] == "hierarchy - species (regularized)" and row["metric"] == "nll")
    assert (macro["n"], macro["mean"]) == (2, pytest.approx(-0.005))
    assert tail["mean"] == pytest.approx(-0.005)
    assert nll["mean"] == pytest.approx(0.005)


def test_support_sensitivity_pairs_seeds_and_uses_difference_in_differences():
    full, limited = {}, {}
    for seed, effect in [(42, 0.02), (43, -0.01)]:
        for variant, base, interaction in [("species_regularized", 0.5, 0), ("hierarchy_regularized", 0.52, 0.03)]:
            full[(variant, seed)] = {"macro_recall": base, "run": {}, "split": "validation"}
            limited[(variant, seed)] = {"macro_recall": base - effect + interaction, "run": {}, "split": "validation"}
    rows = support_sensitivity.effect_rows(full, limited)
    by_seed = {row["seed"]: row for row in rows}
    assert by_seed[42]["species_support_effect"] == pytest.approx(-0.02)
    assert by_seed[42]["hierarchy_support_effect"] == pytest.approx(0.01)
    assert by_seed[42]["hierarchy_x_support_difference_in_differences"] == pytest.approx(0.03)
    assert by_seed[43]["hierarchy_x_support_difference_in_differences"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="matching seeds"):
        support_sensitivity.effect_rows(full, {key: value for key, value in limited.items() if key[1] == 42})


def test_support_sensitivity_rejects_changed_held_out_rows():
    import pandas as pd

    rows = pd.DataFrame(
        {
            "split": ["validation", "train"],
            "sample_id": ["image-a", "train-a"],
            "label": [0, 0],
            "speciesKey": ["s0", "s0"],
            "genusKey": ["g0", "g0"],
            "familyKey": ["f0", "f0"],
        }
    )
    assert support_sensitivity.validate_evaluation_rows(rows, rows.copy(), "validation") == 1
    changed = rows.copy()
    changed.loc[0, "sample_id"] = "image-b"
    with pytest.raises(ValueError, match="samples differ"):
        support_sensitivity.validate_evaluation_rows(rows, changed, "validation")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("prepared", "source", "publication/script.py"), "changed", "validation samples differ"),
        (("prepared", "source", "mini_trainer/train.py"), "changed", "Core training code"),
        (("prepared", "environment", "torch"), "changed", "Core training code"),
        (("classes", "num_classes"), 2, "Class vocabulary"),
        (("analysis_provenance", "analysis_sha256"), "changed", "different code"),
        (("endpoints", (support_sensitivity.VARIANTS[0], 42), "run", "lr"), 1.0, "Training recipes"),
    ],
)
def test_support_comparison_accepts_only_publication_script_differences(path, value, message):
    import pandas as pd

    def cohort():
        run = {"seed": 42, **dict.fromkeys(support_sensitivity.RUN_FIELDS, 0)}
        return {
            "classes": {"num_classes": 1},
            "prepared": {
                "source": {"mini_trainer/train.py": "a", "publication/script.py": "b"},
                "environment": {"torch": "2", "cuda": "13"},
            },
            "analysis_provenance": {"analysis_sha256": "c", "contrast_code_sha256": "d"},
            "plans": [run],
            "endpoints": {(variant, 42): {"run": dict(run), "split": "validation"} for variant in support_sensitivity.VARIANTS},
            "samples": pd.DataFrame(columns=["split", "sample_id", "label", "speciesKey", "genusKey", "familyKey"]),
        }

    limited = cohort()
    target = limited
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    # Empty held-out rows make an otherwise accepted comparison stop at the sample check.
    with pytest.raises(ValueError, match=message):
        support_sensitivity.compare(cohort(), limited, None)


def test_interrupted_attempts_resume_but_failures_need_retry(tmp_path, monkeypatch):
    errors = iter([TimeoutError("deadline"), RuntimeError("broken")])

    def child(*args):
        raise next(errors)

    monkeypatch.setattr(study, "child", child)
    run = {"id": "run"}
    with pytest.raises(TimeoutError):
        study.execute(tmp_path, run, {}, "0", 100)
    with pytest.raises(RuntimeError, match="broken"):  # Resumed without --retry
        study.execute(tmp_path, run, {}, "0", 100)
    with pytest.raises(RuntimeError, match="--retry"):
        study.execute(tmp_path, run, {}, "0", 100)


def test_shared_queue_drains_past_failures_and_leaves_runs_that_cannot_finish(tmp_path, monkeypatch):
    trained = []

    def child(root, attempt, stage, *args):
        run = json.loads((attempt / "run.json").read_text())["id"]
        if stage == "train":
            trained.append(run)
            if run == "broken":
                raise RuntimeError("broken run")
        finish(attempt, stage, wall_seconds=50)

    monkeypatch.setattr(study, "child", child)
    runs = [{"id": name} for name in ["broken", "first", "second"]]
    # After the first 50 s run, the second would need more than the remaining minute.
    with pytest.raises(RuntimeError, match="broken run"):
        study.queue(tmp_path, runs, {}, ["0"], time.monotonic() + 60, False, shared=True)
    assert trained == ["broken", "first"]


def test_only_one_worker_prepares_and_qualifies_a_study(tmp_path, monkeypatch):
    calls = []

    def run_study(command, root, *args):
        calls.append(command)
        root.mkdir(parents=True, exist_ok=True)
        (root / {"prepare": "prepared.json", "qualify": "qualified.json"}[command]).write_text("{}")
        return 0

    monkeypatch.setattr(campaign, "run_study", run_study)
    cohort = tmp_path / "study-name"
    cohort.mkdir()
    with (cohort / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not campaign.ready(cohort, 1)  # Another worker is preparing
    assert campaign.ready(cohort, 1) and campaign.ready(cohort, 1)
    assert calls == ["prepare", "qualify"]


@pytest.mark.parametrize("name", campaign.STUDIES)
def test_campaign_studies_plan_each_cell_once_per_seed_with_flat_species_objectives(name):
    config = {**study.DEFAULTS, **json.loads((campaign.CONFIGS / f"{name}.json").read_text())}
    runs = study.main_runs(config, {"muon": {}})
    factors = ["hidden", "normalized", "regularization", "loss", "rank_weights"]
    cells = {json.dumps([run.get(factor) for factor in factors]) for run in runs}
    assert len(cells) * len(config["seeds"]) == len(runs) and config["seeds"] == [42, 43, 44]
    assert all(run.get("rank_weights") in (None, [1 / 3] * 3) for run in runs)
