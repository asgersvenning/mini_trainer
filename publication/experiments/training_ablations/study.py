"""Run with python -m publication.experiments.training_ablations.study."""

import argparse
import fcntl
import importlib.metadata
import itertools
import json
import math
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from .data import digest, prepare_data, write_json

REPO = Path(__file__).resolve().parents[3]
STOP = threading.Event()
DEFAULTS = {
    "screening": False,
    "targeted": False,
    "data_index": None,
    "source_metadata": None,
    "log_gates": False,
    "hierarchy": False,
    "cohort_families": [],
    "ranks": ["species", "genus", "family"],
    "train_image_budget": None,
    "train_support_cap": None,
    "train_support_seed": 0,
    "variants": None,
    "species": 512,
    "selection_seed": 20261003,
    "size": 384,
    "seeds": [42, 43, 44],
    "epochs": 30,
    "tuning_epochs": 10,
    "batch_size": 128,
    "workers": 8,
    "threads": 1,
    "dtype": "float16",
    "wandb": True,
    "project": "mini-trainer-ablations",
    "entity": "asvenning",
    "device": "cuda",
}
FULL = {"normalized": True, "hidden": True, "regularization": True, "loss": "emla", "optimizer": "muon"}
VARIANTS = {
    "full": {},
    "no_normalization": {"normalized": False},
    "no_regularization": {"regularization": False},
    "ce": {"loss": "ce"},
    "no_normalization_no_regularization": {"normalized": False, "regularization": False},
    "no_normalization_ce": {"normalized": False, "loss": "ce"},
    "no_regularization_ce": {"regularization": False, "loss": "ce"},
    "core_reference": {"normalized": False, "regularization": False, "loss": "ce"},
    "fixed_adjustment": {"loss": "fixed"},
}


def extensions(ranks):
    """Cells beyond the original screens, selected through an explicit "variants" list.

    Species-only objectives use the flat head, gradient-identical to a hierarchical head with weight
    only on species. Hierarchical objectives use mini_trainer's default of weight one at every rank.
    """
    hierarchy = {"rank_weights": [1.0] * ranks}
    return {
        "standard": {"hidden": False, "normalized": False, "regularization": False, "loss": "ce"},
        "no_normalization_fixed": {"normalized": False, "loss": "fixed"},
        "no_regularization_fixed": {"regularization": False, "loss": "fixed"},
        "no_normalization_no_regularization_fixed": {"normalized": False, "regularization": False, "loss": "fixed"},
        "hierarchy_regularized": hierarchy,
        "hierarchy_unregularized": {**hierarchy, "regularization": False},
        "hierarchy_ce_regularized": {**hierarchy, "loss": "ce"},
        "hierarchy_ce_unregularized": {**hierarchy, "regularization": False, "loss": "ce"},
        "hierarchy_fixed_regularized": {**hierarchy, "loss": "fixed"},
        "hierarchy_fixed_unregularized": {**hierarchy, "regularization": False, "loss": "fixed"},
    }


class CUDAOutOfMemory(RuntimeError):
    """Only this failure permits qualification's global batch reduction."""


def source_identity():
    files = [
        *sorted((REPO / "mini_trainer").rglob("*.py")),
        *sorted(Path(__file__).parent.glob("*.py")),
        REPO / "uv.lock",
        REPO / "pyproject.toml",
    ]
    return {str(p.relative_to(REPO)): digest(p) for p in files}


def environment():
    import torch

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
    }


def tuning_runs(config):
    return [
        {
            **FULL,
            "optimizer": opt,
            "lr": lr,
            "weight_decay": wd,
            "seed": 41,
            "epochs": config["tuning_epochs"],
            "tuning": True,
            "id": f"tune_{opt}_{lr}_{wd}",
        }
        for opt, lr, wd in itertools.product(["muon", "adamw"], [0.0003, 0.001], [0.001, 0.01])
    ]


def study_variants(config):
    """The study's cells: an explicit "variants" list, else the campaign's original treatments."""
    requested = config.get("variants")
    if requested is None:
        return campaign_variants(config)
    available = {**campaign_variants(config), **VARIANTS, **extensions(len(config["ranks"]))}
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(available):
        raise ValueError("variants must be a nonempty unique subset of the available treatments")
    variants = {name: available[name] for name in requested}
    if any(v.get("rank_weights") for v in variants.values()) and not (config["hierarchy"] or config["data_index"]):
        raise ValueError("Hierarchical treatments need a cohort with a prepared hierarchy")
    return variants


def main_runs(config, selected):
    runs = []
    variants = study_variants(config)
    for seed in config["seeds"]:
        block = []
        for name, changes in variants.items():
            run = {**FULL, **changes}
            run.update(selected[run["optimizer"]])
            if config.get("targeted"):
                run["targeted"] = True
            if config.get("screening"):
                run["screening"] = True
            block.append({**run, "seed": seed, "epochs": config["epochs"], "variant": name, "id": f"{name}_seed{seed}"})
        if not config.get("targeted"):
            random.Random(seed).shuffle(block)
        runs.extend(block)
    return runs


def campaign_variants(config):
    if config.get("targeted"):
        return {
            **{
                k: VARIANTS[k]
                for k in ["full", "no_normalization", "no_regularization", "no_normalization_no_regularization", "ce", "fixed_adjustment"]
            },
            **{k: v for k, v in hierarchy_variants().items() if k.startswith("hierarchy_")},
        }
    return hierarchy_variants() if config.get("hierarchy") else VARIANTS


def hierarchy_variants():
    """A shared leaf head; only rank supervision and prototype penalty differ."""
    return {
        f"{objective}_{'regularized' if regularization else 'unregularized'}": {
            "rank_weights": weights,
            "regularization": regularization,
        }
        for objective, weights in [("species", [1.0, 0.0, 0.0]), ("hierarchy", [1 / 3] * 3)]
        for regularization in [False, True]
    }


def prepare(config_path, root):
    config = {**DEFAULTS, **json.loads(Path(config_path).read_text())}
    if set(config) - (set(DEFAULTS) | {"parquet", "images", "pretrained"}):
        raise ValueError("Unknown configuration keys")
    if config["hierarchy"] and (not config["screening"] or len(config["cohort_families"]) < 2 or not config["train_image_budget"]):
        raise ValueError("Hierarchy campaign requires screening and explicit complete families")
    if not isinstance(config["screening"], bool):
        raise ValueError("screening must be boolean")
    if config["targeted"] and (not config["data_index"] or not config["screening"] or config["hierarchy"]):
        raise ValueError("Targeted replication requires a corrected index, screening and per-run hierarchy")
    if bool(config.get("parquet")) == bool(config.get("data_index")):
        raise ValueError("Supply exactly one of parquet or data_index")
    for key in ["data_index" if config["data_index"] else "parquet", "images"]:
        config[key] = str(Path(config[key]).expanduser().resolve())
    for key in ["species", "size", "epochs", "tuning_epochs", "batch_size", "threads"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["workers"] < 0 or config["batch_size"] not in [32, 64, 128, 256, 512, 768]:
        raise ValueError("Nonnegative workers and batch size 32, 64, 128, 256, 512 or 768 required")
    if config["train_support_cap"] is not None and (not isinstance(config["train_support_cap"], int) or config["train_support_cap"] < 1):
        raise ValueError("train_support_cap must be a positive integer or null")
    if not isinstance(config["train_support_seed"], int):
        raise ValueError("train_support_seed must be an integer")
    study_variants(config)
    if len(set(config["seeds"])) != len(config["seeds"]) or 41 in config["seeds"]:
        raise ValueError("Main seeds must be unique and distinct from tuning seed 41")
    if config.get("source_metadata"):
        config["source_metadata"] = str(Path(config["source_metadata"]).expanduser().resolve())
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "tuning-plan.json", [] if config["screening"] else tuning_runs(config))
    prepare_data(config, root)
    import torch
    from torchvision.models import EfficientNet_V2_S_Weights

    weights = (
        torch.load(config["pretrained"], map_location="cpu", weights_only=True)
        if config.get("pretrained")
        else EfficientNet_V2_S_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    )
    torch.save(weights, root / "pretrained.pt")
    write_json(
        root / "prepared.json",
        {
            "files": {
                name: digest(root / name)
                for name in [
                    "config.json",
                    "tuning-plan.json",
                    "samples.parquet",
                    "classes.json",
                    "species.csv",
                    "selection.json",
                    "pretrained.pt",
                ]
                + (["source-class-map.csv"] if (root / "source-class-map.csv").exists() else [])
            },
            "source": source_identity(),
            "environment": environment(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "pretrained_enum": "EfficientNet_V2_S_Weights.IMAGENET1K_V1" if not config.get("pretrained") else "operator_supplied",
        },
    )


def verify(root):
    prepared = json.loads((root / "prepared.json").read_text())
    for name, expected in prepared["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    if prepared["source"] != source_identity():
        raise ValueError("Study or trainer source changed; use the prepared revision")
    if prepared["environment"] != environment():
        raise ValueError("Python/package environment changed since preparation")
    return json.loads((root / "config.json").read_text())


def completed(attempt, run):
    if json.loads((attempt / "run.json").read_text()) != run:
        raise ValueError("Run identity changed")
    marker = attempt / "complete.json"
    if not marker.exists():
        return False
    for name, expected in json.loads(marker.read_text()).items():
        if digest(attempt / name) != expected:
            raise ValueError(f"Completed artifact changed: {attempt / name}")
    return True


def latest_complete(root, run):
    attempts = sorted((root / "runs" / run["id"]).glob("attempt-*"))
    if not attempts or not completed(attempts[-1], run):
        raise ValueError(f"Required run is incomplete: {run['id']}")
    return attempts[-1]


def child(root, attempt, stage, device, config, deadline):
    if STOP.is_set():
        raise InterruptedError("Campaign cancelled")
    remaining = deadline - time.monotonic() - 60
    if remaining <= 0:
        raise TimeoutError("Campaign deadline reached; reserve retained for cleanup")
    command = [
        sys.executable,
        "-m",
        "publication.experiments.training_ablations.study",
        "_worker",
        str(root),
        "--attempt",
        str(attempt),
        "--stage",
        stage,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": device, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": str(config["threads"])}
    write_json(attempt / f"{stage}-launch.json", {"command": command, "cwd": str(REPO), "gpu": device, "timeout_seconds": remaining})
    with (attempt / f"{stage}.log").open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                if STOP.is_set():
                    raise InterruptedError("Campaign cancelled")
                remaining = deadline - time.monotonic() - 60
                if remaining <= 0:
                    raise TimeoutError("Campaign deadline reached")
                try:
                    code = process.wait(timeout=min(1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        logfile = attempt / (stage + ".log")
        error = CUDAOutOfMemory if "CUDA out of memory" in logfile.read_text()[-4096:] else RuntimeError
        raise error(f"{stage} exited {code}; see {logfile}")


def predict_run(root, run, output):
    from .training import export_predictions

    shutil.rmtree(output, ignore_errors=True)
    attempt = latest_complete(root, run)
    config = json.loads((attempt / "resolved.json").read_text())
    provenance = {"source": source_identity(), "environment": environment()}
    export_predictions(root, attempt, config, run, output, provenance=provenance)


def interrupted(attempt):
    """No recorded failure (the process died with its node) or stopped by a deadline or signal."""
    failure = attempt / "failure.json"
    return not failure.exists() or json.loads(failure.read_text())["type"] in ("InterruptedError", "TimeoutError")


def expected_seconds(root):
    """Median training time of this study's completed main runs; zero before the first one."""
    records = [p for p in root.glob("runs/*/attempt-*/train.json") if not p.parents[1].name.startswith("qualify_")]
    return statistics.median(json.loads(p.read_text())["wall_seconds"] for p in records) if records else 0


@contextmanager
def run_lock(root, run):
    """Own a run's directory across controllers; BlockingIOError while another holds it."""
    directory = root / "runs" / run["id"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield directory


def execute(root, run, config, device, deadline, retry=False):
    with run_lock(root, run) as directory:
        return execute_locked(root, run, config, device, deadline, retry, directory)


def execute_locked(root, run, config, device, deadline, retry, directory):
    attempts = sorted(directory.glob("attempt-*"))
    attempt = attempts[-1] if attempts else None
    if attempt and completed(attempt, run):
        if json.loads((attempt / "resolved.json").read_text()) != config:
            raise ValueError("Completed run configuration differs")
        return attempt
    # The run lock is held, so an unfinished attempt is not running elsewhere; resume interruptions.
    if attempt and not (retry or interrupted(attempt)):
        raise RuntimeError(f"Incomplete attempt: {attempt}; inspect logs and use --retry")
    if attempt and (attempt / "train.json").exists():
        # Reuse successful training; evaluation is independently restartable.
        if json.loads((attempt / "resolved.json").read_text()) != config:
            raise ValueError("Cannot reuse training with changed configuration")
    else:
        attempt = directory / f"attempt-{len(attempts):03d}"
        attempt.mkdir()
        write_json(attempt / "run.json", run)
        write_json(attempt / "resolved.json", config)
    try:
        if not (attempt / "train.json").exists():
            child(root, attempt, "train", device, config, deadline)
        child(root, attempt, "evaluate", device, config, deadline)
        artifacts = [
            "run.json",
            "resolved.json",
            "train.json",
            "model/weights/last.pt",
            "evaluation.json",
            "predictions.npz",
            "initialization.json",
            "parameter_groups.json",
        ]
        write_json(attempt / "complete.json", {name: digest(attempt / name) for name in artifacts})
        return attempt
    except BaseException as exc:
        write_json(attempt / "failure.json", {"type": type(exc).__name__, "message": str(exc), "time": time.time()})
        raise


def queue(root, runs, config, devices, deadline, retry, shared=False):
    # Deterministic static lanes; paired seed blocks rotate across devices.
    def lane(index):
        results, failures = [], []
        for run in runs if shared else runs[index :: len(devices)]:
            # A shared queue leaves runs that cannot finish in time to controllers with more allocation.
            if shared and deadline - time.monotonic() < 1.25 * expected_seconds(root):
                continue
            try:
                results.append(execute(root, run, config, devices[index], deadline, retry))
            except BlockingIOError:
                if not shared:
                    raise
                # Another controller owns this run. Continue to unclaimed work.
            except (InterruptedError, TimeoutError):
                raise
            except Exception as error:
                if not shared:
                    raise
                # Keep the failed run's evidence for inspection and drain the remaining runs.
                failures.append(error)
        if failures:
            raise failures[0]
        return results

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(lane, index) for index in range(len(devices))]
        results, failures = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(error)
        if failures:
            # Never conceal a non-memory failure behind another lane's OOM.
            raise next((error for error in failures if not isinstance(error, CUDAOutOfMemory)), failures[0])
    return [path for paths in results for path in paths]


def select_tuning(paths):
    winners = {}
    for optimizer in ["muon", "adamw"]:
        candidates = []
        for path in paths:
            run = json.loads((path / "run.json").read_text())
            if run["optimizer"] == optimizer:
                result = json.loads((path / "evaluation.json").read_text())
                candidates.append((-result["macro_recall"], result["nll"], run["lr"], run["weight_decay"]))
        _, _, lr, decay = min(candidates)
        winners[optimizer] = {"lr": lr, "weight_decay": decay}
    return winners


def qualify(root, config, devices, deadline, retry):
    # Full class vocabulary; tiny train/validation sample. Never evaluates test.
    treatments = {name: VARIANTS[name] for name in ["full", "no_normalization", "core_reference", "fixed_adjustment"]}
    if config.get("targeted"):
        treatments["hierarchy_regularized"] = hierarchy_variants()["hierarchy_regularized"]
    if config.get("hierarchy"):
        treatments = {k: v for k, v in hierarchy_variants().items() if v["regularization"]}
    if config.get("variants") is not None:
        # Exercise every planned code path once, on tiny samples, before the main runs.
        treatments = study_variants(config)
    for batch in [v for v in [768, 512, 256, 128, 64, 32] if v <= config["batch_size"]]:
        candidate = {**config, "batch_size": batch}
        runs = [
            {
                **FULL,
                **changes,
                "seed": 40,
                "epochs": 2,
                "lr": 0.003,
                "weight_decay": 0.001,
                "qualification": True,
                "id": f"qualify_b{batch}_{name}",
            }
            for name, changes in treatments.items()
        ]
        try:
            paths = queue(root, runs, candidate, devices, deadline, retry)
        except CUDAOutOfMemory:
            if batch == 32:
                raise
            continue
        hardware = [json.loads((p / "hardware.json").read_text()) for p in paths]
        if len({(h["name"], h["memory_bytes"]) for h in hardware}) != 1:
            raise ValueError("Use equivalent GPUs for the campaign")
        write_json(
            root / "qualified.json", {"batch_size": batch, "hardware": hardware[0], "attempts": [str(p.relative_to(root)) for p in paths]}
        )
        return


def factorial_contrasts(rows, metrics=("macro_recall", "tail_recall", "nll"), factors=None):
    """Equal-cell marginal and conditional finite differences, separately by seed."""
    if any(r.get("targeted") for r in rows) and factors is None:
        if not all(r.get("targeted") for r in rows):
            raise ValueError("Cannot mix targeted and original protocols")
        geometry = [r for r in rows if r.get("rank_weights") is None and r["loss"] == "emla"]
        hierarchy_rows = [
            {**r, "rank_weights": r.get("rank_weights") or [1.0, 0.0, 0.0]} for r in rows if r["normalized"] and r["loss"] == "emla"
        ]
        return factorial_contrasts(geometry, metrics, ("normalization", "regularization")) + factorial_contrasts(
            hierarchy_rows, metrics, ("hierarchy", "regularization")
        )
    hierarchy = any(r.get("rank_weights") is not None for r in rows)
    if hierarchy and any(r.get("rank_weights") is None for r in rows):
        raise ValueError("Cannot mix hierarchical and original factorial protocols")
    factors = factors or (("hierarchy", "regularization") if hierarchy else ("normalization", "regularization", "emla"))
    result = []
    for seed in sorted({row["seed"] for row in rows}):
        block = [r for r in rows if r["seed"] == seed and r["loss"] in ["emla", "ce"]]
        if hierarchy:
            cells = {(int(any(w > 0 for w in r["rank_weights"][1:])), int(r["regularization"])): r for r in block}
            for enabled in [0, 1]:
                weights = {tuple(r["rank_weights"]) for cell, r in cells.items() if cell[0] == enabled}
                if len(weights) > 1:
                    raise ValueError("Factorial cells differ in rank weights within an objective")
        else:
            cells = {
                tuple(
                    {"normalization": int(r["normalized"]), "regularization": int(r["regularization"]), "emla": int(r["loss"] == "emla")}[f]
                    for f in factors
                ): r
                for r in block
            }
        if len(cells) != len(block):
            raise ValueError("Duplicate factorial cell within a seed")
        if len(cells) != 2 ** len(factors):
            continue  # Never fill missing cells or pool seeds to complete a cube.
        for key in ["split", "optimizer", "hidden", "lr", "weight_decay", "epochs", *(["normalized", "loss"] if hierarchy else [])]:
            if len({r[key] for r in block}) != 1:
                raise ValueError(f"Factorial cells differ in {key}")
        for order in range(1, len(factors) + 1):
            for axes in itertools.combinations(range(len(factors)), order):
                other = [i for i in range(len(factors)) if i not in axes]
                contexts = [None, *itertools.product([0, 1], repeat=len(other))] if other else [None]
                for context in contexts:
                    selected = {k: r for k, r in cells.items() if context is None or tuple(k[i] for i in other) == context}
                    divisor = 2 ** len(other) if context is None else 1
                    for metric in metrics:
                        if any(r[metric] is None or not math.isfinite(r[metric]) for r in selected.values()):
                            continue
                        value = sum((-1) ** (order - sum(k[i] for i in axes)) * r[metric] for k, r in selected.items()) / divisor
                        result.append(
                            {
                                "seed": seed,
                                "split": block[0]["split"],
                                "metric": metric,
                                "factors": [factors[i] for i in axes],
                                "condition": {} if context is None else dict(zip([factors[i] for i in other], context)),
                                "difference": value,
                            }
                        )
    return result


def summarize(root):
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = []
    for directory in sorted((root / "runs").glob("*")):
        attempts = sorted(directory.glob("attempt-*"))
        if not attempts:
            continue
        path = attempts[-1]
        run = json.loads((path / "run.json").read_text())
        row = {**run, "attempt": str(path.relative_to(root)), "status": "incomplete"}
        if completed(path, run):
            row["status"] = "complete"
            row.update({k: v for k, v in json.loads((path / "evaluation.json").read_text()).items() if not isinstance(v, list)})
            training = json.loads((path / "train.json").read_text())
            row["training_seconds"] = training["wall_seconds"]
            row["peak_allocated_bytes"] = training.get("peak_allocated_bytes")
            row["images_per_second_including_validation_and_logging"] = training.get("images_per_second_including_validation_and_logging")
        rows.append(row)
    if (root / "plan.json").exists():
        present = {row["id"] for row in rows}
        rows.extend({**run, "status": "not_started"} for run in json.loads((root / "plan.json").read_text()) if run["id"] not in present)
    pd.DataFrame(rows).to_csv(root / "results.csv", index=False)
    main = [row for row in rows if row.get("variant") and row["status"] == "complete"]
    write_json(root / "factorial.json", factorial_contrasts(main))
    variants = study_variants(json.loads((root / "config.json").read_text()))
    contrasts = []
    for seed in sorted({row["seed"] for row in main}):
        block = {row["variant"]: row for row in main if row["seed"] == seed}
        for name in variants:
            if name != "full" and name in block and "full" in block:
                contrasts.append(
                    {
                        "seed": seed,
                        "contrast": f"full - {name}",
                        "macro_recall_difference": block["full"]["macro_recall"] - block[name]["macro_recall"],
                    }
                )
    if json.loads((root / "config.json").read_text()).get("hierarchy"):
        for seed in sorted({row["seed"] for row in main}):
            block = {row["variant"]: row for row in main if row["seed"] == seed}
            for metric in ["macro_recall", "tail_recall", "nll"]:
                effects = {}
                for regularization in ["unregularized", "regularized"]:
                    flat, hierarchical = f"species_{regularization}", f"hierarchy_{regularization}"
                    if flat in block and hierarchical in block:
                        effects[regularization] = block[hierarchical][metric] - block[flat][metric]
                        contrasts.append(
                            {
                                "seed": seed,
                                "contrast": f"hierarchy - species ({regularization})",
                                "metric": metric,
                                "difference": effects[regularization],
                            }
                        )
                if len(effects) == 2:
                    contrasts.append(
                        {
                            "seed": seed,
                            "contrast": "hierarchy x regularization",
                            "metric": metric,
                            "difference": effects["regularized"] - effects["unregularized"],
                        }
                    )
    write_json(root / "paired.json", contrasts)
    paired_groups = {}
    for row in contrasts:
        metric = row.get("metric", "macro_recall")
        value = row.get("difference", row.get("macro_recall_difference"))
        if value is not None:
            paired_groups.setdefault((row["contrast"], metric), []).append(value)
    paired_summary = []
    for (name, metric), values in sorted(paired_groups.items()):
        paired_summary.append(
            {
                "contrast": name,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
            }
        )
    write_json(root / "paired-summary.json", paired_summary)
    if main:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(variants):
            values = [row["macro_recall"] for row in main if row["variant"] == name]
            if values:
                ax.scatter([i] * len(values), values)
                ax.plot(i, np.mean(values), "k_")
        ax.set_xticks(range(len(variants)), list(variants), rotation=35, ha="right")
        ax.set_ylabel("Macro recall (points: individual seeds; split in results.csv)")
        fig.tight_layout()
        fig.savefig(root / "macro-recall.png")
        plt.close(fig)
    planned = [run for run in json.loads((root / "plan.json").read_text()) if run.get("variant")] if (root / "plan.json").exists() else None
    write_json(
        root / "summary.json",
        {
            "completed_main_runs": len(main),
            "expected_main_runs": len(planned)
            if planned is not None
            else len(variants) * len(json.loads((root / "config.json").read_text())["seeds"]),
            "interpretation": "Paired conditional effects; seed variation is not test-sample uncertainty. Incomplete runs remain visible.",
        },
    )


def parse_shard(value):
    try:
        index, count = map(int, value.split("/"))
        if not 0 <= index < count:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Shard must be INDEX/COUNT with 0 <= INDEX < COUNT") from exc
    return index, count


def shard_runs(runs, shard):
    return runs if shard is None else runs[shard[0] :: shard[1]]


def all_complete(root, runs):
    paths = []
    for run in runs:
        attempts = sorted((root / "runs" / run["id"]).glob("attempt-*"))
        if not attempts or not (attempts[-1] / "complete.json").exists():
            return None
        paths.append(latest_complete(root, run))
    return paths


def finalize_tuning(root, config):
    paths = all_complete(root, tuning_runs(config))
    if paths is None:
        return False
    selected = select_tuning(paths)
    write_json(root / "tuning.json", selected)
    write_json(root / "plan.json", main_runs(config, selected))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "qualify", "tune", "run", "status", "summarize", "predict", "_worker"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--devices", default="0", help="Comma-separated equivalent GPU IDs; one process per GPU")
    parser.add_argument("--hours", type=float, default=24, help="Remaining allocation hours, including a cleanup reserve")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--shard", type=parse_shard, help="Zero-based INDEX/COUNT; tune/run on a shared prepared root")
    parser.add_argument("--shared-queue", action="store_true", help="Claim the next free run on a shared filesystem")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--stage", choices=["train", "evaluate"])
    parser.add_argument("--output", type=Path, help="predict: new or partially filled per-run output directory")
    args = parser.parse_args()
    if args.shard is not None and args.command not in ["tune", "run"]:
        parser.error("--shard is supported only for tune and run")
    if args.shared_queue and (args.command not in ["run", "predict"] or args.shard is not None):
        parser.error("--shared-queue requires run or predict and cannot be combined with --shard")
    root = args.root.resolve()
    if args.command == "prepare":
        if not args.config:
            parser.error("prepare requires --config")
        prepare(args.config, root)
        return
    if args.command == "_worker":
        import torch

        from .training import evaluate, train

        config = json.loads((args.attempt / "resolved.json").read_text())
        run = json.loads((args.attempt / "run.json").read_text())
        if config["device"] == "cuda":
            properties = torch.cuda.get_device_properties(0)
            hardware = {"name": properties.name, "memory_bytes": properties.total_memory}
        else:
            hardware = {"name": "cpu", "memory_bytes": 0}
        if (root / "qualified.json").exists() and hardware != json.loads((root / "qualified.json").read_text())["hardware"]:
            raise ValueError("GPU differs from qualified hardware")
        write_json(args.attempt / "hardware.json", hardware)
        (train if args.stage == "train" else evaluate)(root, args.attempt, config, run)
        return
    if args.command == "predict":
        # Prediction needs the prepared data and verified weights, not the preparation-time source.
        prepared = json.loads((root / "prepared.json").read_text())
        for name, expected in prepared["files"].items():
            if digest(root / name) != expected:
                raise ValueError(f"Prepared artifact changed: {name}")
        for run in json.loads((root / "plan.json").read_text()):
            output = args.output / run["id"]
            if (output / "prediction.json").exists():
                continue
            if not args.shared_queue:
                predict_run(root, run, output)
                continue
            # Shared: predict completed runs no other controller holds; others are left for later.
            try:
                with run_lock(root, run) as directory:
                    attempts = sorted(directory.glob("attempt-*"))
                    if attempts and completed(attempts[-1], run) and not (output / "prediction.json").exists():
                        predict_run(root, run, output)
            except BlockingIOError:
                pass
        return
    if args.command == "status":
        for path in sorted((root / "runs").glob("*/attempt-*")):
            print(
                path.relative_to(root),
                "complete" if (path / "complete.json").exists() else "failed" if (path / "failure.json").exists() else "incomplete",
            )
        return
    lock_name = ".controller.lock" if args.shard is None else f".controller-{args.command}-{args.shard[0]}-of-{args.shard[1]}.lock"
    if args.shared_queue:
        lock_name = f".controller-{uuid.uuid4().hex}.lock"
    with (root / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGINT, lambda *_: STOP.set())
        signal.signal(signal.SIGTERM, lambda *_: STOP.set())
        config = verify(root)
        if args.command == "summarize":
            summarize(root)
            return
        devices = args.devices.split(",")
        if len(set(devices)) != len(devices) or not all(devices):
            raise ValueError("Distinct device IDs required")
        deadline = time.monotonic() + args.hours * 3600
        if args.command == "qualify":
            qualify(root, config, devices, deadline, args.retry)
            return
        qualified = json.loads((root / "qualified.json").read_text())
        for relative in qualified["attempts"]:
            path = root / relative
            if not completed(path, json.loads((path / "run.json").read_text())):
                raise ValueError("Qualification evidence incomplete")
        config["batch_size"] = qualified["batch_size"]
        if config.get("screening") and args.command == "tune":
            raise ValueError("Screening uses fixed shared hyperparameters; run directly after qualification")
        if args.command == "tune":
            runs = tuning_runs(config)
            queue(
                root, shard_runs(runs, args.shard), config, devices, deadline, args.retry, **({"shared": True} if args.shared_queue else {})
            )
            if args.shard is None:
                finalize_tuning(root, config)
        elif config.get("screening"):
            selected = {"muon": {"lr": 0.003, "weight_decay": 0.001}}
            runs = main_runs(config, selected)
            with (root / ".screen-plan.lock").open("a") as plan_lock:
                fcntl.flock(plan_lock, fcntl.LOCK_EX)
                if (root / "plan.json").exists():
                    if json.loads((root / "plan.json").read_text()) != runs:
                        raise ValueError("Screening plan differs from frozen configuration")
                else:
                    write_json(root / "plan.json", runs)
            queue(
                root, shard_runs(runs, args.shard), config, devices, deadline, args.retry, **({"shared": True} if args.shared_queue else {})
            )
        else:
            selected = json.loads((root / "tuning.json").read_text())
            paths = [latest_complete(root, run) for run in tuning_runs(config)]
            if selected != select_tuning(paths):
                raise ValueError("Tuning selection differs from completed validation evidence")
            runs = main_runs(config, selected)
            if json.loads((root / "plan.json").read_text()) != runs:
                raise ValueError("Main run plan differs from tuning selection")
            queue(
                root, shard_runs(runs, args.shard), config, devices, deadline, args.retry, **({"shared": True} if args.shared_queue else {})
            )
        if args.shard is None and not args.shared_queue:
            summarize(root)
        else:
            # Serialize shared JSON/summary writes; earlier shards exit successfully.
            with (root / ".controller.lock").open("a") as finalization_lock:
                fcntl.flock(finalization_lock, fcntl.LOCK_EX)
                if args.command == "tune":
                    finalize_tuning(root, config)
                elif all_complete(root, runs) is not None:
                    summarize(root)


if __name__ == "__main__":
    main()
