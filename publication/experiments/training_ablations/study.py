"""Run with python -m publication.experiments.training_ablations.study."""

import argparse
import fcntl
import importlib.metadata
import itertools
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import digest, prepare_data, write_json

REPO = Path(__file__).resolve().parents[3]
STOP = threading.Event()
DEFAULTS = {
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
    "fixed_adjustment": {"loss": "fixed"},
    "no_projection": {"hidden": False},
    "adamw": {"optimizer": "adamw"},
    "reference": {"normalized": False, "hidden": False, "regularization": False, "loss": "ce", "optimizer": "adamw"},
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


def main_runs(config, selected):
    runs = []
    for seed in config["seeds"]:
        block = []
        for name, changes in VARIANTS.items():
            run = {**FULL, **changes}
            run.update(selected[run["optimizer"]])
            block.append({**run, "seed": seed, "epochs": config["epochs"], "variant": name, "id": f"{name}_seed{seed}"})
        random.Random(seed).shuffle(block)
        runs.extend(block)
    return runs


def prepare(config_path, root):
    config = {**DEFAULTS, **json.loads(Path(config_path).read_text())}
    if set(config) - (set(DEFAULTS) | {"parquet", "images", "pretrained"}):
        raise ValueError("Unknown configuration keys")
    for key in ["parquet", "images"]:
        config[key] = str(Path(config[key]).expanduser().resolve())
    for key in ["species", "size", "epochs", "tuning_epochs", "batch_size", "threads"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["workers"] < 0 or config["batch_size"] not in [32, 64, 128, 256, 512, 768]:
        raise ValueError("Nonnegative workers and batch size 32, 64, 128, 256, 512 or 768 required")
    if len(set(config["seeds"])) != len(config["seeds"]) or 41 in config["seeds"]:
        raise ValueError("Main seeds must be unique and distinct from tuning seed 41")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", config)
    write_json(root / "tuning-plan.json", tuning_runs(config))
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


def execute(root, run, config, device, deadline, retry=False):
    directory = root / "runs" / run["id"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute_locked(root, run, config, device, deadline, retry, directory)


def execute_locked(root, run, config, device, deadline, retry, directory):
    attempts = sorted(directory.glob("attempt-*"))
    attempt = attempts[-1] if attempts else None
    if attempt and completed(attempt, run):
        if json.loads((attempt / "resolved.json").read_text()) != config:
            raise ValueError("Completed run configuration differs")
        return attempt
    if attempt and not retry:
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


def queue(root, runs, config, devices, deadline, retry):
    # Deterministic static lanes; paired seed blocks rotate across devices.
    def lane(index):
        return [execute(root, run, config, devices[index], deadline, retry) for run in runs[index :: len(devices)]]

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
    treatments = {name: VARIANTS[name] for name in ["full", "no_projection", "adamw", "no_normalization", "fixed_adjustment"]}
    treatments["adamw_no_projection"] = {"optimizer": "adamw", "hidden": False}
    for batch in [v for v in [768, 512, 256, 128, 64, 32] if v <= config["batch_size"]]:
        candidate = {**config, "batch_size": batch}
        runs = [
            {
                **FULL,
                **changes,
                "seed": 40,
                "epochs": 2,
                "lr": 0.001,
                "weight_decay": 0.01,
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
    contrasts = []
    for seed in sorted({row["seed"] for row in main}):
        block = {row["variant"]: row for row in main if row["seed"] == seed}
        for name in VARIANTS:
            if name != "full" and name in block and "full" in block:
                contrasts.append(
                    {
                        "seed": seed,
                        "contrast": f"full - {name}",
                        "macro_recall_difference": block["full"]["macro_recall"] - block[name]["macro_recall"],
                    }
                )
    write_json(root / "paired.json", contrasts)
    paired_summary = []
    for name in sorted({row["contrast"] for row in contrasts}):
        values = [row["macro_recall_difference"] for row in contrasts if row["contrast"] == name]
        paired_summary.append(
            {
                "contrast": name,
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
        for i, name in enumerate(VARIANTS):
            values = [row["macro_recall"] for row in main if row["variant"] == name]
            if values:
                ax.scatter([i] * len(values), values)
                ax.plot(i, np.mean(values), "k_")
        ax.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=35, ha="right")
        ax.set_ylabel("Test macro recall (points: individual seeds)")
        fig.tight_layout()
        fig.savefig(root / "macro-recall.png")
        plt.close(fig)
    write_json(
        root / "summary.json",
        {
            "completed_main_runs": len(main),
            "expected_main_runs": len(VARIANTS) * len(json.loads((root / "config.json").read_text())["seeds"]),
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
    parser.add_argument("command", choices=["prepare", "qualify", "tune", "run", "status", "summarize", "_worker"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--devices", default="0", help="Comma-separated equivalent GPU IDs; one process per GPU")
    parser.add_argument("--hours", type=float, default=24, help="Remaining allocation hours, including a cleanup reserve")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--shard", type=parse_shard, help="Zero-based INDEX/COUNT; tune/run on a shared prepared root")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--stage", choices=["train", "evaluate"])
    args = parser.parse_args()
    if args.shard is not None and args.command not in ["tune", "run"]:
        parser.error("--shard is supported only for tune and run")
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
    if args.command == "status":
        for path in sorted((root / "runs").glob("*/attempt-*")):
            print(
                path.relative_to(root),
                "complete" if (path / "complete.json").exists() else "failed" if (path / "failure.json").exists() else "incomplete",
            )
        return
    lock_name = ".controller.lock" if args.shard is None else f".controller-{args.command}-{args.shard[0]}-of-{args.shard[1]}.lock"
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
        if args.command == "tune":
            runs = tuning_runs(config)
            queue(root, shard_runs(runs, args.shard), config, devices, deadline, args.retry)
            if args.shard is None:
                finalize_tuning(root, config)
        else:
            selected = json.loads((root / "tuning.json").read_text())
            paths = [latest_complete(root, run) for run in tuning_runs(config)]
            if selected != select_tuning(paths):
                raise ValueError("Tuning selection differs from completed validation evidence")
            runs = main_runs(config, selected)
            if json.loads((root / "plan.json").read_text()) != runs:
                raise ValueError("Main run plan differs from tuning selection")
            queue(root, shard_runs(runs, args.shard), config, devices, deadline, args.retry)
        if args.shard is None:
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
