"""Single-node UCloud comparison controller; no package imports or installation."""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "master_eager": ("master", {}),
    "quant_eager": ("quant", {}),
    "quant_prefetch": ("quant", {"cuda_prefetch": True}),
    "quant_compile_model": ("quant", {"compile": True}),
    "quant_compile_optimizer": ("quant", {"compile_optimizer": True}),
    "quant_combined": ("quant", {"compile": True, "compile_optimizer": True, "cuda_prefetch": True}),
    # Qualification experiments, deliberately excluded from the default matrix.
    "quant_model_graphs": ("quant", {"compile": True, "compile_mode": "reduce-overhead"}),
    "quant_optimizer_graphs": ("quant", {"compile_optimizer": True, "optimizer_cudagraphs": True}),
    "quant_int8": ("quant", {"quantized_training": True}),
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def validate(config):
    if "figures" in config and type(config["figures"]) is not bool:
        raise ValueError("figures must be a boolean")
    if "require_finite_losses" in config and type(config["require_finite_losses"]) is not bool:
        raise ValueError("require_finite_losses must be a boolean")
    if "budget_seconds" in config and (type(config["budget_seconds"]) is not int or config["budget_seconds"] < 1):
        raise ValueError("budget_seconds must be a positive integer")
    qualification = config.get("qualification")
    if qualification is not None:
        if not isinstance(qualification, dict) or set(qualification) != {"seed", "train", "validation", "test"}:
            raise ValueError("qualification requires exactly seed, train, validation and test")
        if type(qualification["seed"]) is not int:
            raise ValueError("qualification.seed must be an integer")
        if any(type(qualification[key]) is not int or qualification[key] < 1 for key in ("train", "validation", "test")):
            raise ValueError("qualification split sizes must be positive integers")
    for key in ("gpus", "global_batch_size", "epochs", "size", "timeout_seconds"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["gpus"] not in (1, 2, 4, 8):
        raise ValueError("Choose 1, 2, 4 or 8 GPUs on one node")
    if config["global_batch_size"] % config["gpus"]:
        raise ValueError("global_batch_size must divide evenly across GPUs")
    if qualification is not None and qualification["train"] < config["global_batch_size"]:
        raise ValueError("qualification.train must contain at least one global batch")
    if type(config["num_workers_per_rank"]) is not int or config["num_workers_per_rank"] < 0:
        raise ValueError("num_workers_per_rank must be a nonnegative integer")
    if not config["seeds"] or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("Supply integer seeds")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate seeds")
    if not config["variants"] or len(set(config["variants"])) != len(config["variants"]):
        raise ValueError("Supply unique variants")
    if set(config["variants"]) - VARIANTS.keys():
        raise ValueError("Unknown variant")
    if "quant_int8" in config["variants"] and config["gpus"] != 1:
        raise ValueError("Native INT8 training requires gpus=1; DDP is unsupported")
    if not {"master_eager", "quant_eager"}.issubset(config["variants"]):
        raise ValueError("Keep both eager controls")
    for branch in ("master", "quant"):
        entry = config["environments"][branch]
        if len(entry["commit"]) != 40 or any(c not in "0123456789abcdef" for c in entry["commit"]):
            raise ValueError("Pin each environment to a full Git commit")
        entry["python"] = str(Path(entry["python"]).expanduser().absolute())
    for key in ("parquet", "output"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    return config


def plan(config):
    runs = []
    for seed in config["seeds"]:
        variants = config["variants"].copy()
        random.Random(seed).shuffle(variants)
        for variant in variants:
            branch, options = VARIANTS[variant]
            runs.append({"name": f"{variant}_seed{seed}", "seed": seed, "branch": branch, "options": options})
    return runs


def command(config, run, config_path):
    python = config["environments"][run["branch"]]["python"]
    worker = [str(HERE / "worker.py"), "train", str(config_path), "--run", run["name"]]
    if config["gpus"] == 1:
        return [python, *worker]
    return [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={config['gpus']}",
        "--max-restarts=0",
        *worker,
    ]


def execute(args, log, cwd, timeout):
    """Kill the entire torchrun process group on timeout or interruption."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    with Path(log).open("w") as handle:
        proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            # The leader may exit before its loader/compiler children do.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            raise


def audit_losses(path, epochs):
    """Check recorded train/eval losses, without changing the measured GPU loop."""
    expected = {(str(epoch), phase) for epoch in range(epochs) for phase in ("train", "eval")}
    seen = set()
    issues = []
    try:
        with Path(path).open() as handle:
            for row in csv.DictReader(handle):
                key = (row.get("epoch"), row.get("type"))
                if key not in expected or key in seen:
                    issues.append(f"Unexpected or duplicate epoch/phase: {key}")
                seen.add(key)
                for column in ("loss", "loss/lvl0", "loss/lvl1", "loss/lvl2"):
                    try:
                        finite = math.isfinite(float(row.get(column, "")))
                    except (TypeError, ValueError):
                        finite = False
                    if not finite:
                        issues.append(f"epoch={key[0]} phase={key[1]} {column}={row.get(column)!r}")
    except OSError as error:
        issues.append(str(error))
    issues.extend(f"Missing epoch={epoch} phase={phase}" for epoch, phase in sorted(expected - seen))
    return issues


def summary(output):
    rows = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*/result.json"))]
    for row in rows:
        phases = [
            [json.loads(line) for line in path.read_text().splitlines()]
            for path in sorted((output / "runs" / row["name"]).glob("phases-rank*.jsonl"))
        ]
        if phases:
            training = [[p["seconds"] for p in rank if p["phase"] == "train"] for rank in phases]
            if training and all(len(t) == len(training[0]) for t in training) and training[0]:
                times = [max(epoch) for epoch in zip(*training, strict=True)]
                row["train_seconds"] = sum(times)
                row["first_epoch_seconds"] = times[0]
                row["later_epoch_mean_seconds"] = sum(times[1:]) / len(times[1:]) if len(times) > 1 else ""
            row["peak_allocated_bytes"] = max(p["max_memory_allocated"] for rank in phases for p in rank)
            scores = [p["selection_metric"] for p in phases[0] if p["phase"] == "validation"]
            row["best_validation_metric"] = max(scores) if scores else ""
    fields = [
        "name",
        "branch",
        "seed",
        "status",
        "loss_check",
        "scope",
        "returncode",
        "wall_seconds",
        "train_seconds",
        "first_epoch_seconds",
        "later_epoch_mean_seconds",
        "peak_allocated_bytes",
        "best_validation_metric",
        "checkpoint",
        "checkpoint_sha256",
    ]
    with (output / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    controls = {row["seed"]: row for row in rows if row["name"].startswith("master_eager_") and row["status"] == "completed"}
    paired = []
    for row in rows:
        if row["status"] == "completed" and row["seed"] in controls:
            control = controls[row["seed"]]
            paired.append(
                {
                    "name": row["name"],
                    "seed": row["seed"],
                    "scope": row.get("scope", "full_dataset"),
                    "wall_speedup_vs_master": control["wall_seconds"] / row["wall_seconds"],
                    "validation_delta_vs_master": (
                        row["best_validation_metric"] - control["best_validation_metric"]
                        if isinstance(row.get("best_validation_metric"), (int, float))
                        and isinstance(control.get("best_validation_metric"), (int, float))
                        else None
                    ),
                }
            )
    write_json(output / "paired.json", paired)


class BudgetExceeded(TimeoutError):
    pass


@contextlib.contextmanager
def execution_budget(config, stage):
    """One wall-clock deadline across prepare and train, including time between commands."""
    if "budget_seconds" not in config or stage not in ("prepare", "train"):
        yield None
        return
    if stage == "prepare":
        deadline = time.time() + config["budget_seconds"]
    else:
        path = Path(config["output"]) / "budget.json"
        if not path.is_file():
            raise ValueError("Preparation budget record is missing; complete preparation in a new output directory")
        deadline = json.loads(path.read_text())["deadline"]
    remaining = deadline - time.time()
    if remaining <= 0:
        raise BudgetExceeded("Comparison time budget expired; preserve results and use a new output for another attempt")

    def expired(signum, frame):
        raise BudgetExceeded("Comparison time budget expired; active workers are being stopped")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=("plan", "prepare", "train", "summary"), default="plan")
    parser.add_argument("--only", help="Run one planned name (for a qualification run)")
    args = parser.parse_args()
    config = validate(json.loads(args.config.read_text()))
    try:
        with execution_budget(config, args.stage) as deadline:
            run_stage(config, args, deadline)
    except BudgetExceeded as error:
        raise SystemExit(str(error)) from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def run_stage(config, args, deadline=None):
    runs = plan(config)
    output = Path(config["output"])
    config_path = output / "comparison.json"
    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "per_rank_batch": config["global_batch_size"] // config["gpus"],
                    "scope": "qualification_subset" if config.get("qualification") else "full_dataset",
                    "qualification": config.get("qualification"),
                    "budget_seconds": config.get("budget_seconds"),
                    "runs": runs,
                },
                indent=2,
            )
        )
        return
    if args.stage == "summary":
        summary(output)
        return
    if args.stage == "prepare":
        if output.exists():
            raise ValueError(f"Output already exists: {output}. Preserve it and choose a new output path; preparation cannot resume")
        output.mkdir(parents=True, exist_ok=False)
        (output / "runs").mkdir()
        if deadline is not None:
            write_json(output / "budget.json", {"deadline": deadline})
        write_json(config_path, config)
        write_json(output / "plan.json", runs)
        # Preserve the harness itself independently of either installed branch.
        for file in HERE.glob("*.py"):
            (output / file.name).write_bytes(file.read_bytes())
        for branch in ("master", "quant"):
            print(f"Checking {branch}; log: {output}/preflight-{branch}.log", flush=True)
            result = execute(
                [config["environments"][branch]["python"], str(HERE / "worker.py"), "preflight", str(config_path), "--branch", branch],
                output / f"preflight-{branch}.log",
                output,
                300,
            )
            if result:
                raise RuntimeError(f"Preflight failed: see {output}/preflight-{branch}.log")
        master = json.loads((output / "environment-master.json").read_text())
        quant = json.loads((output / "environment-quant.json").read_text())
        if master["dependencies"] != quant["dependencies"] or master["python_version"] != quant["python_version"]:
            raise RuntimeError("Environment versions differ; use the same locked dependencies and Python for both branches")
        print(f"Preparing dataset and starting weights; follow: tail -f {output}/prepare.log", flush=True)
        result = execute(
            [config["environments"]["master"]["python"], str(HERE / "worker.py"), "prepare", str(config_path)],
            output / "prepare.log",
            output,
            config["timeout_seconds"],
        )
        if result:
            raise RuntimeError(f"Preparation failed: see {output}/prepare.log; use a new output for a fresh attempt")
        files = ["class_spec.json", "data_index.json", "dataset.json", "preprocessing.txt"]
        files += [p.name for p in HERE.glob("*.py")]
        files += [f"initial_seed{s}.pt" for s in config["seeds"]]
        if config.get("qualification"):
            files.append("qualification.parquet")
        if deadline is not None:
            files.append("budget.json")
        write_json(output / "prepared.json", {name: digest(output / name) for name in files})
        print(f"Preparation complete: {output}/prepared.json", flush=True)
        return
    if not (output / "prepared.json").is_file():
        raise ValueError(f"Preparation is incomplete: {output}. Check the active process and prepare.log before starting a new attempt")
    if json.loads(config_path.read_text()) != config:
        raise ValueError("Configuration differs from the frozen preparation; use a new output directory")
    for name, sha in json.loads((output / "prepared.json").read_text()).items():
        if digest(output / name) != sha:
            raise ValueError(f"Prepared artifact changed: {name}")
        if name.endswith(".py") and digest(HERE / name) != sha:
            raise ValueError(f"Harness changed after preparation: {name}")
    if digest(config["parquet"]) != json.loads((output / "dataset.json").read_text())["parquet_sha256"]:
        raise ValueError("Source Parquet changed")
    if args.only and args.only not in {run["name"] for run in runs}:
        raise ValueError("--only must match a name in --stage plan")
    failed = False
    for run in runs:
        if args.only and args.only != run["name"]:
            continue
        directory = output / "runs" / run["name"]
        if directory.exists():
            record = directory / "result.json"
            if record.exists():
                result = json.loads(record.read_text())
                if result["status"] == "completed" and digest(result["checkpoint"]) == result["checkpoint_sha256"]:
                    continue
            raise RuntimeError(f"Incomplete/failed run exists: {directory}. Preserve it and use a new comparison directory")
        directory.mkdir()
        args_run = command(config, run, config_path)
        write_json(directory / "launch.json", {**run, "argv": args_run})
        result = {**run, "status": "running", "scope": "qualification_subset" if config.get("qualification") else "full_dataset"}
        write_json(directory / "result.json", result)
        print(f"Starting {run['name']}; log: {directory}/console.log", flush=True)
        started = time.monotonic()
        try:
            result["returncode"] = execute(args_run, directory / "console.log", output, config["timeout_seconds"])
            checkpoint = directory / "model" / "weights" / "best.pt"
            result["status"] = "completed" if result["returncode"] == 0 and checkpoint.is_file() else "failed"
            if result["status"] == "completed":
                result.update(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
                if config.get("require_finite_losses", False):
                    issues = audit_losses(directory / "model" / "logs" / "summary.csv", config["epochs"])
                    result["loss_check"] = "failed" if issues else "passed"
                    if issues:
                        result.update(status="invalid_metrics", loss_issues=issues)
        except KeyboardInterrupt:
            result.update(status="interrupted", error="Interrupted by user or termination signal")
            raise SystemExit(130) from None
        except (BudgetExceeded, subprocess.TimeoutExpired) as error:
            result.update(status="timed_out", error=str(error))
            raise
        except BaseException as error:
            result.update(status="failed", error=repr(error))
            raise
        finally:
            result["wall_seconds"] = time.monotonic() - started
            write_json(directory / "result.json", result)
            summary(output)
        failed |= result["status"] != "completed"
    if failed:
        raise SystemExit("Some runs failed; see comparison.csv and individual console logs")


if __name__ == "__main__":
    main()
