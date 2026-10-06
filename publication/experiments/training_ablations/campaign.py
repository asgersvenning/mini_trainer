"""Run the complete ablation campaign from one revision with interchangeable GPU workers.

Every worker drains all studies: the first to reach a study prepares and qualifies it, then
workers claim free runs and export each completed run's test predictions. Workers can join or
stop at any time; interrupted runs resume elsewhere. One CPU job finalizes once all are done.

    python -m publication.experiments.training_ablations.campaign work BASE --hours H
    python -m publication.experiments.training_ablations.campaign check BASE
    python -m publication.experiments.training_ablations.campaign submit REVISION [--workers N]
"""

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import study

CONFIGS = Path(__file__).parent / "campaign-studies"
JOBS = Path(__file__).parents[1] / "ucloud" / "campaign"
# Longest runs first, so the slowest work starts while most workers are available.
STUDIES = ["lepi-1513", "lepi-1513-capped", "lepi-512-duration", "plantnet", "lepi-512"]


def run_study(*args):
    return subprocess.run([sys.executable, "-m", "publication.experiments.training_ablations.study", *map(str, args)]).returncode


def ready(cohort, hours):
    """True once qualified; prepares and qualifies under a lock; False while another worker does."""
    root = cohort / "study"
    if (root / "qualified.json").exists():
        return True
    cohort.mkdir(parents=True, exist_ok=True)
    with (cohort / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (root / "prepared.json").exists():
            shutil.rmtree(root, ignore_errors=True)  # A preparation that died midway
            if run_study("prepare", root, "--config", CONFIGS / f"{cohort.name}.json"):
                raise RuntimeError(f"Preparation failed: {cohort.name}")
        if not (root / "qualified.json").exists() and run_study("qualify", root, "--hours", hours):
            raise RuntimeError(f"Qualification failed: {cohort.name}")
    return True


def work(base, hours):
    deadline = time.monotonic() + hours * 3600
    failures = []
    while True:
        waiting = False
        for name in STUDIES:
            cohort, remaining = base / name, (deadline - time.monotonic()) / 3600
            if remaining <= 0:
                break
            try:
                if not ready(cohort, remaining):
                    waiting = True
                    continue
            except RuntimeError as error:
                failures.append(str(error))
                continue
            # Train any free runs, then export test predictions for every completed run.
            if run_study("run", cohort / "study", "--shared-queue", "--hours", remaining):
                failures.append(f"Run failures: {name}")
            if run_study("predict", cohort / "study", "--shared-queue", "--output", cohort / "predictions"):
                failures.append(f"Prediction failures: {name}")
        if not waiting or time.monotonic() >= deadline:
            break
        time.sleep(60)  # Another worker is preparing a study; its runs become available afterwards.
    if failures:
        sys.exit("\n".join(failures))


def check(base):
    """Fail unless every planned run is complete and predicted; print progress either way."""
    missing = 0
    for name in STUDIES:
        root, predictions = base / name / "study", base / name / "predictions"
        runs = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else []
        complete = predicted = 0
        for run in runs:
            attempts = sorted((root / "runs" / run["id"]).glob("attempt-*"))
            complete += bool(attempts) and study.completed(attempts[-1], run)
            predicted += (predictions / run["id"] / "prediction.json").exists()
        print(f"{name}: {complete}/{len(runs)} complete, {predicted}/{len(runs)} predicted")
        missing += not runs or predicted < len(runs)
    if missing:
        sys.exit("Campaign incomplete")


def submit(revision, workers):
    """Queue workers and the finalizer with `ucloud q`; rerun to top up after failures or expiry."""
    if len(revision) != 40:
        raise ValueError("Supply the full reviewed commit")
    if workers is None:
        workers = sum(
            len(study.main_runs({**study.DEFAULTS, **json.loads((CONFIGS / f"{n}.json").read_text())}, {"muon": {}})) for n in STUDIES
        )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with tempfile.TemporaryDirectory() as directory:
        names = []
        for kind, count in [("worker", workers), ("finalize", 1)]:
            spec = Path(directory) / f"{kind}.toml"
            text = (JOBS / f"{kind}.toml").read_text().replace("REVIEWED_COMMIT", revision)
            spec.write_text(text.replace('local = "', f'local = "{study.REPO}/'))
            for index in range(count):
                name = f"campaign-{revision[:7]}-{kind}-{stamp}-{index:03d}"
                after = [arg for worker in names for arg in ("--after", worker)] if kind == "finalize" else []
                subprocess.run(["ucloud", "q", "submit", str(spec), "--name", name, "--no-tick", *after], check=True)
                names.append(name)
    subprocess.run(["ucloud", "q", "tick"], check=True)
    print(f"Queued {workers} workers and a finalizer; keep `ucloud q daemon` running to advance them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("work")
    worker.add_argument("base", type=Path)
    worker.add_argument("--hours", type=float, required=True, help="Remaining allocation, including a cleanup reserve")
    commands.add_parser("check").add_argument("base", type=Path)
    queue = commands.add_parser("submit")
    queue.add_argument("revision")
    queue.add_argument("--workers", type=int, help="Default: one per planned run")
    args = parser.parse_args()
    if args.command == "work":
        work(args.base.resolve(), args.hours)
    elif args.command == "check":
        check(args.base.resolve())
    else:
        submit(args.revision, args.workers)


if __name__ == "__main__":
    main()
