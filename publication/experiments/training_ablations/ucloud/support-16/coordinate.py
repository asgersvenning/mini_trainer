"""Prepare once, verify the shared result mount, then dispatch two run lanes."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/work/results/support-16")
CONFIG = "publication/experiments/training_ablations/support-sensitivity.json"
HOURS = 8
MINIMUM_RUN_HOURS = 5


def study(*args):
    subprocess.run(
        [sys.executable, "-m", "publication.experiments.training_ablations.study", *map(str, args)],
        check=True,
    )


def wait_for(path):
    # Any later start would fail the minimum-run-time check anyway.
    deadline = time.monotonic() + (HOURS - MINIMUM_RUN_HOURS) * 3600
    while not path.exists():
        failures = [p.name for p in ROOT.glob("node-*.exit-code") if p.read_text().strip() not in ("", "0")]
        if failures:
            raise RuntimeError(f"Peer node failed before readiness: {failures}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Waiting for {path.name}")
        time.sleep(2)


def main(index, started):
    owner = False
    try:
        (ROOT / "preparation-owner").mkdir()
        owner = True
    except FileExistsError:
        pass

    if owner:
        lock = (ROOT / "cross-node.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT / "lock-held").touch()
        study("prepare", ROOT / "study", "--config", CONFIG)
        selection = json.loads((ROOT / "study/selection.json").read_text())
        support = selection["support_reduction"]
        if support["cap"] != 16 or support["tail_species_count"] != 505:
            raise RuntimeError("Prepared support cohort differs from the frozen 16-image, 505-species design")
        study("qualify", ROOT / "study", "--hours", "0.75")
        if json.loads((ROOT / "study/qualified.json").read_text())["batch_size"] != 512:
            raise RuntimeError("Qualification changed batch size; stop before main runs")
        (ROOT / "ready").touch()
    else:
        wait_for(ROOT / "lock-held")
        with (ROOT / "cross-node.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                (ROOT / "lock-verified").write_text(json.dumps({"verified": True, "node": os.environ["HOSTNAME"]}))
            else:
                raise RuntimeError("Cross-node lock was free: the owner exited or flock is not shared; check node logs")
        wait_for(ROOT / "ready")

    remaining = HOURS - (time.time() - float(started)) / 3600
    if remaining < MINIMUM_RUN_HOURS:
        raise RuntimeError("Insufficient allocation time remains for the paired run lanes")
    study("run", ROOT / "study", "--devices", "0", "--shared-queue", "--hours", str(remaining))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
