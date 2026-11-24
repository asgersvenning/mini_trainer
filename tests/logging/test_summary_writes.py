"""summary.csv rows are appended by a background thread: phase changes do not wait for the output file system."""

import csv
import subprocess
import sys
import time

import pytest

import mini_trainer.logging.core as core

SLOW = 0.4  # seconds per append, standing in for a slow network file system


def slow_append(monkeypatch, fail=False):
    real = core.write_csv_from_dict

    def append(row, path):
        time.sleep(SLOW)
        if fail:
            raise OSError("disk gone")
        real(row, path)

    monkeypatch.setattr(core, "write_csv_from_dict", append)


def run_phases(tmp_path, epochs=3):
    """Train/eval phases logging the loss ``10 * epoch + phase``; returns the logger and the time spent in update()."""
    log = core.MultiLogger(train_loader=[0], val_loader=[0], epochs=epochs, output=str(tmp_path), name="run", diagnostics=False)
    waited = 0.0
    for epoch in range(epochs):
        for phase, kind in enumerate(("train", "eval")):
            started = time.monotonic()
            log.update(epoch=epoch, type=kind)
            waited += time.monotonic() - started
            log.log_statistic(loss=10.0 * epoch + phase)
    return log, waited


def rows(tmp_path):
    with open(tmp_path / "run" / "logs" / "summary.csv", newline="") as f:
        return [(r["epoch"], r["type"], float(r["loss"])) for r in csv.DictReader(f)]


def test_phase_changes_do_not_wait_for_summary_writes_and_rows_keep_their_order(tmp_path, monkeypatch):
    slow_append(monkeypatch)
    log, waited = run_phases(tmp_path)
    assert waited < SLOW  # five queued appends, 2 s of writing, did not hold up the phase changes
    log.finish()  # waits for every row
    expected = [(str(e), kind, 10.0 * e + p) for e in range(3) for p, kind in enumerate(("train", "eval"))]
    assert rows(tmp_path) == expected


def test_a_failed_summary_write_is_raised_at_finish(tmp_path, monkeypatch):
    slow_append(monkeypatch, fail=True)
    log, _ = run_phases(tmp_path, epochs=1)
    with pytest.raises(OSError, match="disk gone"):
        log.finish()


CRASH = f"""
import sys, time
import mini_trainer.logging.core as core
real = core.write_csv_from_dict
def append(row, path):
    time.sleep({SLOW})
    real(row, path)
core.write_csv_from_dict = append
log = core.MultiLogger(train_loader=[0], val_loader=[0], epochs=2, output=sys.argv[1], name="run", diagnostics=False)
for epoch in range(2):
    for phase, kind in enumerate(("train", "eval")):
        log.update(epoch=epoch, type=kind)
        log.log_statistic(loss=10.0 * epoch + phase)
raise RuntimeError("training crashed")
"""


def test_queued_rows_are_written_when_training_crashes(tmp_path):
    done = subprocess.run([sys.executable, "-c", CRASH, str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "training crashed" in done.stderr
    # the three rows queued before the crash (the last phase never ended) are all on disk
    assert rows(tmp_path) == [("0", "train", 0.0), ("0", "eval", 1.0), ("1", "train", 10.0)]
