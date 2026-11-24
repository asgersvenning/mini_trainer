"""Opt-in stack dumps while a phase waits for its first batch (MT_PHASE_TRACEBACK_SECONDS)."""

import time

from mini_trainer.trainer import PHASE_TRACEBACK_ENV, watch_phase_start


def slow_start(delay):
    time.sleep(delay)
    yield from range(3)


def test_off_by_default_returns_the_loader_itself(monkeypatch):
    monkeypatch.delenv(PHASE_TRACEBACK_ENV, raising=False)
    batches = [1, 2]
    assert watch_phase_start(batches) is batches


def test_stacks_are_dumped_until_the_first_batch_arrives(monkeypatch, capfd):
    monkeypatch.setenv(PHASE_TRACEBACK_ENV, "0.2")
    watched = watch_phase_start(slow_start(0.7))
    batches = iter(watched)
    assert next(batches) == 0
    during = capfd.readouterr().err
    assert during.count("most recent call first") >= 2 and "slow_start" in during  # every 0.2 s, showing where the start waits
    time.sleep(0.5)
    assert list(batches) == [1, 2]
    assert capfd.readouterr().err == ""  # stopped at the first batch
