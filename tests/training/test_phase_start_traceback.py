"""Opt-in stack dumps while a phase start is pending (MT_PHASE_TRACEBACK_SECONDS)."""

import time

import pytest

import mini_trainer.trainer as trainer
from mini_trainer.trainer import PHASE_TRACEBACK_ENV, close_phase_window, watch_phase_start


@pytest.fixture(autouse=True)
def no_open_window():
    close_phase_window()
    yield
    close_phase_window()


def slow_start(delay, n=3):
    time.sleep(delay)
    yield from range(n)


def test_off_by_default_returns_the_loader_itself(monkeypatch):
    monkeypatch.delenv(PHASE_TRACEBACK_ENV, raising=False)
    batches = [1, 2]
    assert watch_phase_start(batches) is batches


def test_stacks_are_dumped_until_the_first_batch_arrives(monkeypatch, capfd):
    monkeypatch.setenv(PHASE_TRACEBACK_ENV, "0.2")
    batches = iter(watch_phase_start(slow_start(0.7)))
    assert next(batches) == 0
    during = capfd.readouterr().err
    assert during.count("most recent call first") >= 2 and "slow_start" in during  # every 0.2 s, showing where the start waits
    time.sleep(0.5)
    assert next(batches) == 1
    assert capfd.readouterr().err == ""  # stopped at the first batch


def end_of_phase_work():
    time.sleep(0.5)  # metrics, checkpoints: after the last batch, before the next phase is set up


def test_the_window_opens_at_the_previous_phase_last_batch(monkeypatch, capfd):
    monkeypatch.setenv(PHASE_TRACEBACK_ENV, "0.2")
    messages = []
    monkeypatch.setattr(trainer, "get_logger", lambda: type("L", (), {"info": staticmethod(messages.append)})())
    assert list(watch_phase_start(slow_start(0.0), "train 0")) == [0, 1, 2]
    capfd.readouterr()
    end_of_phase_work()
    assert "end_of_phase_work" in capfd.readouterr().err  # dumped although the next phase has not started yet
    assert list(watch_phase_start(slow_start(0.3), "eval 0")) == [0, 1, 2]
    window = float(messages[-1].split("first batch ")[1].split(" s")[0])
    assert messages[-1].startswith("Phase start eval 0") and 0.75 < window < 2.0  # 0.5 s of end-of-phase work + 0.3 s start
    close_phase_window()  # the trainer closes it after the last phase
    capfd.readouterr()
    time.sleep(0.5)
    assert capfd.readouterr().err == ""


def test_an_abandoned_phase_closes_the_window(monkeypatch, capfd):
    monkeypatch.setenv(PHASE_TRACEBACK_ENV, "0.2")
    batches = watch_phase_start(slow_start(0.0))
    next(batches)
    batches.close()  # the loop left early (exception, break)
    capfd.readouterr()
    time.sleep(0.5)
    assert capfd.readouterr().err == ""
