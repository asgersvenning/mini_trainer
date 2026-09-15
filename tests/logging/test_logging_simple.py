import time

import pytest
import torch

from mini_trainer.logging.core import ETA, SmoothedValue, Timer, accuracy, compute_aligned_steps, format_duration


def test_format_duration():
    assert format_duration(3661) == "01h01m01s"
    assert format_duration(60) == "01m00s"


def test_dendrogram_export_retains_text_and_closes_all_figures_on_failure(monkeypatch):
    from contextlib import nullcontext
    from pathlib import Path

    from matplotlib import pyplot as plt

    import mini_trainer.logging.core as core

    logger = core.MultiLogger.__new__(core.MultiLogger)
    monkeypatch.setattr(logger, "confusion_matrix", lambda: {})
    monkeypatch.setattr(core, "main_process_first", nullcontext)
    monkeypatch.setattr(core, "get_rank", lambda: 0)

    def class_matrix(model, *, log_domain, log_range):
        assert log_domain is True
        assert log_range[0] < -18 and log_range[1] == 0
        return []

    monkeypatch.setattr(core, "plot_class_distance_matrix", class_matrix)
    figures = [plt.figure(), plt.figure()]
    figures[0].text(0.5, 0.5, "Species test")
    monkeypatch.setattr(core, "plot_probabilistic_dendrogram", lambda model: [(fig, {}) for fig in figures])
    exported = []

    def write(name, path):
        exported.append(Path(path).read_text())
        raise OSError("simulated writer failure")

    monkeypatch.setattr(logger, "add_figure", write)
    with pytest.warns(UserWarning, match="simulated writer failure"):
        logger.figures(object())
    assert "<text" in exported[0] and "Species test" in exported[0]
    assert not any(plt.fignum_exists(fig.number) for fig in figures)


def test_figures_saved_locally_without_external_backend(tmp_path, monkeypatch):
    import numpy as np
    from matplotlib import pyplot as plt

    import mini_trainer.logging.core as core

    logger = core.MultiLogger(train_loader=[0], val_loader=[0], epochs=1, output=str(tmp_path), name="run")
    logger.update(epoch=0, type="eval")
    monkeypatch.setattr(core, "get_rank", lambda: 0)
    source = tmp_path / "temporary.svg"
    source.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>Species</text></svg>')
    logger.add_figure("Probabilistic dendrogram/lvl0", str(source))
    source.unlink()
    logger.add_figure("Confusion matrix/lvl0", np.zeros((10, 10, 3), dtype=np.uint8))
    fig = plt.figure()
    logger.add_figure("Example", fig)
    destination = tmp_path / "run" / "logs" / "figures" / "epoch-0001"
    assert "Species" in (destination / "Probabilistic_dendrogram_lvl0.svg").read_text()
    assert (destination / "Confusion_matrix_lvl0.png").is_file()
    assert not plt.fignum_exists(fig.number)
    monkeypatch.setattr(core, "get_rank", lambda: 1)
    logger.add_figure("rank1", np.zeros((10, 10, 3), dtype=np.uint8))
    assert not (destination / "rank1.png").exists()


def test_Timer():
    t = Timer()
    assert not t.running
    t.start()
    assert t.running
    time.sleep(0.01)
    t.stop()
    assert not t.running
    assert t.total >= 0.0

    # Test errors
    with pytest.raises(RuntimeError):
        t.stop()  # already stopped

    t.start()
    with pytest.raises(RuntimeError):
        t.start()  # already running
    with pytest.raises(RuntimeError):
        _ = t.total  # total is invalid while running? -> Code says: raise RuntimeError("Attempting to grab total of a running timer!")

    assert "Timer[Running]" in str(t)
    t.stop()
    assert "Timer[Stopped]" in str(t)


def test_ETA():
    eta = ETA(total_steps=10)
    assert eta.remaining == 10
    eta.step()
    assert eta.remaining == 9
    assert eta.eta is not None


def test_accuracy():
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    acc1 = accuracy(output, target, topk=(1,))
    assert acc1[0] == 100.0

    output = torch.tensor([[0.9, 0.1], [0.2, 0.8]])  # Wrong
    acc1 = accuracy(output, target, topk=(1,))
    assert acc1[0] == 0.0


def test_compute_aligned_steps():
    # target len 10, origin len 10
    steps = compute_aligned_steps(10, 10, 1, 0)
    assert len(steps) == 10
    assert steps[0] == 0
    assert steps[-1] == 9

    # Validation usually has fewer steps or different freq
    # Origin 5, Target 10
    steps = compute_aligned_steps(10, 5, 1, 0)
    assert len(steps) == 5
    # linspace(0, 9, 5) -> 0, 2.25, 4.5, 6.75, 9
    # round: 0, 2, 4 (4.5 rounds to nearest even? or Up? Py3 round ties to even: 4).
    # 6.75 -> 7.
    # 9 -> 9.
    # [0, 2, 4, 7, 9]
    assert steps == [0, 2, 4, 7, 9]


def test_SmoothedValue():
    sv = SmoothedValue(window_size=2)
    sv.update(1.0)
    sv.update(2.0)
    assert sv.count == 2
    assert sv.total == 3.0
    assert sv.value == 2.0
    assert sv.avg == 1.5
    assert sv.global_avg == 1.5

    sv.update(3.0)  # deque [2.0, 3.0]
    assert sv.value == 3.0
    assert sv.avg == 2.5
    # Global avg is over all history
    assert sv.global_avg == 6.0 / 3.0  # 2.0
