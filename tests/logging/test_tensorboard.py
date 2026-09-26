"""Phase ownership contracts without the optional TensorBoard package or a server."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mini_trainer.logging import MetricLogger, MultiLogger, TensorboardLogger


@pytest.fixture
def writers(monkeypatch):
    created = []

    def create(**kwargs):
        writer = Mock(spec=["add_scalar", "close"])
        created.append(writer)
        return writer

    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard.writer", SimpleNamespace(SummaryWriter=create))
    return created


def make_logger(tmp_path, **kwargs):
    return MultiLogger([0] * 5, [0] * 5, 1, str(tmp_path), "run", statistics=["loss"], **kwargs)


@pytest.mark.parametrize("boundary", ["phase", "finish"])
def test_phase_boundary_flushes_partial_buffer_and_closes_writer(tmp_path, writers, boundary):
    logger = make_logger(tmp_path, logger_cls=[MetricLogger, TensorboardLogger])
    logger.update(0, "train")
    for value in (1.0, 3.0):
        logger.log_statistic(loss=value)
        logger.step()
    writer = writers[0]
    writer.add_scalar.assert_not_called()
    if boundary == "phase":
        logger.update(0, "eval")
        assert len(writers) == 2
        writers[1].close.assert_not_called()
    else:
        logger.finish()
    writer.add_scalar.assert_called_once_with("loss/train", 2.0, 1)
    writer.close.assert_called_once()
    if boundary == "phase":
        logger.finish()
        writers[1].close.assert_called_once()
        writer.close.assert_called_once()


@pytest.mark.parametrize("failure", ["summary", "flush"])
def test_finish_closes_all_backends_even_if_saving_fails(tmp_path, writers, monkeypatch, failure):
    class OtherLogger(MetricLogger):
        close = Mock()

    logger = make_logger(tmp_path, logger_cls=[OtherLogger, TensorboardLogger])
    logger.update(0, "train")
    logger.log_statistic(loss=1.0)
    if failure == "summary":
        monkeypatch.setattr(logger, "_store_summary", Mock(side_effect=OSError("write failed")))
    else:
        writers[0].add_scalar.side_effect = OSError("write failed")
    with pytest.raises(OSError, match="write failed"):
        logger.finish()
    writers[0].close.assert_called_once()
    OtherLogger.close.assert_called_once()


@pytest.mark.parametrize("failure", ["constructor", "statistic"])
def test_partial_backend_initialization_closes_created_writer(tmp_path, writers, failure):
    def fail(*args, **kwargs):
        raise RuntimeError("initialization failed")

    options = {"logger_cls": [TensorboardLogger]}
    if failure == "constructor":
        options["logger_cls"].append(fail)
    else:
        options["logger_cls_stat_factory"] = [fail]
    logger = make_logger(tmp_path, **options)
    with pytest.raises(RuntimeError, match="initialization failed"):
        logger.update(0, "train")
    writers[0].close.assert_called_once()
    assert logger.loggers == []
