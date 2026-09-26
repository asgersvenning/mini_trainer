from unittest.mock import MagicMock

import pytest

import mini_trainer.logging.wandb as wandb_module
from mini_trainer.logging import BaseStatistic, WandbLogger


@pytest.fixture
def sdk(monkeypatch):
    sdk = MagicMock()
    monkeypatch.setattr(wandb_module, "wandb", sdk)
    return sdk


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [("__init__", {"steps": [0], "output": None}), ("step", {}), ("add_figure", {"name": "plot", "figure": None})],
)
def test_wandb_logger_not_installed(monkeypatch, method, kwargs):
    monkeypatch.setattr(wandb_module, "wandb", None)
    with pytest.raises(ImportError, match="wandb is not installed"):
        getattr(WandbLogger.__new__(WandbLogger), method)(**kwargs)


@pytest.mark.parametrize("rank", [None, 0, 1], ids=["local", "primary", "secondary"])
@pytest.mark.parametrize(("run_id", "run_name"), [(None, None), ("", "batch64"), ("shared-trial", "batch64")])
def test_new_run_identity_and_finish_owner(sdk, monkeypatch, tmp_path, rank, run_id, run_name):
    sdk.run = None
    monkeypatch.setattr(wandb_module.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(wandb_module.os, "getcwd", lambda: "CWD")
    monkeypatch.setattr(wandb_module, "is_dist_avail_and_initialized", lambda: rank is not None)
    monkeypatch.setattr(wandb_module, "get_rank", lambda: rank)
    WandbLogger(steps=[0, 1], output=str(tmp_path), name="model trial/1", project="project", run_name=run_name, run_id=run_id)
    expected = dict(project="project", name=run_name or "model trial/1", dir=str(tmp_path), config=None, tags=["host", "CWD"])
    if rank is not None:
        sdk.Settings.assert_called_once_with(mode="shared", x_primary=rank == 0, x_update_finish_state=rank == 0, x_label=f"rank_{rank}")
        expected.update(id=run_id or "model_trial_1", settings=sdk.Settings.return_value)
    else:
        sdk.Settings.assert_not_called()
        if run_id is not None:
            expected["id"] = run_id
    sdk.init.assert_called_once_with(**expected)


@pytest.mark.parametrize("distributed", [False, True])
def test_existing_run_is_adopted(sdk, monkeypatch, distributed):
    monkeypatch.setattr(wandb_module, "is_dist_avail_and_initialized", lambda: distributed)
    WandbLogger(steps=[0, 1], output=None, run_id="must-not-replace-existing")
    sdk.init.assert_not_called()
    sdk.Settings.assert_not_called()
    with pytest.raises(TypeError):
        WandbLogger(steps=None, output=None)


def test_wandb_logger_update_and_step(sdk):
    logger = WandbLogger(steps=[0, 10], output=None)
    logger.add_stat("loss", BaseStatistic)
    assert isinstance(logger.statistics["loss"], BaseStatistic)
    logger.update("loss", 1.5)
    logger.step()
    sdk.log.assert_called_once_with({"loss/main": 1.5, "global_step": 0})
    assert logger._internal_step == 1
    assert logger._current_step_logs == {}


def test_wandb_logger_add_figure(sdk):
    import matplotlib.pyplot as plt

    logger = WandbLogger(steps=[0, 10], output=None)
    fig = plt.figure()
    try:
        logger.add_figure("my_plot", fig, epoch=1)
        logger.step()
        sdk.Image.assert_called_once_with(fig)
        sdk.log.assert_called_once_with({"my_plot/main": sdk.Image.return_value, "epoch": 1})
    finally:
        plt.close(fig)
