from unittest.mock import MagicMock, patch

import pytest

import mini_trainer.logging.wandb as wandb_module
from mini_trainer.logging import BaseStatistic, WandbLogger


def test_wandb_logger_not_installed():
    with patch.object(wandb_module, "wandb", None):
        with pytest.raises(ImportError, match="wandb is not installed"):
            WandbLogger(steps=[0, 1], output=None)


def test_wandb_logger_init():
    mock_wandb = MagicMock()
    mock_wandb.run = None
    with (
        patch.object(wandb_module, "wandb", mock_wandb),
        patch("socket.gethostname", return_value="dummy_host"),
        patch("os.getcwd", return_value="CWD"),
    ):
        WandbLogger(steps=[0, 1], output="dummy_dir", name="test", project="test_proj")
        mock_wandb.init.assert_called_once_with(project="test_proj", name="test", dir="dummy_dir", config=None, tags=["dummy_host", "CWD"])

    # Test initialization with steps=None
    with pytest.raises(TypeError):
        WandbLogger(steps=None, output=None)


def test_wandb_logger_add_stat():
    mock_wandb = MagicMock()
    with patch.object(wandb_module, "wandb", mock_wandb):
        logger = WandbLogger(steps=[0, 1], output=None)
        logger.add_stat("loss", BaseStatistic)
        assert "loss" in logger.statistics
        assert isinstance(logger.statistics["loss"], BaseStatistic)


def test_wandb_logger_update_and_step():
    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()
    mock_wandb.run.step = 0
    with patch.object(wandb_module, "wandb", mock_wandb):
        logger = WandbLogger(steps=[0, 10], output=None)
        logger.add_stat("loss", BaseStatistic)

        logger.update("loss", 1.5)
        assert logger._current_step_logs["loss/main"] == 1.5

        logger.step()
        mock_wandb.log.assert_called_once_with({"loss/main": 1.5, "step/main": 0, "trainer/global_step": 0})
        assert logger._idx == 1
        assert logger._current_step_logs == {}


def test_wandb_logger_add_figure():
    import matplotlib.pyplot as plt

    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()
    mock_wandb.run.step = 0
    with patch.object(wandb_module, "wandb", mock_wandb):
        logger = WandbLogger(steps=[0, 10], output=None)
        fig = plt.figure()

        logger.add_figure("my_plot", fig, epoch=1)
        logger.step()
        mock_wandb.Image.assert_called_once_with(fig)
        mock_wandb.log.assert_called_once_with(
            {"my_plot/main": mock_wandb.Image.return_value, "epoch": 1, "step/main": 0, "trainer/global_step": 0}
        )
