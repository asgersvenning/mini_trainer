from .collector import BaseResultCollector, RawResultCollector, _ResultsCollector
from .core import BaseStatistic, MetricLogger, MultiLogger, _Logger, _Statistic
from .tensorboard import TensorboardLogger
from .wandb import WandbLogger


def configure_loggers(use_tensorboard: bool = False, use_wandb: bool = False):
    """Assemble a list of logger classes based on the requested backends."""
    loggers: list[type[_Logger]] = [MetricLogger]
    if use_tensorboard:
        loggers.append(TensorboardLogger)
    if use_wandb:
        loggers.append(WandbLogger)
    return loggers


__all__ = [
    "BaseStatistic",
    "MetricLogger",
    "MultiLogger",
    "RawResultCollector",
    "BaseResultCollector",
    "_ResultsCollector",
    "_Logger",
    "_Statistic",
    "TensorboardLogger",
    "WandbLogger",
    "configure_loggers",
]
