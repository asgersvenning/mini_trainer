import os
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter

from mini_trainer.utils import increment_name_dir
from mini_trainer.utils.logging import BaseStatistic, _Logger, _Statistic


def make_empty_array(s : int) -> np.typing.NDArray[np.float64]:
    """Create a 1-dimensional array filled with ``np.nan``."""
    arr = np.empty((s,))
    arr[:] = np.nan
    return arr


class TensorboardLogger(_Logger):
    """Tensorboard logger."""
    def __init__(
            self, 
            steps : list[int], 
            output : str | None,
            name : str | None=None,
            tag : str | list[str] | None=None,
            flush_rate : int=5
        ):
        """Tensorboard logger."""
        if steps is None:
            raise TypeError(f'Initializing {TensorboardLogger} with `steps=None` is invalid.')
        if output is None:
            raise TypeError('Cannot initialize tensorboard logger without an output directory!')
        if name is None:
            name = increment_name_dir("run", os.path.join(output, "tensorboard"))
        if not isinstance(flush_rate, int) or flush_rate <= 1:
            raise ValueError(
                'Invalid `flush_rate`, flush rate must be an integer '
                f'greater than 1, but {flush_rate} was supplied.'
            )
        
        self.global_steps = steps
        self.output = output
        self.name = name
        self.writer = SummaryWriter(
            log_dir=os.path.join(output, "tensorboard", name), 
            flush_secs=30
        )
        
        self.tag = tag or "main"
        self.flush_rate = flush_rate

        self._idx = 0
        self._statistics : dict[str, _Statistic] = dict()
        self.clear_buffer()

    def add_stat(
            self,
            name : str,
            container : _Statistic | type[_Statistic]=BaseStatistic
        ):
        """Add new statistic to tensorboard."""
        if isinstance(container, type):
            container = container()
        self._statistics[name] = container
    
    @property
    def statistics(self):
        return self._statistics

    def _make_scalar_hierarchical_tag(self, name : str):
        if isinstance(self.tag, str):
            return f"{name}/{self.tag}"
        return "/".join([name, *self.tag])

    def clear_buffer(self):
        self._buffer = defaultdict(
            lambda : (
                make_empty_array(self.flush_rate), 
                make_empty_array(self.flush_rate)
            )
        )

    def buffer_scalar(self, tag : str, value : int | float, step : int):
        """Buffer incoming values before writing to tensorboard file(s)."""
        buf = self._buffer[tag]
        buf[0][step % self.flush_rate] = step
        buf[1][step % self.flush_rate] = value

    def flush(self):
        """Flush logger buffer to tensorboard file(s)."""
        for tag, (steps, values) in self._buffer.items():
            if np.all(np.isnan(steps)) or np.all(np.isnan(values)) or len(steps) == 0:
                continue
            step, value = np.nanmax(steps), np.nanmean(values)
            self.writer.add_scalar(tag, value, self.global_steps[int(step)])
        self.clear_buffer()

    def update(self, name : str, values):
        """Add values to tensorboard."""
        if isinstance(values, (torch.Tensor, np.ndarray)):
            values = values.tolist()
        tag = self._make_scalar_hierarchical_tag(name)
        if isinstance(values, (float, int)):
            self.buffer_scalar(tag, values, self._idx)
        else:
            for i, v in enumerate(values):
                self.buffer_scalar(tag, v, self._idx + i)
        super().update(name, values)

    def add_figure(self, name : str, figure : plt.Figure | np.ndarray | str, epoch : int): # pyright: ignore[reportPrivateImportUsage]
        """Add figure to tensorboard."""
        tag = self._make_scalar_hierarchical_tag(name)
        if isinstance(figure, plt.Figure): # pyright: ignore[reportPrivateImportUsage]
            self.writer.add_figure(tag, figure, epoch, close=False)
        else:
            if isinstance(figure, np.ndarray):
                figure = np.permute_dims(figure, (2, 0, 1))
            self.writer.add_image(tag, figure, epoch)

    def step(self):
        """Step tensorboard logger."""
        self._idx += 1
        if (
            (self._idx > 0 and self._idx % self.flush_rate == 0) or 
            (self._idx + 1) >= len(self.global_steps)
        ):
            self.flush()
