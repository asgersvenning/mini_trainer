from typing import Optional, Type, Union

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from mini_trainer.utils.logging import BaseStatistic, _Logger, _Statistic

from matplotlib import pyplot as plt
from collections import defaultdict


def make_empty_array(s : int) -> np.typing.NDArray[np.float64]:
    arr = np.empty((s,))
    arr[:] = np.nan
    return arr

class TensorboardLogger(_Logger):
    def __init__(
            self, 
            writer : SummaryWriter, 
            steps : list[int], 
            tag : Optional[Union[str, list[str]]]=None,
            flush_rate : int=5
        ):
        self.writer = writer
        self._statistics : dict[str, _Statistic] = dict()
        if steps is None:
            raise TypeError(f'Initializing {TensorboardLogger} with `steps=None` is invalid.')
        self.global_steps = steps
        self._idx = 0
        self.tag = tag or "main"
        if not isinstance(flush_rate, int) or flush_rate <= 1:
            raise ValueError(f'Invalid `flush_rate`, flush rate must be an integer greater than 1, but {flush_rate} was supplied.')
        self.flush_rate = flush_rate
        self.clear_buffer()


    def add_stat(self, name : str, container : Union[_Statistic, Type[_Statistic]]=BaseStatistic):
        if isinstance(container, type):
            container = container()
        self._statistics[name] = container

    def get(self, name : str):
        return self._statistics[name]
    
    @property
    def statistics(self):
        return self._statistics

    def _make_scalar_hierarchical_tag(self, name : str):
        if isinstance(self.tag, str):
            return f"{name}/{self.tag}"
        return "/".join([name, *self.tag])

    def clear_buffer(self):
        self._buffer = defaultdict(lambda : (make_empty_array(self.flush_rate), make_empty_array(self.flush_rate)))

    def buffer_scalar(self, tag : str, value : int | float, step : int):
        buf = self._buffer[tag]
        buf[0][step % self.flush_rate] = step
        buf[1][step % self.flush_rate] = value

    def flush(self):
        for tag, (steps, values) in self._buffer.items():
            if np.all(np.isnan(steps)) or np.all(np.isnan(values)) or len(steps) == 0:
                continue
            step, value = np.nanmax(steps), np.nanmean(values)
            self.writer.add_scalar(tag, value, self.global_steps[int(step)])
        self.clear_buffer()

    def update(self, name : str, values):
        if isinstance(values, (torch.Tensor, np.ndarray)):
            values = values.tolist()
        tag = self._make_scalar_hierarchical_tag(name)
        if isinstance(values, (float, int)):
            self.buffer_scalar(tag, values, self._idx)
        else:
            for i, v in enumerate(values):
                self.buffer_scalar(tag, v, self._idx + i)
        super().update(name, values)

    def add_figure(self, name : str, figure : Union[plt.Figure, str], epoch : int):
        tag = self._make_scalar_hierarchical_tag(name)
        if isinstance(figure, plt.Figure):
            self.writer.add_figure(tag, figure, epoch, close=False)
        else:
            if isinstance(figure, np.ndarray):
                figure = np.permute_dims(figure, (2, 0, 1))
            self.writer.add_image(tag, figure, epoch)

    def step(self):
        self._idx += 1
        if self._idx > 0 and self._idx % self.flush_rate == 0 or (self._idx + 1) >= len(self.global_steps):
            self.flush()
