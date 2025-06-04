from typing import Optional, Type, Union

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from mini_trainer.utils.logging import BaseStatistic, _Logger, _Statistic

from matplotlib import pyplot as plt


class TensorboardLogger(_Logger):
    def __init__(self, writer : SummaryWriter, steps : list[int], tag : Optional[Union[str, list[str]]]=None):
        self.writer = writer
        self._statistics : dict[str, _Statistic] = dict()
        if steps is None:
            raise TypeError(f'Initializing {TensorboardLogger} with `steps=None` is invalid.')
        self.global_steps = steps
        self._idx = 0
        self.tag = tag or "main"

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

    def update(self, name : str, values):
        if isinstance(values, (torch.Tensor, np.ndarray)):
            values = values.tolist()
        tag = self._make_scalar_hierarchical_tag(name)
        if isinstance(values, (float, int)):
            self.writer.add_scalar(tag, values, self.global_steps[self._idx])
        else:
            for i, v in enumerate(self.values):
                self.writer.add_scalar(tag, values, self.global_steps[self._idx + i])
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
