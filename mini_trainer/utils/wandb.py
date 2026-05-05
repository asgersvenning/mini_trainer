import numpy as np
import torch
from matplotlib import pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

from mini_trainer.utils.logging import BaseStatistic, _Logger, _Statistic


class WandbLogger(_Logger):
    """Weights & Biases logger."""

    def __init__(
        self,
        steps: list[int],
        output: str | None,
        name: str | None = None,
        tag: str | list[str] | None = None,
        project: str | None = "mini_trainer",
    ):
        """Wandb logger."""
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it using `uv add wandb`.")
        if steps is None:
            raise TypeError(f"Initializing {WandbLogger} with `steps=None` is invalid.")

        self.global_steps = steps
        self.name = name
        self.tag = tag or "main"

        # Initialize wandb run if not already initialized
        if wandb.run is None:
            wandb.init(project=project, name=name, dir=output)

        self._idx = 0
        self._statistics: dict[str, _Statistic] = dict()
        self._current_step_logs = {}

    def add_stat(self, name: str, container: _Statistic | type[_Statistic] = BaseStatistic):
        """Add new statistic to wandb."""
        if isinstance(container, type):
            container = container()
        self._statistics[name] = container

    @property
    def statistics(self):
        return self._statistics

    def _make_scalar_hierarchical_tag(self, name: str):
        if isinstance(self.tag, str):
            return f"{name}/{self.tag}"
        return "/".join([name, *self.tag])

    def update(self, name: str, values):
        """Add values to wandb log dict for the current step."""
        if isinstance(values, (torch.Tensor, np.ndarray)):
            values = values.tolist()
        tag = self._make_scalar_hierarchical_tag(name)

        if isinstance(values, (float, int)):
            self._current_step_logs[tag] = values
        else:
            # We can log the mean of the values if it's an array
            self._current_step_logs[tag] = float(np.mean(values))

        super().update(name, values)

    def add_figure(self, name: str, figure: plt.Figure | np.ndarray | str, epoch: int):  # pyright: ignore[reportPrivateImportUsage]
        """Add figure to wandb."""
        if wandb.run is None:
            return

        tag = self._make_scalar_hierarchical_tag(name)
        if isinstance(figure, plt.Figure):  # pyright: ignore[reportPrivateImportUsage]
            wandb.log({tag: wandb.Image(figure), "epoch": epoch}, step=self.global_steps[self._idx])
        else:
            if isinstance(figure, np.ndarray):
                # Ensure it's correctly shaped for image (H, W, C)
                if figure.shape[0] in [1, 3, 4] and figure.shape[2] not in [1, 3, 4]:
                    figure = np.transpose(figure, (1, 2, 0))
            wandb.log({tag: wandb.Image(figure), "epoch": epoch}, step=self.global_steps[self._idx])

    def step(self):
        """Step wandb logger."""
        if wandb.run is not None and self._current_step_logs:
            global_step = self.global_steps[self._idx]
            wandb.log(self._current_step_logs, step=global_step)
            self._current_step_logs = {}

        self._idx += 1
