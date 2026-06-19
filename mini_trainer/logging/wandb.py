import json
import os
import socket

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from mini_trainer import get_logger
from mini_trainer.utils import get_rank, is_dist_avail_and_initialized

from .core import BaseStatistic, _Logger, _Statistic

try:
    import wandb as _wandb

    wandb = _wandb
except ImportError:
    wandb = None


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
        global wandb
        if steps is None:
            raise TypeError(f"Initializing {WandbLogger} with `steps=None` is invalid.")

        self.global_steps = steps
        self.name = name
        self.tag = tag or "main"

        output_dir = None
        if output is not None and name is not None:
            output_dir = os.path.join(output, name)

        config = None
        dataset = None
        machine = socket.gethostname()

        if output_dir is not None:
            config_path_yaml = os.path.join(output_dir, "config.yaml")
            config_path_json = os.path.join(output_dir, "config.json")
            if os.path.exists(config_path_yaml):
                with open(config_path_yaml, encoding="utf-8") as f:
                    config = yaml.safe_load(f)
            elif os.path.exists(config_path_json):
                with open(config_path_json, encoding="utf-8") as f:
                    config = json.load(f)

            if config and "input" in config:
                dataset = os.path.basename(config["input"])

        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it using `uv pip install mini_trainer[recommended]`, "
                "`uv sync --extra recommended`, or `uv add wandb`."
            )
        if wandb.run is None:
            tags = []
            if machine:
                tags.append(machine[:64])
            if dataset:
                tag_val = os.path.join(os.path.basename(os.getcwd()), dataset)
                if len(tag_val) > 64:
                    tag_val = dataset
                tags.append(tag_val[:64])
            cwd_val = os.getcwd()
            if cwd_val:
                if len(cwd_val) > 64:
                    cwd_val = os.path.basename(cwd_val)
                tags.append(cwd_val[:64])

            tags = [t for t in tags if t]

            if is_dist_avail_and_initialized():
                run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
                settings = wandb.Settings(
                    mode="shared",
                    x_primary=(get_rank() == 0),
                    x_label=f"rank_{get_rank()}",
                )
                wandb.init(
                    project=project,
                    name=name,
                    id=run_id,
                    dir=output,
                    config=config,
                    tags=tags if tags else None,
                    settings=settings,
                )
            else:
                wandb.init(
                    project=project,
                    name=name,
                    dir=output,
                    config=config,
                    tags=tags if tags else None,
                )

        self._idx = 0
        self._internal_step = 0
        self._statistics: dict[str, _Statistic] = dict()
        self._current_step_logs = {}

        if wandb.run is not None:
            self.custom_step_key = self._make_scalar_hierarchical_tag("step")
            metric_glob = self._make_scalar_hierarchical_tag("*")
            wandb.define_metric(self.custom_step_key)
            wandb.define_metric(metric_glob, step_metric=self.custom_step_key)

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
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()
            if values.numel() == 1:
                values = values.item()
            else:
                values = values.tolist()
        elif isinstance(values, np.ndarray):
            values = values.tolist()
        tag = self._make_scalar_hierarchical_tag(name)

        if isinstance(values, (float, int)):
            self._current_step_logs[tag] = values
        else:
            self._current_step_logs[tag] = float(np.mean(values))

        super().update(name, values)

    def add_figure(self, name: str, figure: plt.Figure | np.ndarray | str, epoch: int):
        """Add figure to wandb, queued to commit atomically with step()."""
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it using `uv pip install mini_trainer[recommended]`, "
                "`uv sync --extra recommended`, or `uv add wandb`."
            )
        if get_rank() > 0 or wandb.run is None:
            return

        tag = self._make_scalar_hierarchical_tag(name)
        is_svg = False
        svg_content = ""

        if isinstance(figure, str):
            if figure.lower().endswith(".svg") and os.path.isfile(figure):
                try:
                    with open(figure, encoding="utf-8") as f:
                        svg_content = f.read()
                    is_svg = True
                except OSError as e:
                    get_logger().warning(f"Failed to read SVG file {figure}: {e}")
            elif "<svg" in figure[:500].lower():
                svg_content = figure
                is_svg = True

        if is_svg:
            html_payload = f'<div style="background-color: white; width: 100%; overflow: auto; padding: 10px;">{svg_content}</div>'
            elem = wandb.Html(html_payload)
        elif isinstance(figure, plt.Figure):  # pyright: ignore[reportPrivateImportUsage]
            elem = wandb.Image(figure)
        elif isinstance(figure, np.ndarray):
            if figure.shape[0] in [1, 3, 4] and figure.shape[2] not in [1, 3, 4]:
                figure = np.transpose(figure, (1, 2, 0))
            elem = wandb.Image(figure)
        else:
            elem = wandb.Image(figure)
        
        self._current_step_logs[tag] = elem
        self._current_step_logs["epoch"] = epoch

    def step(self):
        """Step wandb logger."""
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it using `uv pip install mini_trainer[recommended]`, "
                "`uv sync --extra recommended`, or `uv add wandb`."
            )
        if wandb.run is not None and self._current_step_logs:
            if get_rank() == 0:
                self._current_step_logs[self.custom_step_key] = self._internal_step

                step_idx = min(self._idx, len(self.global_steps) - 1)
                if self.global_steps:
                    self._current_step_logs["trainer/global_step"] = self.global_steps[step_idx]

                wandb.log(self._current_step_logs)

            self._current_step_logs = {}

        self._idx += 1
        self._internal_step += 1

    def synchronize_between_processes(self):
        pass
