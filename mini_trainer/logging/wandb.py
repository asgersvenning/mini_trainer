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


class _Unimported:
    pass


wandb = _Unimported


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
        if wandb is _Unimported:
            try:
                import wandb as _wandb

                wandb = _wandb
            except ImportError:
                wandb = None

        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it using `uv pip install mini_trainer[recommended]`, "
                "`uv sync --extra recommended`, or `uv add wandb`."
            )
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

        # Initialize wandb run if not already initialized
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
        """Add figure to wandb, with robust native SVG support."""
        if get_rank() > 0:
            return
        if wandb.run is None:
            return

        tag = self._make_scalar_hierarchical_tag(name)
        step_idx = min(self._idx, len(self.global_steps) - 1)
        global_step = self.global_steps[step_idx] if len(self.global_steps) > 0 else 0

        # W&B strictly requires monotonically increasing steps
        if getattr(wandb.run, "step", 0) > global_step:
            global_step = wandb.run.step

        # 1. Robust SVG Detection
        is_svg = False
        svg_content = ""

        if isinstance(figure, str):
            # Case A: File Path
            if figure.lower().endswith(".svg") and os.path.isfile(figure):
                try:
                    with open(figure, encoding="utf-8") as f:
                        svg_content = f.read()
                    is_svg = True
                except OSError as e:
                    get_logger().warning(f"Failed to read SVG file {figure}: {e}")

            # Case B: Raw String (Check only the first 500 chars to save memory)
            elif "<svg" in figure[:500].lower():
                svg_content = figure
                is_svg = True

        # 2. Render SVG
        if is_svg:
            # Wrapper handles Dark Mode visibility and allows scrolling if massive
            html_payload = f'<div style="background-color: white; width: 100%; overflow: auto; padding: 10px;">{svg_content}</div>'
            wandb.log({tag: wandb.Html(html_payload), "epoch": epoch}, step=global_step)
            return

        # 3. Render Standard Formats
        if isinstance(figure, plt.Figure):  # pyright: ignore[reportPrivateImportUsage]
            wandb.log({tag: wandb.Image(figure), "epoch": epoch}, step=global_step)
        elif isinstance(figure, np.ndarray):
            # Ensure it's correctly shaped for image (H, W, C)
            if figure.shape[0] in [1, 3, 4] and figure.shape[2] not in [1, 3, 4]:
                figure = np.transpose(figure, (1, 2, 0))
            wandb.log({tag: wandb.Image(figure), "epoch": epoch}, step=global_step)
        else:
            # Fallback for standard image paths (e.g., "plot.png")
            wandb.log({tag: wandb.Image(figure), "epoch": epoch}, step=global_step)

    def step(self):
        """Step wandb logger."""
        if wandb.run is not None and self._current_step_logs:
            if get_rank() == 0:
                global_step = self.global_steps[self._idx]

                # W&B strictly requires monotonically increasing steps
                if getattr(wandb.run, "step", 0) > global_step:
                    global_step = wandb.run.step

                wandb.log(self._current_step_logs, step=global_step)
            self._current_step_logs = {}

        self._idx += 1

    def synchronize_between_processes(self):
        pass
