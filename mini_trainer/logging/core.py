import math
import os
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from itertools import chain, repeat
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any, TextIO, TypeVar

import numpy as np

try:
    import psutil as _psutil

    psutil = _psutil
except ImportError:
    psutil = None
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn

from mini_trainer import get_logger
from mini_trainer.training import raw_confusion_matrix
from mini_trainer.utils import float_signif_decimal, get_rank, main_process_first, reduce_across_processes, write_csv_from_dict
from mini_trainer.visualization import (
    plot_class_distance_matrix,
    plot_heatmap,
    plot_probabilistic_dendrogram,
)


def format_duration(sec: int, suffix="dhms"):
    """Format duration in seconds."""
    sec = int(sec)
    days, rem = divmod(sec, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    tms = [days, hours, mins, secs]
    # First non-zero element (but clamped to len - 1)
    start = next((i for i, v in enumerate(tms[:-1]) if v), len(tms) - 1)
    return "".join(f"{t:02d}{s}" for t, s in zip(tms[start:], suffix[start:]))


class Timer:  # noqa: D101
    def __init__(self):  # noqa: D107
        self._last = None
        self._total = 0.0

    @property
    def total(self) -> int:
        if self.running:
            raise RuntimeError("Attempting to grab total of a running timer!")
        return round(self._total)

    @property
    def running(self):
        return self._last is not None

    def start(self):
        if self.running:
            raise RuntimeError("Attempting to start timer which is already running.")
        self._last = time.time()

    def stop(self):
        if not self.running or self._last is None:
            raise RuntimeError("Attempting to stop timer which is not running.")
        self._total += time.time() - self._last
        self._last = None

    def __str__(self):
        status = self.running
        if status:
            self.stop()
        retval = f"Timer[{'Running' if status else 'Stopped'}]: {format_duration(self.total)}"
        if status:
            self.start()
        return retval

    def __repr__(self):
        return str(self)


class ETA:
    """ETA tracker."""

    def __init__(self, total_steps: int, smoothing: float = 0.3, fmt: str = "%H:%M:%S"):
        """Create ETA tracker.

        Args:
            total_steps: total number of steps expected
            smoothing: EMA smoothing factor (between 0 and 1)
            fmt: strftime-format string for duration display, e.g. "%H:%M:%S"
        """
        self.total_steps = total_steps
        self.smoothing = smoothing
        self.fmt = fmt
        self._start_time = time.time()
        self._last_time = self._start_time
        self._ema = None
        self._step = 0

    def __len__(self):
        return self.total_steps

    def __bool__(self):
        return self._step < self.total_steps

    @property
    def remaining(self):
        return max(self.total_steps - self._step, 0)

    @property
    def eta(self):
        if self._ema is None:
            return None
        return self.remaining * self._ema

    def step(self, steps: int = 1):
        """Step ETA.

        Args:
            steps: Number of steps to progress (default=1).

        Returns:
            Estimated number of (fractional) seconds left.
        """
        now = time.time()
        elapsed = now - self._last_time
        per_step = elapsed / steps
        self._ema = per_step if self._ema is None else self.smoothing * per_step + (1 - self.smoothing) * self._ema
        self._step += steps
        self._last_time = now
        return self.eta

    def __str__(self):
        used_str = format_duration(time.time() - self._start_time)
        if self._ema is None:
            return f"{used_str}/??"
        eta_str = format_duration(self.eta)
        return f"{used_str}/{eta_str}"


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """The accuracy over the k top predictions for the specified values of k."""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class _Statistic:
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    sum: float | None = None

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __bool__(self):
        return len(self) > 0

    def __str__(self):
        raise NotImplementedError()

    @property
    def data(self) -> list[float]:
        raise NotImplementedError()

    def update(self, value, *args, **kwargs):
        raise NotImplementedError()


S = TypeVar("S", bound=_Statistic)


class _Logger:
    """The __init__ method of any subclass should swallow any additional keyword arguments.

    Methods:
        **`add_stat`**: Add a new statistic container.
        **`get`**: Get a statistic container.
        **`update`**: Add new values to a statistic.
        **`step`**: Function to indicate that the current iteration has completed.
    """

    def __init__(
        self,
        steps: list[int] | None = None,
        tag: type | str | None = None,
        name: str | None = None,
        output: str | None = None,
        **kwargs,
    ):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def add_stat(self, name: str, container: _Statistic):
        raise NotImplementedError()

    def get(self, name: str) -> _Statistic:
        return self.statistics[name]

    def rename_stat(self, name: str, new: str):
        self.statistics[new] = self.statistics.pop(name)

    def update(self, name, values: float | int | list[float | int] | torch.Tensor | np.ndarray):
        if isinstance(values, np.ndarray):
            values = values.tolist()  # ty:ignore[no-matching-overload]
        self.get(name).update(values)

    def add_figure(self, name: str, figure: plt.Figure | str, **kwargs):
        pass

    def step(self):
        """This function may not be necessary for your logger."""
        pass

    @property
    def statistics(self) -> dict[str, _Statistic]:
        raise NotImplementedError()

    def synchronize_between_processes(self):
        raise NotImplementedError()


L = TypeVar("L", bound=_Logger)


class SmoothedValue(_Statistic):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt_vars: list[str] = ["median"]):  # noqa: D107
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.min = None
        self.fmt_vars = fmt_vars
        self.fmt_digs: dict[str, deque[int]] = {var: deque(maxlen=window_size * 5) for var in self.fmt_vars}

    def __len__(self):
        return self.count

    def update(self, value: float | int | torch.Tensor | Sequence[float | int], n=1):
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        self.deque.append(value)
        if not isinstance(value, (float, int)):
            for v in value:
                self.update(float(v), n=n)
            return

        if self.min is None or value < self.min:
            self.min = float(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque!"""
        min_val = self.min if self.min is not None else 0.0
        total_val = self.total
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.item()
        if isinstance(total_val, torch.Tensor):
            total_val = total_val.item()
        t = reduce_across_processes([self.count, total_val, min_val])
        t = t.tolist()
        self.count = int(t[0])
        self.total = float(t[1])
        self.min = float(t[2]) if self.count > 0 else None

    @property
    def median(self):
        if not self.deque:
            return float("nan")
        if isinstance(self.deque[0], torch.Tensor):
            return torch.stack(list(self.deque)).median().item()
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return float("nan")
        if isinstance(self.deque[0], torch.Tensor):
            return torch.stack(list(self.deque)).mean().item()
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return float("nan")
        val = self.total / self.count
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    @property
    def mean(self):
        return self.global_avg

    @property
    def max(self):
        if not self.deque:
            return float("nan")
        if isinstance(self.deque[0], torch.Tensor):
            return torch.stack(list(self.deque)).max().item()
        return max(self.deque)

    @property
    def sum(self):
        val = self.total
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    @property
    def value(self):
        if self.deque:
            val = self.deque[-1]
            if isinstance(val, torch.Tensor):
                return val.item()
            return val
        else:
            return float("nan")

    def __str__(self):
        parts = []
        for var in self.fmt_vars:
            value = getattr(self, var)
            cur_digs = float_signif_decimal(value)
            self.fmt_digs[var].append(cur_digs)
            digs = max(self.fmt_digs[var])
            part = f"{value:>{digs + 3}.{digs}f}"
            parts.append(part)
        return "/".join(parts)


class MetricLogger(_Logger):  # noqa: D101
    def __init__(self, delimiter=" | ", printer=print, **kwargs):  # noqa: D107
        self._statistics: defaultdict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.printer = printer

    @property
    def statistics(self):
        return self._statistics

    def __getattr__(self, attr):
        if attr in self.statistics:
            return self.statistics[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, stat in self.statistics.items():
            loss_str.append(f"{name}: {str(stat)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for stat in self.statistics.values():
            stat.synchronize_between_processes()

    def add_stat(self, name, container):
        if not isinstance(container, SmoothedValue):
            raise TypeError("Only `SmoothedValue` containers for statistics in `MetricLogger` supported, not: {type(container)}")
        self.statistics[name] = container


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.

    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):  # noqa: D107
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class BaseStatistic(_Statistic):  # noqa: D101
    def __init__(self, values: list[float | int] | None = None):
        """A basic thread-safeish statistic container.

        Args:
            values: A list of values to initially populate the statistic, optional.
        """
        self.values: list[float] = []
        self.lock = RLock()
        self.min: float = float("inf")
        self.max: float = float("-inf")
        self.sum: float = 0
        self.digs: deque[int] = deque(maxlen=30)
        self._len: int = 0
        if values is not None:
            self.update(values)

    def __len__(self):
        return self._len

    def __getitem__(self, i: int | slice):
        get_logger().warning("The behaviour of this is currently ill defined")  # TODO!
        return self.values[i]

    def __iter__(self):
        """Iter on the base statistic is not thread safe,
        if you want to ensure this, you must acquire the lock manually first.
        """
        yield from self.values

    @property
    def data(self):
        return self.values

    def update(
        self,
        value: float | int | list[int | float] | np.ndarray | torch.Tensor,
        **kwargs,
    ):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            try:
                value = torch.as_tensor(value)
            except Exception as e:
                raise RuntimeError(f"Unable to update statistic ({self}) with heterogenous data or invalid type.") from e
        value = value.detach().cpu()

        n = value.numel()
        if n == 0:
            return

        if n == 1:
            tmin = tmax = tsum = float(value.item())
        else:
            tmin = float(value.min().item())
            tmax = float(value.max().item())
            tsum = float(value.sum().item())

        with self.lock:
            self.min = tmin if self.min is None else min(self.min, tmin)
            self.max = tmax if self.max is None else max(self.max, tmax)
            self.sum = tsum if self.sum is None else self.sum + tsum
            self._len += n

    @property
    def mean(self):
        if self.sum is None or len(self) == 0:
            return float("nan")
        val = self.sum / len(self)
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def __str__(self):
        with self.lock:
            if not self:
                return "NaN |0|"
            m = self.mean  # noqa: E221
            mn = self.min
            mx = self.max
        if isinstance(m, torch.Tensor):
            m = float(m.item())
        if isinstance(mn, torch.Tensor):
            mn = float(mn.item())
        if isinstance(mx, torch.Tensor):
            mx = float(mx.item())
            # filter(lambda x: math.isfinite(x) and x != 0, map(abs, [m, mn, mx])
        digs = float_signif_decimal(float(min((abs(x) for x in [m, mn, mx] if math.isfinite(x) and x != 0), default=1.0)))
        self.digs.append(digs)
        digs = max(self.digs)
        return f"{m:>{digs + 2}.{digs}f} [{mn:>{digs + 3}.{digs}f}; {mx:>{digs + 3}.{digs}f}]"

    def __repr__(self):
        return f"BaseStatistic({str(self)})|{len(self)}|"


def compute_aligned_steps(target_length: int, origin_length: int, total_epochs: int, current_epoch: int) -> list[int]:
    """Rescale steps to align origin with target."""
    if not (0 <= current_epoch < total_epochs):
        raise ValueError(f"current_epoch must be in [0, {total_epochs}), got {current_epoch!r}")

    start = current_epoch * target_length
    end = (current_epoch + 1) * target_length - 1  # noqa: E221

    return [int(round(step)) for step in np.linspace(start, end, num=origin_length)]


class MultiLogger:
    """Multi-backend training/evaluation logger.

    Orchestrates one or more concrete loggers (e.g., terminal metrics, TensorBoard)
    and provides a single interface for recording statistics, figures and
    heterogeneous artifacts per step and per epoch.

    Args:
        train_loader: Training dataloader; used to build aligned global steps.
        val_loader: Validation dataloader; used to build aligned global steps.
        epochs: Total number of epochs used to pre-compute global steps.
        output: Directory to store serialized logs (JSON).
        name: Base filename for the serialized log (auto-incremented).
        statistics: Names of statistics to track and expose to backends.
        private_statistics: Internal statistics not forwarded to backends.
        logger_cls: Concrete logger classes to instantiate.
        logger_cls_extra_kwargs: Per-logger extra keyword arguments.
        logger_cls_stat_factory: Factories for statistic containers per logger.
        canonical_statistic: Name of the main metric (used for returns and summaries).
        clear_store_on_update: If True, clears transient storage at epoch/phase switch.
        verbose: If True, prints summaries and progress information.
    """

    @staticmethod
    def _reset_cuda_memory_stats():
        """Reset CUDA peak memory stats on the current device.

        Notes:
            This intentionally avoids synchronization to reduce performance
            impact. As a result, memory statistics can be slightly biased
            towards lower values, but remain consistent across steps.
        """
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                # Guard against environments without CUDA context, etc.
                pass

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        output: str | None,
        name: str,
        statistics: list[str] = ["acc1", "acc5", "loss", "lr", "item/s", "mem", "step", "time", "eta", "epoch", "type"],
        private_statistics: list[str] = ["step", "time", "eta", "epoch", "type"],
        logger_cls: list[type[_Logger]] = [MetricLogger],
        logger_cls_extra_kwargs: list[dict[str, Any]] = [],
        logger_cls_stat_factory: list[Callable[[], _Statistic]] = [lambda: SmoothedValue(window_size=10, fmt_vars=["avg"])],
        clear_store_on_update: bool = True,
        verbose: bool = False,
    ):
        """Create multi-logger orchestrator."""
        self.total_epochs = epochs
        self.private_statistics = private_statistics
        self.statistics = sorted(list(set(statistics) - set(self.private_statistics)), key=lambda x: statistics.index(x))
        self.statistics_storage = defaultdict(list)
        self.heterogeneous_storage = defaultdict(list)
        self.output = output
        self.name = name
        self.output_dir = None if self.output is None else os.path.join(self.output, self.name, "logs")
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.clear_store_on_update: bool = clear_store_on_update

        if get_rank() > 0:
            logger_cls = [cls for cls in logger_cls if cls.__name__ != "TensorboardLogger"]

        self.logger_cls: list[type[_Logger]] = logger_cls
        self.logger_cls_extra_kwargs: list[dict[str, Any]] = logger_cls_extra_kwargs
        self.logger_cls_stat_factory: list[Callable[[], _Statistic]] = logger_cls_stat_factory
        self.verbose: bool = verbose

        # Get aligned steps (mainly for use with tensorboard)
        self.train_steps, self.val_steps = self.compute_steps(self.total_epochs, len(train_loader), len(val_loader))
        self.total_steps: int = sum(map(len, self.train_steps)) + sum(map(len, self.val_steps))

        # Initialize dynamic attributes
        self._type_timing: defaultdict[str, Timer] = defaultdict(Timer)
        self._last_save: float = time.time()
        self._step: int = 0
        self._start_time: float | None = None
        self.eta: ETA | None = None
        self._current_loggers: list[_Logger] = []
        self._epoch: int | None = None
        self._type: str | None = None
        self._batch_size: int | None = None
        self._idx: int | None = None
        self._n_classes: int | None = None
        self._soft_confusion_matrix: dict[str, torch.Tensor] = dict()
        self._finished = False

    def start_timing(self):
        assert self._type is not None
        self._type_timing[self._type].start()

    def stop_timing(self):
        assert self._type is not None
        self._type_timing[self._type].stop()

    def timings(self, format: bool = True, title: str = "", prefix="\n", fmt: str = "{name} : {time}"):
        timings = {k: v.total for k, v in self._type_timing.items()}
        if not format:
            return timings
        return f"{title}{prefix}" + prefix.join([fmt.format(name=name, time=format_duration(time)) for name, time in timings.items()])

    def is_train(self):
        return isinstance(self._type, str) and self._type.lower().strip().startswith("train")

    def compute_steps(self, epochs: int, train_steps: int, val_steps: int) -> tuple[list[list[int]], list[list[int]]]:
        """Compute aligned steps for train and validation."""
        return [list(range(e * train_steps, (e + 1) * train_steps)) for e in range(epochs)], [
            compute_aligned_steps(train_steps, val_steps, epochs, e) for e in range(epochs)
        ]

    @property
    def canonical_statistic(self):
        return self.statistics[0]

    @property
    def steps(self):
        if self._epoch is None:
            raise RuntimeError("Attempted to retrieve steps before starting an epoch (i.e. training).")
        if self.is_train():
            return self.train_steps[self._epoch]
        else:
            return self.val_steps[self._epoch]

    def store(self, name: str, value: Any):
        self.heterogeneous_storage[name].append((value, self._epoch, self._type))

    def update(self, epoch: int, type: str):
        """Begin a new phase (train/eval) for a given epoch.

        Initializes/rotates backend loggers, clears transient storage (if
        configured), resets per-phase CUDA peak memory, and prepares aligned
        global steps for the selected phase.

        Args:
            epoch: Zero-based epoch index.
            type: Phase name (e.g., ``"train"`` or ``"eval"``).
        """
        if self._finished:
            raise RuntimeError("Attempted to update finished logger!")
        if self._start_time is None:
            self._start_time = time.time()
            self.eta = ETA(self.total_steps, 0.999)
        else:
            self.save()
            self._store_summary()
            if self.clear_store_on_update:
                self.statistics_storage = defaultdict(list)
                self.heterogeneous_storage = defaultdict(list)
        self._epoch = epoch
        self._type = type
        self._reset_cuda_memory_stats()
        self._current_loggers = []
        self._soft_confusion_matrix = dict()
        for cls, kwargs, stat_factory in zip(
            self.logger_cls,
            chain(self.logger_cls_extra_kwargs, repeat(dict())),
            chain(self.logger_cls_stat_factory, repeat(BaseStatistic)),
        ):
            this_logger = cls(steps=self.steps, tag=type, name=self.name, output=self.output, **kwargs)
            for stat in self.statistics:
                this_logger.add_stat(stat, stat_factory())
            self._current_loggers.append(this_logger)

    def step(self):
        """Advance one logging step.

        Records bookkeeping statistics (step/time/eta/epoch/type) and then
        resets the CUDA peak memory so that the next call to ``log_memory_use``
        reflects the peak for the next batch only.
        """
        if self._finished:
            raise RuntimeError("Attempted to step finished logger.")
        assert self.eta is not None
        self.log_statistic(step=self._step)
        self._step += 1
        self.eta.step()
        for logger in self.loggers:
            logger.step()
        self.log_statistic(time=time.time())
        self.log_statistic(eta=self.eta.eta)
        self.log_statistic(epoch=self._epoch)
        self.log_statistic(type=self._type)
        # Reset peak CUDA memory so next batch's max reflects only the next step's work
        self._reset_cuda_memory_stats()

    @property
    def loggers(self) -> list[_Logger]:
        if self._current_loggers is None:
            raise RuntimeError("Attempted to log statistics before initializing loggers.")
        return self._current_loggers

    def get_logger(self, cls: type[L]) -> L:
        for logger in self.loggers:
            if isinstance(logger, cls):
                return logger
        raise KeyError(f"No logger of type {cls} found.")

    def rename_stat(self, current2new: dict[str, str]):
        if len(current2new) > 1:
            for k, v in current2new.items():
                self.rename_stat({k: v})
            return
        current = list(current2new)[0]
        new = current2new[current]
        self.statistics[self.statistics.index(current)] = new
        for logger in self.loggers:
            logger.rename_stat(current, new)

    def log_statistic(self, **kwargs):
        for stat, value in kwargs.items():
            if stat not in self.private_statistics:
                if stat not in self.statistics:
                    self.statistics.append(stat)
                    for logger, stat_factory in zip(self.loggers, chain(self.logger_cls_stat_factory, repeat(BaseStatistic))):
                        logger.add_stat(stat, stat_factory())
                for logger in self.loggers:
                    logger.update(stat, value)
            # Disable for now to avoid OOM
            # if isinstance(value, (torch.Tensor, np.ndarray)):
            #     value = value.tolist()
            # if isinstance(value, (tuple, list)):
            #     self.statistics_storage[stat].extend(value)
            # else:
            #     self.statistics_storage[stat].append(value)

    @property
    def data(self):
        return {
            "statistics": dict(self.statistics_storage),
            "extra": dict(),  # dict(self.heterogeneous_storage)
        }

    def _store_summary(self):
        if self.output_dir is not None:
            sum = self.summary()
            sum["epoch"] = self._epoch
            sum["type"] = self._type
            write_csv_from_dict({k: [v] for k, v in sum.items()}, os.path.join(self.output_dir, "summary.csv"))
        return

    def save(self, fp: str | TextIO | None = None, encoding: str = "utf-8", **kwargs):
        pass  # Disable saving for now to avoid OOM
        # ext = ".json"
        # if self._start_time is None:
        #     warnings.warn("Attempting to save logs before starting the loggers is a no-op!")
        #     return
        # if self._epoch is None:
        #     raise NotImplementedError(
        #       f'Saving logs while {self._epoch=} is not supported and probably not meaningful.
        #       'If this happens you are probably doing something wrong!'
        #     )
        # if self._type is None:
        #     raise NotImplementedError(
        #       f'Saving logs while {self._type=} is not supported and probably not meaningful. '
        #       'If this happens you are probably doing something wrong!'
        #     )
        # if fp is None:
        #     if self.output_dir is None:
        #         return
        #     output_dir, name = self.output_dir, f'log_{self._type}_epoch{self._epoch}'
        # elif isinstance(fp, TextIO):
        #     json.dump(self.data, fp, **kwargs)
        #     return
        # elif isinstance(fp, str):
        #     output_dir, name = os.path.split(os.path.abspath(fp))
        #     _, ext = os.path.splitext(name)
        #
        # if os.path.exists(os.path.join(output_dir, name)):
        #     name = increment_name_dir(name, output_dir)
        # fp = os.path.join(output_dir, name + ext)
        #
        # temp_file_name = None
        # try:
        #     with NamedTemporaryFile("w", encoding=encoding, suffix=".json", delete=False) as tmpfile:
        #         json.dump(self.data, tmpfile, **kwargs)
        #         tmpfile.flush()
        #         os.fsync(tmpfile.fileno())
        #         temp_file_name = tmpfile.name
        #
        #     shutil.move(temp_file_name, fp)
        #     self._last_save = time.time() # Assuming self._last_save is defined
        # except Exception as e:
        #     if temp_file_name and os.path.exists(temp_file_name):
        #         try:
        #             os.remove(temp_file_name)
        #         except OSError:
        #             pass # Suppress error during cleanup
        #     raise

    def finish(self):
        self._store_summary()
        self.save()
        self._start_time = None
        self.eta = None
        self.statistics_storage = defaultdict(list)
        self.heterogeneous_storage = defaultdict(list)
        self._epoch = None
        self._type = None
        self._reset_cuda_memory_stats()
        self._current_loggers: list[_Logger] = []
        self._soft_confusion_matrix = dict()
        self._finished = True

    def log_batch(self, batch):
        pass

    def log_accuracy(self, target: torch.Tensor, prediction: list[torch.Tensor] | torch.Tensor):
        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]
            target = target[:, 0]
        if not isinstance(prediction, list):
            if self._n_classes is None:
                self._n_classes = prediction.shape[1]
            if not self.is_train():
                self.render_soft_confusion_matrix(target, prediction)
                self.log_labels_predictions(target.tolist(), prediction.argmax(1).tolist())
            if prediction.shape[1] >= 5:
                acc1, acc5 = accuracy(prediction, target, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(prediction, target, topk=(1,))[0], torch.nan
            self.log_statistic(acc1=acc1, acc5=acc5)
        else:
            for lvl in range(len(prediction)):
                tp = prediction[lvl]
                tl = target[:, lvl]
                self._n_classes = tp.shape[1]
                if not self.is_train():
                    self.render_soft_confusion_matrix(tl, tp, level=lvl)
                    self.log_labels_predictions(tl.tolist(), tp.argmax(1).tolist(), level=lvl)
                if tp.shape[1] >= 5:
                    acc1, acc5 = accuracy(tp, tl, topk=(1, 5))
                else:
                    acc1, acc5 = accuracy(tp, tl, topk=(1,))[0], torch.nan
                if lvl == 0:
                    self.log_statistic(acc1=acc1, acc5=acc5)
                else:
                    self.log_statistic(**{f"acc1/lvl{lvl}": acc1, f"acc5/lvl{lvl}": acc5})

    def log_labels_predictions(self, labels: list[int], predictions: list[int], level: int | None = None):
        if self.is_train():
            return
        if level is None:
            self.store("labels", labels)
            self.store("predictions", predictions)
        else:
            self.store(f"labels/lvl{level}", labels)
            self.store(f"predictions/lvl{level}", predictions)

    def log_loss(self, loss: torch.Tensor | list[torch.Tensor]):
        if isinstance(loss, (list, tuple)) and len(loss) == 1:
            loss = loss[0]
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            self.log_statistic(loss=loss.detach())
        else:
            self.log_statistic(loss=sum(loss).detach())  # type: ignore
            for i, term in enumerate(loss):
                if term.numel() != 1:
                    raise RuntimeError(f"Expected scalar loss term but found {term.shape}.")
                self.log_statistic(**{f"loss/lvl{i}": term.detach()})

    def log_optim(self, optimizer: torch.optim.Optimizer | None):
        if optimizer is None:
            self.log_statistic(lr=float("nan"))
        else:
            grps = optimizer.param_groups
            self.log_statistic(lr=grps[0]["lr"])
            if len(grps) > 1:
                for grp in grps:
                    self.log_statistic(**{f"lr/{grp['name']}": grp["lr"]})

    def log_speed(self, start_time: float):
        self.log_statistic(**{"item/s": self._batch_size / (time.time() - start_time)})

    def log_memory_use(self):
        """Log per-batch peak CUDA memory usage (no sync).

        Notes:
            Uses ``torch.cuda.max_memory_allocated()`` which reports the peak
            allocation since the last reset. The logger resets the peak at the
            end of each step and at phase boundaries, so this value reflects a
            best-effort per-batch peak. No device synchronization is performed
            to avoid performance impact, so values may be slightly under the
            true peak but are consistent across steps.
        """
        MB = 1024.0**2
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / MB
        else:
            if psutil is not None:
                mem = psutil.Process().memory_info().rss / MB
            else:
                mem = 0.0
        self.log_statistic(mem=mem)

    def default_consume(
        self,
        index: int,
        batch: torch.Tensor,
        target: torch.Tensor,
        prediction: list[torch.Tensor] | torch.Tensor,
        loss: Any,
        optimizer: torch.optim.Optimizer,
        start_time: float,
    ):
        # These are set first, so they may be used while logging the other statistics
        self._idx = index
        self._batch_size = len(batch)

        self.log_batch(batch)
        self.log_optim(optimizer)
        self.log_accuracy(target, prediction)
        self.log_loss(loss)
        self.log_memory_use()
        self.log_speed(start_time)

        self._idx = self._batch_size = None

    @torch.no_grad()
    def consume(self, **kwargs):
        default_consume_kwargs = ["index", "batch", "target", "prediction", "loss", "optimizer", "start_time"]
        dca = {key: kwargs.pop(key) for key in default_consume_kwargs}

        self.log_statistic(**kwargs)
        if len(dca) == len(default_consume_kwargs):
            self.default_consume(**dca)
        self.step()

    def status(self):
        if not self.loggers:
            stats_str = ""
        else:
            stats_list = [f"{name}: {str(self.loggers[0].statistics[name])}" for name in self.statistics[: min(4, len(self.statistics))]]
            stats = " | ".join(stats_list)
            stats_str = f" | {stats}"
        epoch = self._epoch
        if epoch is None:
            epoch = "?"
        else:
            epoch += 1
        return f"E{epoch}/{self.total_epochs} ({self._step / self.total_steps:.1%} {self.eta}){stats_str}"

    def _last_epoch_value(self, stat: str):
        # Assume that the first logger is "canonical"
        if not self.loggers:
            return 0.0
        return self.loggers[0].statistics[stat].mean

    def summary(self, stats: list[str] | None = None):
        if stats is None:
            stats = self.statistics
        return {stat: self._last_epoch_value(stat) for stat in stats}

    def summary_string(self, stats: list[str] | None = None):
        if stats is None:
            stats = self.statistics
        values = self.summary(stats)
        parts = []
        for stat in stats:
            if stat not in values:
                continue
            value = values[stat]
            if not bool(np.isfinite(value)):
                continue
            part = f"{stat}={value:>5.{float_signif_decimal(value)}f}"
            parts.append(part)
        return " | ".join(parts)

    @property
    def canonical_scalar(self):
        return self._last_epoch_value(self.canonical_statistic)

    def render_soft_confusion_matrix(self, labels: torch.Tensor, predictions: torch.Tensor, level: int = 0):
        predictions = predictions.softmax(dim=1)
        cf = self._soft_confusion_matrix.get(level, None)
        new_cf = False
        if cf is None:
            new_cf = True
            n_cls = predictions.shape[1]
            cf = torch.zeros((n_cls, n_cls), dtype=torch.float32)
        if not (isinstance(cf, torch.Tensor) and cf.dtype == torch.float32 and cf.device.type == "cpu"):
            raise RuntimeError(f'Unexpected soft confusion matrix found {cf}, expected a torch.Tensor of torch.float32 with device="cpu".')
        grps = labels.unique()
        for gidx in grps:
            lmask = labels == gidx
            cf[gidx] += predictions[lmask, :].sum(dim=0).float().cpu()
        if new_cf:
            self._soft_confusion_matrix[level] = cf

    def confusion_matrix(self):
        lvl = None
        figs = dict()
        while True:
            # Check if there is one or multiple levels, and if so if the current level exists
            if lvl is None:
                counts = {"labels": [], "predictions": []}
                lvl = 0
                if any([key not in self.heterogeneous_storage for key in counts]):
                    continue
            else:
                counts = {f"labels/lvl{lvl}": [], f"predictions/lvl{lvl}": []}
                if any([key not in self.heterogeneous_storage for key in counts]):
                    break

            # Find label and prediction combinations (at the current level if applicable)
            hits = 0
            for what in counts:
                for cls_idxs, epoch, tp in reversed(self.heterogeneous_storage[what]):
                    if epoch != self._epoch:
                        break
                    ctp = tp.lower().strip()
                    if not (ctp.startswith("val") or ctp.startswith("eval")):
                        continue
                    hits += 1
                    counts[what].extend(cls_idxs)
            if hits == 0:
                get_logger().warning(f"No labels or predictions found for {self._epoch}!")

            # Create confusion matrix from counts
            cm = raw_confusion_matrix(*counts.values())
            m = cm.sum(axis=1) > 0
            cm = cm[m][:, m]
            if not bool(np.any(np.isfinite(cm))):
                get_logger().warning(f"Confusion matrix has no valid values, produced from counts: {counts}")

            cm_lab = "Confusion matrix"
            if lvl is not None:
                cm_lab += f"/lvl{lvl}"
            figs[cm_lab] = plot_heatmap(cm)

            lvl += 1

        if len(self._soft_confusion_matrix) == 0:
            pass
        elif len(self._soft_confusion_matrix) == 1:
            cm = list(self._soft_confusion_matrix.values())[0]
            zm = cm.max(dim=1).values < (1 / cm.sum())
            cm = cm[~zm][:, ~zm]
            cm_rs = cm.sum(dim=1, keepdim=True)
            cm = cm / cm_rs
            cm = cm.clamp(1 / cm_rs.sum().item(), 1)
            figs["Soft confusion matrix"] = plot_heatmap(cm)
        else:
            for lvl, cm in self._soft_confusion_matrix.items():
                zm = cm.max(dim=1).values < (1 / cm.sum())
                cm = cm[~zm][:, ~zm]
                cm_rs = cm.sum(dim=1, keepdim=True)
                cm = cm / cm_rs
                cm = cm.clamp(1 / cm_rs.sum().item(), 1)
                figs[f"Soft confusion matrix/lvl{lvl}"] = plot_heatmap(cm)
        return figs

    def add_figure(self, name: str, figure: Figure | np.ndarray | torch.Tensor | str):
        for logger in self.loggers:
            logger.add_figure(name=name, figure=figure, epoch=self._epoch)
        if isinstance(figure, Figure):
            plt.close(figure)

    def synchronize_between_processes(self):
        for logger in self.loggers:
            if hasattr(logger, "synchronize_between_processes"):
                logger.synchronize_between_processes()

    def figures(self, model: nn.Module | None):
        cm_figs = self.confusion_matrix()
        for lab, fig in cm_figs.items():
            self.add_figure(lab, fig)

        with main_process_first():
            if get_rank() == 0 and model is not None:
                cdm_fig = plot_class_distance_matrix(model)
                self.add_figure("Class distance matrix", cdm_fig)
                try:
                    pd_fig, _ = plot_probabilistic_dendrogram(model)
                    with NamedTemporaryFile(suffix=".svg") as tmp_file:
                        pd_fig.savefig(tmp_file.name, bbox_inches="tight")
                        self.add_figure("Probabilistic dendrogram", tmp_file.name)
                except Exception as e:
                    warnings.warn(f"Warning: Failed to plot probabilistic dendrogram: {e}", UserWarning)
