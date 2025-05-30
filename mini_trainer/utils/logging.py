import datetime
import json
import os
import shutil
import time
from collections import defaultdict, deque
from itertools import chain, repeat
from tempfile import NamedTemporaryFile
from threading import RLock
from types import GeneratorType
from typing import Any, Callable, Optional, TextIO, Type, TypeVar, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn

from mini_trainer.utils import (cuda_memory_stats, float_signif_decimal,
                                increment_name_dir, reduce_across_processes)
from mini_trainer.utils.plot import (named_confusion_matrix, plot_heatmap,
                                     plot_model_class_distance,
                                     raw_confusion_matrix)


def format_duration(sec : int, suffix="dhms"):
    sec = int(sec)
    days, rem = divmod(sec, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    tms = [days, hours, mins, secs]
    # First non-zero element (but clamped to len - 1)
    start = next((i for i, v in enumerate(tms[:-1]) if v), len(tms) - 1)
    return "".join(f"{t:02d}{s}" for t, s in zip(tms[start:], suffix[start:]))

class ETA:
    def __init__(
            self, 
            total_steps : int, 
            smoothing : float=0.3, 
            fmt : str="%H:%M:%S"
        ):
        """
        Args:
            total_steps (int): total number of steps expected
            smoothing (float): EMA smoothing factor (between 0 and 1)
            fmt (str): strftime-format string for duration display, e.g. "%H:%M:%S"
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

    def step(self, steps : int=1):
        """
        Args:
            steps (int): Number of steps to progress (default=1).

        Returns:
            ETA (float): Estimated number of (fractional) seconds left.
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

class _ResultsCollector:
    """
    This is an abstract base class, it is likely easier to subclass `BaseResultCollector` instead.

    If you are using the `mini_trainer` train and prediction scripts/APIs, it is very unlikely that this is the correct class to subclass.
    However, if you are building entirely new train and/or predictions scripts/APIs, it may be an option.
    """
    def collect(
            self, 
            *args, 
            **kwargs
        ):
        raise NotImplementedError('Result collectors must have a `collect` class method.')
    
    def evaluate(self):
        raise NotImplementedError('Result collector must have a `evaluate` class method.')
    
    @property
    def data(self):
        raise NotImplementedError('Result collector must have `data` class propery suitable for JSON serialization.')

class BaseResultCollector(_ResultsCollector):
    def __init__(
            self, 
            idx2cls : dict[int, str], 
            verbose : bool=False, 
            training_format : bool=False,
            additional_attributes : Optional[list[str]]=None, 
            *args, 
            **kwargs
        ):
        self.paths = []
        self.preds = []
        self.confs = []
        self.idx2cls = idx2cls
        self._training_format = training_format
        self.verbose = verbose
        self._extra_attr = set(additional_attributes or [])
        if self._training_format:
            self._extra_attr.add("labels")
        for attr in self._extra_attr:
            setattr(self, attr, [])

    def collect(self, paths : list[str], predictions : torch.Tensor, **kwargs):
        self._collect_base_attributes(paths, predictions)
        if self._training_format and "labels" not in kwargs:
            kwargs["labels"] = [os.path.basename(os.path.dirname(path)) for path in paths]
        self._collect_extra_attributes(**kwargs)

    def _collect_base_attributes(self, paths : list[str], predictions : torch.Tensor):
        """
        Override in subclasses!
        """
        self.paths.extend(paths)
        self.preds.extend([self.idx2cls[idx] for idx in predictions.argmax(1).tolist()])
        self.confs.extend(predictions.softmax(1).max(1).values.tolist())

    def _collect_extra_attributes(self, **kwargs : Union[list, tuple, GeneratorType, np.ndarray, torch.Tensor]):
        if len(self._extra_attr) == 0:
            return
        if not all([attr in kwargs for attr in self._extra_attr]):
            raise ValueError(f'To ensure proper ordering and avoid data loss it is required to always pass all extra attributes ([{", ".join(self._extra_attr)}])')
        for key, value in kwargs.items():
            if isinstance(value, list):
                pass
            elif isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.tolist()
                if not isinstance(value, list):
                    raise ValueError(
                        f'Value passed for {key} is likely a zero-dimensional (scalar) array/tensor containing a {type(value)}.\n' \
                        "If you want to pass a single value, it should still be contained in a 1-dimensional array/tensor:\n" \
                        "\tIncorrect: `torch.tensor(1)`/`np.array(1)`\n" \
                        "\tCorrect: `torch.tensor([1])`/`np.array([1])`"
                    )
            elif isinstance(value, (tuple, GeneratorType)):
                value = list(value)
            else:
                raise TypeError(f'Unexpected value type `{type(value)}` supplied for {key}.')
            getattr(self, key).extend(value)

    def eval_label_fn(self, data : dict, outdir : Optional[str], save : bool, prefix : str="", **kwargs):
        if kwargs:
            raise RuntimeError(f'Unknown arguments ([{", ".join(kwargs)}]) passed. Perhaps you forgot to implement the intended `eval_label_fn` in your subclass.')
        if save and not isinstance(outdir, str):
            raise RuntimeError("Attempted to save evaluated results against labels without specifying an output directory.")
        return named_confusion_matrix(
            results=data, 
            idx2cls=self.idx2cls, 
            verbose=self.verbose, 
            plot_conf_mat=save and os.path.join(outdir, f"{prefix}confusion_matrix.png")
        )

    def evaluate(self, outdir : Optional[str]=None, prefix : str="", **kwargs):
        do_save = isinstance(outdir, str)
        if do_save and not os.path.isdir(outdir):
            raise IOError(f'Specified output directory (`{outdir}`) does not exist.')
        if "labels" in self._extra_attr:
            results = self.eval_label_fn(data=self.data, outdir=outdir, save=do_save, prefix=prefix, **kwargs)
            if do_save:
                with open(os.path.join(outdir, f'{prefix}eval_results.json'), "w") as f:
                    json.dump(results, f)
            return results

    @property
    def data(self):
        return {
            "paths" : self.paths,
            "preds" : self.preds,
            "confs" : self.confs,
            **{attr : getattr(self, attr) for attr in self._extra_attr}
        }

class _Statistic:
    min : Optional[float] = None
    max : Optional[float] = None
    mean : Optional[float] = None
    sum : Optional[float] = None

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self):
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

S = TypeVar('S', bound=_Statistic)

class _Logger:
    """
    The __init__ method of any subclass should swallow any additional keyword arguments.

    Methods:
        **`add_stat`**: Add a new statistic container.
        **`get`**: Get a statistic container.
        **`update`**: Add new values to a statistic.
        **`step`**: Function to indicate that the current iteration has completed. 
    """
    def __str__(self):
        raise NotImplementedError()
    
    def add_stat(self, name : str, container : _Statistic):
        raise NotImplementedError()
    
    def get(self, name : str) -> _Statistic:
        raise NotImplementedError()

    def update(self, name, values : Union[float, int, list[Union[float, int]], torch.Tensor, np.ndarray]):
        if isinstance(values, (torch.Tensor, np.ndarray)):
            values = values.tolist()
        self.get(name).update(values)

    def add_figure(self, name : str, figure : Union[plt.Figure, str], **kwargs):
        pass

    def step(self):
        """
        This function may not be necessary for your logger.
        """
        pass

    @property
    def statistics(self) -> dict[str, _Statistic]:
        raise NotImplementedError()

L = TypeVar('L', bound=_Logger)

class SmoothedValue(_Statistic):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt_vars : list[str]=["median"]):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.min = None
        self.fmt_vars = fmt_vars
        self.fmt_digs : dict[str, deque[int]] = {var : deque(maxlen=window_size*5) for var in self.fmt_vars}

    def __len__(self):
        return self.count

    def update(self, value, n=1):
        self.deque.append(value)
        if isinstance(value, (float, int)):
            if self.min is None or value < self.min:
                self.min = float(value)
        else:
            for v in value:
                v = float(v)
                if self.min is None or v < self.min:
                    self.min = v
        
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total, self.min])
        t = t.tolist()
        self.count = int(t[0])
        self.total = float(t[1])
        self.min = float(t[2])

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def mean(self):
        return self.global_avg

    @property
    def max(self):
        return max(self.deque)
    
    @property
    def sum(self):
        return self.total

    @property
    def value(self):
        if self.deque:
            return self.deque[-1]
        else:
            return float("nan")

    def __str__(self):
        parts = []
        for var in self.fmt_vars:
            value = getattr(self, var)
            cur_digs = float_signif_decimal(value)
            self.fmt_digs[var].append(cur_digs)
            digs = max(self.fmt_digs[var])
            part = f'{value:>{digs+3}.{digs}f}'
            parts.append(part)
        return "/".join(parts)

class MetricLogger(_Logger):
    def __init__(self, delimiter=" | ", printer=print, **kwargs):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.printer = printer

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def add_stat(self, name, container):
        self.add_meter(name=name, meter=container)

    def get(self, name : str):
        return self.meters[name]
    
    @property
    def statistics(self):
        return self.meters

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

class BaseStatistic(_Statistic):
    def __init__(self, values : Optional[list[Union[float, int]]]=None):
        """
        A basic thread-safeish statistic container.

        Args:
            values (`Optional[list[Union[float, int]]]`): A list of values to initially populate the statistic, optional.
        """
        self.values : list[float] = []
        self.lock = RLock()
        self.min : Optional[float] = None
        self.max : Optional[float] = None
        self.mean : Optional[float] = None
        self.sum : Optional[float] = None
        self.digs : deque[int] = deque(maxlen=30)
        self._len : int = 0
        if values is not None:
            self.update(values)

    def __len__(self):
        return self._len
    
    def __getitem__(self, i):
        return self.values[i]
    
    def __iter__(self):
        """
        Iter on the base statistic is not thread safe, if you want to ensure this, you must acquire the lock manually first.
        """
        for el in self.values:
            yield el

    @property
    def data(self):
        return self.values

    def update(self, value : Union[float, int, list[Union[int, float]], np.ndarray, torch.Tensor], **kwargs):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if sum(s > 1 for s in value.shape) <= 1:
                value = value.flatten()
            value = value.tolist()
            if not isinstance(value, (float, int, list)):
                raise RuntimeError(f'Unable to update statistic ({self}) with heterogenous data or invalid type.')

        with self.lock:
            if isinstance(value, (float, int)):
                tmin = tmax = tsum = value
                n = 1
                self.values.append(float(value))
            else:
                tmin, tmax, tsum = [fn(value) for fn in [min, max, sum]]
                n = len(value)
                self.values.extend([float(v) for v in value])
            if self.min is None or tmin < self.min:
                self.min = float(tmin)
            if self.max is None or tmax > self.max:
                self.max = float(tmax)
            if self.sum is None:
                self.sum = float(tsum)
            else:
                self.sum += tsum 
            self._len += n
            self.mean = self.sum / len(self)

    def __str__(self):
        with self.lock:
            if not self:
                return "NaN |0|"
            m  = self.mean
            mn = self.min
            mx = self.max
        digs = float_signif_decimal(min(filter(lambda x : x is not None and x != 0, map(abs, [m, mn, mx])), default=1))
        self.digs.append(digs)
        digs = max(self.digs)
        return f'{m:>{digs+2}.{digs}f} [{mn:>{digs+3}.{digs}f}; {mx:>{digs+3}.{digs}f}]'
    
    def __repr__(self):
        return f'BaseStatistic({str(self)})|{len(self)}|'

def compute_aligned_steps(
        target_length: int,
        origin_length: int,
        total_epochs: int,
        current_epoch: int
    ) -> list[int]:
    if not (0 <= current_epoch < total_epochs):
        raise ValueError(f"current_epoch must be in [0, {total_epochs}), got {current_epoch!r}")

    start = current_epoch * target_length
    end   = (current_epoch + 1) * target_length - 1

    return [int(round(step)) for step in np.linspace(start, end, num=origin_length)]

class MultiLogger:
    def __init__(
            self, 
            train_loader : torch.utils.data.DataLoader,
            val_loader : torch.utils.data.DataLoader,
            epochs : int,
            output : str,
            name : str="log",
            statistics : list[str]=["loss", "lr", "acc1", "acc5", "item/s", "mem", "step", "time", "eta", "epoch", "type"],
            private_statistics : list[str]=["step", "time", "eta", "epoch", "type"], 
            logger_cls : list[Type[_Logger]]=[MetricLogger],
            logger_cls_extra_kwargs : list[dict[str, Any]]=[],
            logger_cls_stat_factory : list[Callable[[], _Statistic]]=[
                lambda : SmoothedValue(window_size=10, fmt_vars=["value"])
            ],
            canonical_statistic : Optional[str]=None,
            save_interval : int=5,
            verbose : bool=False
        ):
        self.total_epochs = epochs
        self.statistics = statistics
        self.private_statistics = private_statistics
        self._original_statistics = sorted(list(set(self.statistics) - set(self.private_statistics)), key=lambda x : self.statistics.index(x))
        self.statistics_storage = defaultdict(list)
        if canonical_statistic is None:
            canonical_statistic = statistics[0]
        self.canonical_statistic = canonical_statistic
        if self.canonical_statistic not in self.statistics:
            raise KeyError(f'Supplied canonical statistic "{self.canonical_statistic}" should be one of the supplied statistics: {", ".join([str(stat) for stat in self.statistics])}')
        self.heterogeneous_storage = defaultdict(list)
        self.output_path = os.path.join(output, increment_name_dir(name, output) + ".json")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.save_interval = save_interval
        
        self.logger_cls = logger_cls
        self.logger_cls_extra_kwargs = logger_cls_extra_kwargs
        self.logger_cls_stat_factory = logger_cls_stat_factory
        self.verbose = verbose

        # Get aligned steps (mainly for use with tensorboard)
        self.train_steps, self.val_steps = self.compute_steps(self.total_epochs, len(train_loader), len(val_loader))
        self.total_steps = sum(map(len, self.train_steps)) + sum(map(len, self.val_steps))

        # Initialize dynamic attributes
        self._last_save = time.time()
        self._step = 0
        self._start_time = None
        self.eta = None
        self._current_loggers = None
        self._epoch = None
        self._type = None
        self._batch_size = None
        self._idx = None
        self._n_classes = None

    def compute_steps(self, epochs : int, train_steps : int, val_steps : int):
        return \
            [list(range(e*train_steps, (e+1)*train_steps)) for e in range(epochs)], \
            [compute_aligned_steps(train_steps, val_steps, epochs, e) for e in range(epochs)]

    @property
    def steps(self):
        if self._epoch is None:
            raise RuntimeError(f'Attempted to retrieve steps before starting an epoch (i.e. training).')
        clean_type = self._type.lower().strip()
        if clean_type.startswith("train"):
            return self.train_steps[self._epoch] 
        elif clean_type.startswith("val") or clean_type.startswith("eval"):
            return self.val_steps[self._epoch]

    def store(self, name : str, value : Any):
        self.heterogeneous_storage[name].append((value, self._epoch, self._type))

    def update(
        self,
        epoch : int,
        type : str
    ):  
        if self._start_time is None:
            self._start_time = time.time()
            self.eta = ETA(self.total_steps, 0.999)
        self._epoch = epoch
        self._type = type
        self._current_loggers : list[_Logger] = []
        for cls, kwargs, stat_factory in zip(
            self.logger_cls, 
            chain(self.logger_cls_extra_kwargs, repeat(dict())),
            chain(self.logger_cls_stat_factory, repeat(BaseStatistic))
        ):
            this_logger = cls(steps=self.steps, tag=type, **kwargs)
            for stat in self.statistics:
                if stat in self.private_statistics:
                    continue
                this_logger.add_stat(stat, stat_factory())
            self._current_loggers.append(this_logger)

    def step(self):
        self.log_statistic(step=self._step)
        self._step += 1
        self.eta.step()
        for logger in self.loggers:
            logger.step()
        self.log_statistic(time=time.time())
        self.log_statistic(eta=self.eta.eta)
        self.log_statistic(epoch=self._epoch)
        self.log_statistic(type=self._type)
        if (time.time() - self._last_save) >= self.save_interval:
            self.save()

    @property
    def loggers(self) -> list[_Logger]:
        if self._current_loggers is None:
            raise RuntimeError("Attempted to log statistics before initializing loggers.")
        return self._current_loggers
    
    def get_logger(self, cls : Type[L]) -> L:
        for logger in self.loggers:
            if isinstance(logger, cls):
                return logger
        raise KeyError(f'No logger of type {cls} found.')

    def log_statistic(self, **kwargs):
        for stat, value in kwargs.items():
            if stat not in self.private_statistics:
                if stat not in self.statistics:
                    self.statistics.append(stat)
                    for logger, stat_factory in zip(
                        self.loggers, 
                        chain(self.logger_cls_stat_factory, repeat(BaseStatistic))
                    ):
                        logger.add_stat(stat, stat_factory())
                for logger in self.loggers:
                    logger.update(stat, value)
            if isinstance(value, (torch.Tensor, np.ndarray)):
                value = value.tolist()
            if isinstance(value, (tuple, list)):
                self.statistics_storage[stat].extend(value)
            else:
                self.statistics_storage[stat].append(value)

    @property
    def data(self):
        return {
            "statistics" : dict(self.statistics_storage),
            "extra" : dict(self.heterogeneous_storage)
        }

    def save(self, fp: Optional[Union[str, TextIO]] = None, encoding: str = "utf-8", **kwargs):
        if fp is None:
            fp = self.output_path
        if isinstance(fp, TextIO):
            json.dump(self.data, fp, **kwargs)
            return

        temp_file_name = None
        try:
            with NamedTemporaryFile("w", encoding=encoding, suffix=".json", delete=False) as tmpfile:
                json.dump(self.data, tmpfile, **kwargs)
                tmpfile.flush()
                os.fsync(tmpfile.fileno())
                temp_file_name = tmpfile.name
            
            shutil.move(temp_file_name, fp)
            self._last_save = time.time() # Assuming self._last_save is defined
        except Exception:
            if temp_file_name and os.path.exists(temp_file_name):
                try:
                    os.remove(temp_file_name)
                except OSError:
                    pass # Suppress error during cleanup
            raise

    def log_batch(self, batch):
        pass

    def log_accuracy(
            self, 
            target : torch.Tensor, 
            prediction : Union[list[torch.Tensor], torch.Tensor]
        ):
        if isinstance(prediction, list):
            prediction = prediction[0]
            target = target[:, 0]
        if self._n_classes is None:
            self._n_classes = prediction.shape[1]
        self.log_labels_predictions(target.tolist(), prediction.argmax(1).tolist())
        acc1, acc5 = accuracy(prediction, target, topk=(1, 5))
        self.log_statistic(acc1=acc1, acc5=acc5)

    def log_labels_predictions(
            self,
            labels : list[int], 
            predictions : list[int]
        ):
        self.store("labels", labels)
        self.store("predictions", predictions)

    def log_loss(
            self,
            loss : Union[torch.Tensor, list[torch.Tensor]]
        ):
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            loss : float = loss.item()
            self.log_statistic(loss=loss)
        else:
            self.log_statistic(loss=sum(loss).item())
            for i, term in enumerate(loss):
                if term.numel() != 1:
                    raise RuntimeError(f'Expected scalar loss term but found {loss.shape}.')
                term = term.item()
                self.log_statistic(**{f'loss/lvl{i}' : term})

    def log_optim(
            self,
            optimizer : Optional[torch.optim.Optimizer]
        ):
        if optimizer is None:
            self.log_statistic(lr=float('nan'))
        else:
            grps = optimizer.param_groups
            self.log_statistic(lr=grps[0]["lr"])
            if len(grps) > 1:
                for grp in grps:
                    self.log_statistic(**{f'lr/{grp["name"]}' : grp["lr"]})

    def log_speed(
            self,
            start_time : int
        ):
        self.log_statistic(**{"item/s" : self._batch_size / (time.time() - start_time)})

    def log_memory_use(self):
        MB = 1024.0 ** 2
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / MB
        else:
            mem = None
        self.log_statistic(mem=mem)

    def default_consume(
            self,
            index : int,
            batch : torch.Tensor,
            target : torch.Tensor,
            prediction : Union[list[torch.Tensor], torch.Tensor],
            loss : Any,
            optimizer : torch.optim.Optimizer,
            start_time : int
        ):    
        # These are set first, so they may be used while logging the other statistics
        self._idx = index
        self._batch_size = len(batch)
        
        self.log_batch(batch)
        self.log_optim(optimizer)
        self.log_accuracy(target, prediction)
        self.log_loss(loss)
        self.log_speed(start_time)
        self.log_memory_use()

        self._idx = self._batch_size = None

    def consume(self, **kwargs):
        default_consume_kwargs = ["index", "batch", "target", "prediction", "loss", "optimizer", "start_time"]
        dca = {key : kwargs.pop(key) for key in  default_consume_kwargs}
        if len(dca) == len(default_consume_kwargs):
            self.default_consume(**dca)
        self.log_statistic(**kwargs)
        self.step()

    def status(self):
        stats = " | ".join([f'{name}: {str(self.loggers[0].statistics[name])}' for name in self._original_statistics])
        # cuda_memory_use = cuda_memory_stats()
        # used = total = free = 0
        # for dev_idx, mem_stats in cuda_memory_use.items():
        #     used  += mem_stats["used_mb"]
        #     free  += mem_stats["free_mb"]
        #     total += mem_stats["total_mb"]
        # pfree = used/free
        # ptotal = used/total
        epoch = self._epoch
        if epoch is None:
            epoch = "?"
        else:
            epoch += 1
        return f'E{epoch}/{self.total_epochs} ({self._step/self.total_steps:.1%} {self.eta}) | {stats}'# '(mem: {pfree:.1%} of free, {ptotal:.1%} of total)'

    def summary(self, stats : list[str]=["acc1", "acc5", "loss"]):
        parts = []
        for stat in stats:
            values = []
            for v, e, t in zip(reversed(self.statistics_storage[stat]), reversed(self.statistics_storage["epoch"]), reversed(self.statistics_storage["type"])):
                if e != self._epoch or t != self._type:
                    break
                values.append(v)
            value = type(v)(np.median(np.array(values)))
            part = f'{stat}={value:>5.{float_signif_decimal(value)}f}'
            parts.append(part)
        return " | ".join(parts)
    
    def confusion_matrix(self):
        counts = {"labels" : [], "predictions" : []}
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
            print(f"WARNING: No labels or predictions found for {self._epoch}!")
        cm = raw_confusion_matrix(
            **counts,
            n_classes = self._n_classes
        )
        if not bool(np.any(np.isfinite(cm))):
            print(f'Confusion matrix has no valid values, produced from counts: {counts}')
        return plot_heatmap(cm, "magma")[0]

    def add_figure(self, name : str, figure : Figure):
        for logger in self.loggers:
            logger.add_figure(name=name, figure=figure, epoch=self._epoch)

    def figures(self, model : Optional[nn.Module]):
        cm_fig = self.confusion_matrix()
        self.add_figure("Confusion matrix", cm_fig)
        plt.close(cm_fig)

        if model is not None:
            cdm_fig, _ = plot_model_class_distance(model)
            self.add_figure("Class distance", cdm_fig)
            plt.close(cdm_fig)