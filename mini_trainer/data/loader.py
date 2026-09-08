from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, default_collate
from torch.utils.data.distributed import DistributedSampler

from mini_trainer import get_logger
from mini_trainer.utils import is_dist_avail_and_initialized

from ._workers import _default_worker_count
from .io import (
    CACHE_MODE,
    LazyDataset,
    _FetchedBatch,
    guess_cache_mode,
    make_read_and_resize_fn,
)


def _normalize_resize_size(resize_size, *, error_suffix=""):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    if not (isinstance(resize_size, (tuple, list)) and len(resize_size) == 2 and all(isinstance(x, int) for x in resize_size)):
        raise TypeError(
            f"Invalid resize size passed, found {resize_size}, but expected an integer or a tuple of two integers{error_suffix}"
        )
    return resize_size


def label_to_tensor(label: int | list[int] | tuple[int, ...] | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert label input to a LongTensor."""
    if isinstance(label, (int, tuple, list)):
        return torch.tensor(label, dtype=torch.long)
    if isinstance(label, np.ndarray):
        return torch.from_numpy(label).clone().long()
    return torch.as_tensor(label).long()


class PathLabelProcessor:
    """Processor to load images and preprocess labels, suitable for pickling in spawn context."""

    def __init__(self, reader: Callable[[str], torch.Tensor], hook: Callable[[torch.Tensor], torch.Tensor] | None, multilabel: bool):
        self.reader = reader
        self.hook = hook
        self.multilabel = multilabel

    def __call__(self, path_label: tuple[str, int | list[int] | np.ndarray | torch.Tensor]):
        path, label = path_label
        label = label_to_tensor(label)
        if not self.multilabel and label.numel() > 1:
            label = label[0]
        label = label.detach().cpu().clone().long()
        image = self.reader(path)
        if self.hook is not None:
            image = self.hook(image)
        return image, label


class HookedReader:
    """Reader wrapper to apply a hook, suitable for pickling in spawn context."""

    def __init__(self, reader: Callable[[str], torch.Tensor], hook: Callable[[torch.Tensor], torch.Tensor]):
        self.reader = reader
        self.hook = hook

    def __call__(self, x: str) -> torch.Tensor:
        return self.hook(self.reader(x))


def _collate_batch(samples):
    if isinstance(samples, _FetchedBatch):
        # Match default_collate's tuple-to-list convention for (image, label).
        return list(samples.data) if isinstance(samples.data, tuple) else samples.data
    return default_collate(samples)


def get_dataloader(  # noqa: D103
    dataset: torch.utils.data.Dataset,
    mode: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
    *,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
):
    assert isinstance(mode, str)
    if mode.strip().lower() == "train":
        shuffle = drop_last = True
    else:
        shuffle = drop_last = False

    if is_dist_avail_and_initialized():
        base_sampler = DistributedSampler(dataset, shuffle=shuffle)
        # Use spawn in DDP to avoid CUDA context inheritance crashes
        mp_context = "spawn" if num_workers > 0 else None
    else:
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)  # type: ignore
        mp_context = None

    if num_workers > 0 and multiprocessing_context is not None:
        mp_context = multiprocessing_context

    sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=drop_last)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=_collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_context,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def get_dataset_dataloader(  # noqa: D103
    *metadata: dict,
    resize_size: int | tuple[int, int],
    modes: tuple[str, ...] = ("train", "val"),
    batch_size: int = 16,
    num_workers: int | None = None,
    subsample: int | None = None,
    resample: bool | str = False,  # Enable with: "ilog1p",
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    cache: CACHE_MODE | str | int | None = None,
    multilabel: bool = False,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
    hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    resize_size = _normalize_resize_size(resize_size, error_suffix=".")
    if isinstance(device, str):
        device = torch.device(device)

    if len(metadata) != len(modes):
        raise ValueError(f"Number of supplied datasets: {len(metadata)} and modes: {len(modes)} do not match!")

    log = get_logger()
    log.info(f"Building datasets with image size {resize_size}")
    if subsample is not None and subsample > 1:
        metadata = tuple([{k: v[::subsample] for k, v in md.items()} for md in metadata])

    dataset_shape = list((sum(len(md["path"]) for md in metadata), *resize_size, 3))
    cache = CACHE_MODE(cache)
    if cache is CACHE_MODE.GUESS:
        cache = guess_cache_mode(dataset_shape, dtype)

    reader = make_read_and_resize_fn(resize_size, torch.device("cpu"), torch.uint8)

    proc_path_label = PathLabelProcessor(reader, hook, multilabel)

    datasets = []
    for mode, data in zip(modes, metadata):
        if mode.strip().lower() == "train" and resample:
            raise NotImplementedError("Resampling is currently not supported.")
        dset = LazyDataset(func=proc_path_label, items=(data["path"], data["class"]), cache=cache)
        datasets.append(dset)

    if cache is CACHE_MODE.CUDA:
        # When the entire dataset is preloaded there is no need to use multiprocessing for dataloading
        num_workers = 0
    elif num_workers is None:
        num_workers = _default_worker_count(16)

    # A gather from a pinned cache allocates an unpinned result. Pin the actual
    # CPU batch before asynchronous H2D, including for the CPU cache path.
    pin_memory = device.type == "cuda" and cache is not CACHE_MODE.CUDA
    loaders = [
        get_dataloader(
            dataset,
            mode,
            batch_size,
            num_workers,
            pin_memory,
            device,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )
        for mode, dataset in zip(modes, datasets)
    ]

    return datasets, loaders


def get_inference_dataloader(  # noqa: D103
    images: list[str],
    resize_size: int | tuple[int, int],
    batch_size: int = 16,
    num_workers: int | None = None,
    subsample: int | None = None,
    device: torch.device | str = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
    prefetch_factor: int | None = None,
    multiprocessing_context: str | None = None,
    **kwargs,
):
    resize_size = _normalize_resize_size(resize_size)
    if isinstance(device, str):
        device = torch.device(device)

    if subsample is not None and subsample > 1:
        images = images[::subsample]

    reader = make_read_and_resize_fn(resize_size, torch.device("cpu"), torch.uint8)
    if hook is not None:
        reader = HookedReader(reader, hook)

    dataset = LazyDataset(func=reader, items=(images,), cache=CACHE_MODE.NONE)

    if num_workers is None:
        num_workers = _default_worker_count(32)

    loader = get_dataloader(
        dataset,
        "test",
        batch_size,
        num_workers,
        device.type == "cuda",
        device,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
    )

    return dataset, loader
