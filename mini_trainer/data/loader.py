import logging
import os
from collections import Counter
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from mini_trainer.data.io import (
    CACHE_MODE,
    LazyDataset,
    Reindexed,
    guess_cache_mode,
    make_read_and_resize_fn,
)
from mini_trainer.utils import is_dist_avail_and_initialized


def label_to_tensor(label: int | list[int] | tuple[int, ...] | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert label input to a LongTensor."""
    if isinstance(label, (int, tuple, list)):
        return torch.tensor(label, dtype=torch.long)
    if isinstance(label, np.ndarray):
        return torch.from_numpy(label).clone().long()
    return label.long()


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
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
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


def get_dataloader(  # noqa: D103
    dataset: torch.utils.data.Dataset, mode: str, batch_size: int, num_workers: int, pin_memory: bool, device: torch.device
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
        base_sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset) # type: ignore
        mp_context = None

    sampler = BatchSampler(base_sampler, batch_size=batch_size, drop_last=drop_last)

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=str(device) if pin_memory else "",
        persistent_workers=num_workers > 0,
        multiprocessing_context=mp_context,
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
    hook: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    if not (isinstance(resize_size, (tuple, list)) and len(resize_size) == 2 and all(map(lambda x: isinstance(x, int), resize_size))):
        raise TypeError(f"Invalid resize size passed, found {resize_size}, but expected an integer or a tuple of two integers.")
    if isinstance(device, str):
        device = torch.device(device)

    if len(metadata) != len(modes):
        raise ValueError(f"Number of supplied datasets: {len(metadata)} and modes: {len(modes)} do not match!")

    log = logging.getLogger("mini_trainer")
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
        items = list(zip(data["path"], data["class"]))
        # TODO: Abstract and modularize
        if mode.strip().lower() == "train" and resample:
            if cache != CACHE_MODE.NONE:
                raise NotImplementedError("Resampling with caching is not supported.")
            labs = data["class"]
            if isinstance(labs[0], (list, tuple)):
                labs = [lab[0] for lab in labs]
            cc = Counter(labs)
            resample_kwargs = {}
            if isinstance(resample, str):
                resample_kwargs["transform"] = resample
            items = Reindexed(items, [cc.get(k, 0) for k in labs], inflation=2, **resample_kwargs)
        dset = LazyDataset(func=proc_path_label, items=items, cache=cache)
        datasets.append(dset)

    if num_workers is None:
        num_workers = (os.cpu_count() or 0) - 4
        num_workers -= num_workers % 2
        num_workers = min(16, max(0, num_workers))
    if cache is CACHE_MODE.CUDA:
        # When the entire dataset is preloaded there is no need to use multiprocessing for dataloading
        num_workers = 0

    pin_memory = cache not in [CACHE_MODE.CUDA, CACHE_MODE.CPU]
    loaders = [get_dataloader(dataset, mode, batch_size, num_workers, pin_memory, device) for mode, dataset in zip(modes, datasets)]

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
    **kwargs,
):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    if not (isinstance(resize_size, (tuple, list)) and len(resize_size) == 2 and all(map(lambda x: isinstance(x, int), resize_size))):
        raise TypeError(f"Invalid resize size passed, found {resize_size}, but expected an integer or a tuple of two integers")
    if isinstance(device, str):
        device = torch.device(device)

    if subsample is not None and subsample > 1:
        images = images[::subsample]

    reader = make_read_and_resize_fn(resize_size, torch.device("cpu"), torch.uint8)
    if hook is not None:
        reader = HookedReader(reader, hook)

    dataset = LazyDataset(func=reader, items=images, cache=CACHE_MODE.NONE)

    if num_workers is None:
        num_workers = (os.cpu_count() or 0) - 4
        num_workers -= num_workers % 2
        num_workers = min(32, max(0, num_workers))

    loader = get_dataloader(dataset, "test", batch_size, num_workers, False, device)

    return dataset, loader
