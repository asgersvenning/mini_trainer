import hashlib
import math
import operator
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from queue import Queue
from threading import Thread
from typing import Any, TypeVar, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Sampler
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from mini_trainer import TQDM
from mini_trainer.utils import make_convert_dtype, memory_proportion, multithread_vectorize

T = TypeVar("T")
V = TypeVar("V")


class CACHE_MODE(int, Enum):  # noqa: D101
    NONE = 0  # noqa: E221
    GUESS = 1  # noqa: E221
    DISK = 2  # noqa: E221
    CPU = 3  # noqa: E221
    CUDA = 4  # noqa: E221

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        if isinstance(value, str):
            name = value.strip().upper()
            try:
                return cls[name]
            except KeyError:
                raise ValueError(f'Only cache modes "NONE", "GUESS", "DISK", "CPU" or "CUDA" are defined, not {value!r}.')
        # let IntEnum's default handling raise for bad ints:
        return super()._missing_(value)


DEFAULT_THRESHOLDS = {CACHE_MODE.NONE: -1, CACHE_MODE.GUESS: None, CACHE_MODE.DISK: 0.5, CACHE_MODE.CPU: 0.5, CACHE_MODE.CUDA: None}


def guess_cache_mode(shape: list[int], dtype: torch.dtype, thresholds: dict[CACHE_MODE, float | int] | None = None):
    """Heuristic to guess/select a caching strategy."""
    if thresholds is None:
        thresholds = dict()
    for mode, default in DEFAULT_THRESHOLDS.items():
        if mode not in thresholds:
            thresholds[mode] = default
    accepted: list[CACHE_MODE] = []
    for mode, threshold in thresholds.items():
        if threshold is None:
            continue
        if threshold < 0 or memory_proportion(tuple(shape), mode.name, dtype) < threshold:
            accepted.append(mode)
    if len(accepted) == 0:
        raise RuntimeError(f"Unable to determine a valid caching location using thresholds:\n{thresholds}")
    return sorted(accepted)[-1]


@multithread_vectorize(desc="Checking images...", disable=True)
def is_image(path: str):
    """Check if path(s) is/are image(s)."""
    if not os.path.exists(path):
        return False

    try:
        with open(path, "r+b", buffering=16) as f:
            header = f.read(16)  # read enough bytes for JPEG and PNG signatures
    except Exception:
        return False

    # JPEG files start with: 0xFF, 0xD8
    if header.startswith(b"\xff\xd8"):
        return True

    # PNG files start with: 0x89, 'PNG', CR, LF, 0x1A, LF
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True

    return False


def _pil_to_torch_interp(interp: int) -> InterpolationMode:
    m = {
        Image.Resampling.NEAREST: InterpolationMode.NEAREST,
        Image.Resampling.BILINEAR: InterpolationMode.BILINEAR,
        Image.Resampling.BICUBIC: InterpolationMode.BICUBIC,
        Image.Resampling.LANCZOS: InterpolationMode.LANCZOS,
        Image.Resampling.BOX: InterpolationMode.BOX,
        Image.Resampling.HAMMING: InterpolationMode.NEAREST,
    }
    return m.get(interp, InterpolationMode.BILINEAR)  # type: ignore


class ReadAndResize:
    """Callable class to read and resize images from paths."""

    def __init__(
        self,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        interpolation=Image.Resampling.NEAREST,
        **kwargs,
    ):
        self.converter = make_convert_dtype(dtype)
        self.interp = _pil_to_torch_interp(interpolation)
        self.antialias = kwargs.get("antialias", True)
        self.w, self.h = size
        self.device = device
        self.dtype = dtype

    def __call__(self, path: str) -> torch.Tensor:
        try:
            img = decode_image(path, mode=ImageReadMode.RGB, apply_exif_orientation=False)  # uint8 [C,H,W]
        except Exception as e:
            e.add_note(f"Image path: {path}")
            raise
        img = TF.resize(img, size=[self.h, self.w], interpolation=self.interp, antialias=self.antialias)
        if img.dtype != self.dtype:
            img = self.converter(img)
        return img.to(self.device)


def make_read_and_resize_fn(
    size: tuple[int, int], device: torch.device, dtype: torch.dtype | str, interpolation=Image.Resampling.NEAREST, **kwargs
):
    """Factory to create function to read and resize image from path."""
    if isinstance(dtype, str):
        _dtype = getattr(torch, dtype, None)
        if not isinstance(_dtype, torch.dtype):
            raise ValueError(f'Unknown dtype "{dtype}"')
        dtype = _dtype
    return ReadAndResize(size, device, dtype, interpolation, **kwargs)


def _normalize_to_tuple(data):
    return data if isinstance(data, (tuple, list)) else (data,)


# From `flatbug`: https://github.com/darsa-group/flat-bug/blob/9093de0f89756b7f59e63f3bd7161f5574eb90ac/src/flat_bug/datasets.py#L42
def reweight(weights: list[float], target_sum: float | int):
    """Reweights the provided list of weights so that their sum equals the target sum.

    Args:
        weights: List of weights to reweight.
        target_sum: Desired sum of the weights.

    Returns:
        Reweighted weights.
    """
    sum_weights = sum(weights)
    return [int(max(round(w * target_sum / sum_weights), 1)) for w in weights]


def generate_indices(weights: list[float], target_size: int | None = None):
    """Deterministically generates a list of indices based on the provided weights to oversample the items.

    Args:
        weights: List of weights for each item.
            The weights should correspond to the desired oversampling rate for each item.
        target_size: Desired size of the output list.
            If None, the size of the output is approximately the sum of the weights.

    Returns:
        tuple of list of indices to oversample the items and list of final weights.
    """
    if target_size is None:
        target_size = round(sum(weights))
        if target_size < len(weights):
            raise ValueError(f"Target size not specified, and could not be derived from weights with sum: {target_size}")
    weights = list(map(round, weights))
    assert all([w >= 0 for w in weights]), "Weights have to be >= 0"
    indices = []

    if target_size is not None:
        for _ in range(10):
            if (abs(sum(weights) - target_size) / target_size) < 0.01:
                break
            weights = list(map(float, reweight(weights, target_size)))

    out_weights = list(map(int, weights))
    for i, w in enumerate(out_weights):
        indices.extend([i] * w)

    return indices, out_weights


def _vectorize[V](func: Callable[[V], V]):
    return lambda x: list(map(func, x))


def uniform_mixture(x: list[float], p: float):
    assert p >= 0 and p <= 1
    if len(x) == 0:
        raise ValueError("Unable to flatten empty distribution.")
    if p == 0:
        return list(x)
    xm = sum(x) / len(x)
    if p == 1:
        return [xm] * len(x)
    return [p * xm + (1 - p) * xi for xi in x]


STANDARD_TRANSFORMS: dict[str, Callable[[list[float]], list[float]] | None] = {
    "identity": None,
    "ilog1p": _vectorize(lambda x: 1 / math.log1p(x)),
    "ilog": _vectorize(lambda x: 1 / math.log(x)),
    "log": _vectorize(math.log),
    "sqrt": _vectorize(math.sqrt),
    "isqrt": _vectorize(lambda x: x**-0.5),
    "pow2": _vectorize(lambda x: x**2),
}


class Reindexed[T]:  # noqa: D101
    def __init__(  # noqa: D107
        self,
        items: list[T],
        weights: list[float | int],
        inflation: float | int = 2,
        flatten: float = 0.1,
        transform: Callable[[list[float]], list[float]] | str | None = "isqrt",
    ) -> None:
        if isinstance(transform, str):
            try:
                transform = STANDARD_TRANSFORMS[transform]  # type: ignore
            except KeyError as e:
                raise ValueError(f"Unknown transform {transform!r}. Expected one of {tuple(STANDARD_TRANSFORMS)}.") from e
        assert not isinstance(transform, str)

        processed_weights = uniform_mixture(list(map(float, weights)), p=flatten)
        if transform is not None:
            processed_weights = transform(processed_weights)
        mw = sum(processed_weights) / len(processed_weights)
        processed_weights = [w / mw * inflation for w in processed_weights]

        if any(not math.isfinite(w) for w in processed_weights):
            raise ValueError("All transformed weights must be finite.")

        self.items = items
        self._length = len(items)

        if len(processed_weights) != self._length:
            raise ValueError(
                f"Length of supplied items ({self._length}) does not match length of supplied weights ({len(processed_weights)})."
            )

        target_size = round(self._length * float(inflation))
        self._indices, self.weights = generate_indices(processed_weights, target_size)

    def __len__(self) -> int:
        return self._length

    def _get_single(self, x: int) -> T:
        return self.items[self._indices[x]]

    def _gather_from_positions(self, positions) -> T | list[T] | list:
        if isinstance(positions, int):
            return self.items[positions]
        return [self._gather_from_positions(p) for p in positions]

    def __getitem__(self, x):
        if isinstance(x, slice):
            return [self.items[i] for i in self._indices[x]]

        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return self._get_single(operator.index(x.item()))
            mapped = np.asarray(self._indices, dtype=np.int64)[x]
            return self._gather_from_positions(mapped.tolist())

        if torch.is_tensor(x):
            if x.ndim == 0:
                return self._get_single(operator.index(int(x.item())))
            mapped = torch.as_tensor(self._indices, dtype=torch.long)[x]
            return self._gather_from_positions(mapped.tolist())

        try:
            return self._get_single(operator.index(x))
        except TypeError:
            pass

        if isinstance(x, Iterable):
            return [self._get_single(operator.index(i)) for i in x]

        raise TypeError(f"Unsupported index type: {type(x)!r}")

    def __repr__(self) -> str:
        return f"[{', '.join(f'({repr(e)} * {w})' for e, w in zip(self.items, self.weights))}]"

    @property
    def indices(self) -> list[int]:
        return list(self._indices)

    @property
    def reindexed_len(self) -> int:
        return len(self._indices)


class ReindexedSampler(Reindexed[T], Sampler[int]):  # noqa: D101
    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class LazyDataset(torch.utils.data.Dataset):
    """A general lazy dataset which calls func on items to
    obtain the image (and label) when needed.

    Includes caching options:
        * None : No caching.
        * "cpu" : Cache in RAM as CPU tensor.
        * "cuda" : Cache in VRAM as CUDA tensor.
        * "guess" : Select a caching strategy via heuristic.
    """

    def __init__(  # noqa: D107
        self,
        func: Callable[[Any], torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
        items: Sequence,
        cache: str | int | CACHE_MODE | None = None,
    ):
        self.func = func
        self.items = items
        self._cache_mode = CACHE_MODE(cache)
        self._init_cache()

    @staticmethod
    def _hash_item(item):
        if isinstance(item, str):
            return item.encode("utf-8")
        if isinstance(item, (list, tuple)):
            str_item = next((e for e in item if isinstance(e, str)), str(item))
            return str_item.encode("utf-8")
        return str(item).encode("utf-8")

    def _get_cache_hash(self) -> str:
        s256 = hashlib.sha256(b"mini_trainer", usedforsecurity=False)
        for item_hash in sorted(map(self._hash_item, self.items)):
            s256.update(item_hash)
        return s256.hexdigest()

    def _init_cache(self):
        match self._cache_mode:
            case CACHE_MODE.NONE:
                print("On-the-fly data loading enabled (no cache).")
                return
            case CACHE_MODE.DISK:
                raise NotImplementedError("Disk caching is obsolete and has been removed.")
            case CACHE_MODE.CPU:
                self._cache_ram()
            case CACHE_MODE.CUDA:
                self._cache_ram(desc="Writing to CUDA RAM cache...")
                guess_device = torch.device(torch.cuda.current_device())
                warnings.warn(
                    f"CUDA caching is currently in development and may not work properly. Using device: `{guess_device}` for cache."
                )
                self._ram_cache.tensors = tuple([t.to(guess_device) for t in self._ram_cache.tensors])
            case _:
                raise ValueError(f'Invalid cache mode "{self._cache_mode}". Choose from [None, "cpu", "cuda"].')

    def _cache_ram(self, desc: str = "Writing to CPU RAM cache..."):
        if not self.items:
            self._ram_cache = torch.utils.data.TensorDataset()  # Handle empty case
            self._ram_was_single_tensor = False
            return

        first_item_processed = self.func(self.items[0])

        if isinstance(first_item_processed, torch.Tensor):
            self._ram_was_single_tensor = True
            templates = [first_item_processed.new_empty(first_item_processed.shape)]
        elif isinstance(first_item_processed, (tuple, list)):
            self._ram_was_single_tensor = False
            templates = [e.new_empty(e.shape) for e in first_item_processed]
        else:
            raise TypeError(f"The provided function must return a tensor ora tuple/list of tensors, but got {type(first_item_processed)}")

        stacked_tensors = [
            torch.empty(
                (len(self), *template.shape),
                dtype=template.dtype,
                device=template.device,
                pin_memory=(template.device == torch.device("cpu")) and torch.cuda.is_available(),
            )
            for template in templates
        ]

        max_workers = max(0, min(128, (((os.cpu_count() or 0) - 2) // 2) * 2 or 1))
        batch_size = min(256, 4 * max_workers)
        fetch_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fetcher")
        fetched_queue: Queue[tuple[int, torch.Tensor | Sequence[torch.Tensor]]] = Queue(max(32, batch_size * 4))
        insert_buffer: dict[int, torch.Tensor | Sequence[torch.Tensor]] = dict()
        insert_queue: Queue[tuple[int, torch.Tensor | Sequence[torch.Tensor]]] = Queue()

        def _fetch_one(idx_item):
            idx, item = idx_item
            try:
                data = self.func(item)
                fetched_queue.put((idx, data))
            except Exception as e:
                fetched_queue.put((idx, e))

        def _contiguous_write(idx: list[int], data: list[torch.Tensor] | list[list[torch.Tensor]]) -> None:
            """Write data to indexes.

            Args:
                idx: A list of contigous increasing indices for each corresponding torch.Tensor (element) in `data`.
                data: A list of torch.Tensor with the same length as `idx`.
            """
            if len(idx) == 0:
                return
            slc = slice(idx[0], idx[-1] + 1)
            # Insert data into slice along first dimension in dst (in-place)
            if self._ram_was_single_tensor:
                assert not data or isinstance(data[0], torch.Tensor)
                data = cast(list[torch.Tensor], data)
                torch.stack(data, out=stacked_tensors[0][slc])
            else:
                for i, elements in enumerate(zip(*data)):
                    torch.stack(elements, out=stacked_tensors[i][slc])

        def _write():
            end_idx = len(self) - 1
            with TQDM(range(len(self)), desc=desc, leave=False) as pbar:
                batch = ([], [])
                while True:
                    idx, data = insert_queue.get()
                    batch[0].append(idx)
                    batch[1].append(data)
                    if len(batch[0]) >= batch_size:
                        _contiguous_write(*batch)
                        batch = ([], [])
                    pbar.update()
                    if idx == end_idx:
                        break
                _contiguous_write(*batch)

        write_thread = Thread(target=_write, daemon=True)
        write_thread.start()

        try:
            fetch_pool.map(_fetch_one, enumerate(self.items))

            nxt_idx = 0
            for _ in range(len(self)):
                idx, data = fetched_queue.get()
                if isinstance(data, Exception):
                    raise data
                if idx == nxt_idx:
                    insert_queue.put((idx, data))
                    nxt_idx += 1
                    while nxt_idx in insert_buffer:
                        insert_queue.put((nxt_idx, insert_buffer.pop(nxt_idx)))
                        nxt_idx += 1
                else:
                    insert_buffer[idx] = data

            write_thread.join()
        finally:
            fetch_pool.shutdown(wait=False)

        self._ram_cache = torch.utils.data.TensorDataset(*[t for t in stacked_tensors])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        match self._cache_mode:
            case CACHE_MODE.NONE:
                if isinstance(i, int):
                    return self.func(self.items[i])
                if isinstance(i, (torch.Tensor, np.ndarray)):
                    i = i.tolist()
                if isinstance(i, list):
                    elements = [self.func(self.items[j]) for j in i]
                elif isinstance(i, slice):
                    elements = [self.func(item) for item in self.items[i]]
                else:
                    raise NotImplementedError(
                        f"Indexing with {i} is not implemented. Only integer, "
                        "slice, or list/np.ndarray/torch.Tensor of integers indexing is supported."
                    )
                if isinstance(elements[0], torch.Tensor):
                    elements = cast(list[torch.Tensor], elements)
                    return torch.stack(elements)
                return [torch.stack(values) for values in zip(*elements)]
            case CACHE_MODE.DISK:
                raise NotImplementedError("Disk caching is obsolete and has been removed.")
            case CACHE_MODE.CPU | CACHE_MODE.CUDA:
                data = self._ram_cache[i]
                return data[0] if self._ram_was_single_tensor else data
            case _:
                raise RuntimeError(f'Invalid caching mode found {self._cache_mode}, expected one of None, "cpu" or "cuda".')
