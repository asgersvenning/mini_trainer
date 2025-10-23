import hashlib
import os
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from queue import Queue
from tempfile import gettempdir
from threading import Semaphore, Thread
from typing import Any

import numpy as np
import torch
import zarr
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from zarr.storage import LocalStore

from mini_trainer import TQDM
from mini_trainer.utils import (make_convert_dtype, memory_proportion,
                                multithread_vectorize)


class CACHE_MODE(int, Enum):
    NONE  = 0
    GUESS = 1
    DISK  = 2
    CPU   = 3
    CUDA  = 4
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

DEFAULT_THRESHOLDS = {
    CACHE_MODE.NONE : -1,
    CACHE_MODE.GUESS : None,
    CACHE_MODE.DISK : 0.5,
    CACHE_MODE.CPU : 0.5,
    CACHE_MODE.CUDA : None
}

def guess_cache_mode(
        shape : list[int], 
        dtype : torch.dtype,
        thresholds : dict[CACHE_MODE, float | int] | None=None
    ):
    if thresholds is None:
        thresholds = dict()
    for mode, default in DEFAULT_THRESHOLDS.items():
        if mode not in thresholds:
            thresholds[mode] = default
    accepted : list[CACHE_MODE] = []
    for mode, threshold in thresholds.items():
        if threshold is None:
            continue
        if threshold < 0 or memory_proportion(shape, mode.name, dtype) < threshold:
            accepted.append(mode)
    if len(accepted) == 0:
        raise RuntimeError(f'Unable to determine a valid caching location using thresholds:\n{thresholds}')
    return sorted(accepted)[-1]

@multithread_vectorize(desc="Checking images...", disable=True)
def is_image(path : str):
    if not os.path.exists(path):
        return False
    
    try:
        with open(path, "r+b", buffering=16) as f:
            header = f.read(16)  # read enough bytes for JPEG and PNG signatures
    except Exception:
        return False

    # JPEG files start with: 0xFF, 0xD8
    if header.startswith(b'\xff\xd8'):
        return True

    # PNG files start with: 0x89, 'PNG', CR, LF, 0x1A, LF
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return True

    return False

def make_read_and_resize_fn(
    size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype | str,
    interpolation=Image.Resampling.NEAREST,
    **kwargs
):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f'Unknown dtype "{dtype}"')

    converter = make_convert_dtype(dtype)

    def read_and_resize(path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize(size, interpolation)
        tensor = pil_to_tensor(img)  # returns torch.uint8 [C,H,W]
        if tensor.dtype != dtype:
            tensor = converter(tensor)
        return tensor.to(device)

    return read_and_resize

def _normalize_to_tuple(data):
    return data if isinstance(data, (tuple, list)) else (data,)

# From `flatbug`: https://github.com/darsa-group/flat-bug/blob/9093de0f89756b7f59e63f3bd7161f5574eb90ac/src/flat_bug/datasets.py#L42
def reweight(
        weights : list[float], 
        target_sum : float | int
    ) -> list[float]:
    """
    Reweights the provided list of weights so that their sum equals the target sum.

    Args:
        weights: List of weights to reweight.
        target_sum: Desired sum of the weights.

    Returns:
        Reweighted weights.
    """
    sum_weights = sum(weights)
    return [max(round(w * target_sum / sum_weights), 1) for w in weights]

def generate_indices(
        weights : list[float | int], 
        target_size : int | None=None
    ) -> list[int]:
    """
    Deterministically generates a list of indices based on the provided weights to oversample the items.

    Args:
        weights: List of weights for each item, the weights should correspond to the desired oversampling rate for each item.
        target_size: Desired size of the output list. If None, the size of the output is approximately the sum of the weights.

    Returns:
        tuple of list of indices to oversample the items and list of final weights.
    """
    weights = [max(round(w), 1) for w in weights]
    indices = []

    if target_size is not None:
        for _ in range(10):
            if abs(sum(weights) - target_size)/target_size < 0.01:
                break
            weights = reweight(weights, target_size)

    for i, w in enumerate(weights):
        indices.extend([i] * w)

    return indices, weights

class Reindexed:
    def __init__(
            self, 
            items : list, 
            weights : list[float | int], 
            inflation : float | int=2
        ):
        self.items = items
        self._length = len(self.items)
        if len(weights) != len(self):
            raise ValueError(f'Length of the supplied items ({len(items)}) to reindex does not match the length of the supplied weights ({len(weights)}).')
        self._indices, self.weights = generate_indices(weights, round(len(self)*inflation))
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, x):
        indices = self._indices[x]
        if isinstance(indices, int):
            return self.items[indices]
        return [self.items[idx] for idx in indices]

    def __repr__(self):
        return f'[{", ".join(f"({repr(e)} * {w})" for e, w in zip(self.items, self.weights))}]'

class LazyDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            func : Callable[[Any], torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]], 
            items : list,
            cache : str | int | CACHE_MODE | None=None
        ):
        self.func = func
        self.items = items
        self._zarr_root = None
        self.cache_dir = None
        self._cache_mode = CACHE_MODE(cache) # one of None (no caching), disk (precompute .npy files), cpu (preload cpu tensor)
        self._init_cache()

    @staticmethod
    def _hash_item(item):
        if isinstance(item, str): 
            return item.encode('utf-8')
        if isinstance(item, (list, tuple)):
            str_item = next((e for e in item if isinstance(e, str)), str(item))
            return str_item.encode('utf-8')
        return str(item).encode('utf-8')

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
                cache_dir = os.path.join(gettempdir(), ".mini_trainer")
                self.cache_path = os.path.join(cache_dir, f"{self._get_cache_hash()}.zarr")
                self._cache_disk_zarr()
            case CACHE_MODE.CPU:
                self._cache_ram()
            case CACHE_MODE.CUDA:
                self._cache_ram()
                guess_device = torch.device(torch.cuda.current_device())
                warnings.warn(f'CUDA caching is currently in development and may not work properly. Using device: `{guess_device}` for cache.')
                self._ram_cache.tensors = [t.to(guess_device) for t in self._ram_cache.tensors]
            case _:
                raise ValueError(f"Invalid cache mode '{self._cache_mode}'. Choose from [None, 'disk', 'cpu'].")

    def _cache_disk_zarr(self):
        print(f"Using Zarr disk cache at: {self.cache_path}")
        if os.path.exists(self.cache_path):
            print("Found existing Zarr cache.")
            # We need to know if the original data was a single array or a tuple
            store = LocalStore(self.cache_path, read_only=True)
            root = zarr.open(store, mode='r')
            self._disk_cache_is_single_array = 'data_0' in root and 'data_1' not in root
            store.close()
            return

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        print("Building Zarr disk cache...")

        # Process the first item to determine shapes and dtypes for Zarr arrays
        if not self.items: 
            return
        
        first_item_processed = _normalize_to_tuple(self.func(self.items[0]))
        self._disk_cache_is_single_array = len(first_item_processed) == 1
        
        store = LocalStore(self.cache_path, read_only=False)
        root = zarr.open(store, mode="w", zarr_format=3)
        
        # Create a Zarr array for each component of the data
        zarr_arrays : list[zarr.Array] = []
        shard_size = 1024
        for i, component in enumerate(first_item_processed):
            if isinstance(component, torch.Tensor):
                component = component.detach().cpu().numpy()
            if not isinstance(component, np.ndarray):
                raise TypeError(f"For 'disk' cache, `func` must return NumPy arrays. Got {type(component)}.")
            
            arr = root.create_array(
                name=f'data_{i}',
                shape=(len(self), *component.shape),
                chunks=(1, *component.shape),  # Crucial for fast random access by item,
                shards=(shard_size, *component.shape),
                dtype=component.dtype,
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
            )
            if not isinstance(arr, zarr.Array):
                raise RuntimeError(f"Created `zarr.Array` is {arr}?")
            zarr_arrays.append(arr)

        # We need two pools: one for CPU-bound producers, one for I/O-bound writers
        producer_workers = min(64, (os.cpu_count() - 1) or 1) # Can be high
        writer_workers = min(8, (os.cpu_count() // 2) or 1) # Lower, I/O-limited
        
        results_queue = Queue(maxsize=shard_size*2)
        producer_pool = ThreadPoolExecutor(max_workers=producer_workers, thread_name_prefix="producer")
        writer_pool = ThreadPoolExecutor(max_workers=writer_workers, thread_name_prefix="writer")
        writer_semaphore = Semaphore(writer_workers + 4)

        def producer_task(idx_item):
            idx, item = idx_item
            data = _normalize_to_tuple(self.func(item))
            data_np = tuple(d.detach().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data)
            results_queue.put((idx, data_np))

        def writer_task(indices, components):
            try:
                shard_start, shard_end = indices[0], indices[-1] + 1
                for i, chunk in enumerate(components):
                    zarr_arrays[i][shard_start : shard_end, ...] = np.stack(chunk)
                    # np.stack(chunk, out=)
            finally:
                writer_semaphore.release()

        def assembler_task():
            # This thread is now extremely fast. It only manages buffers and delegates.
            buffer = {}
            items_assembled = 0
            total_items = len(self)
            
            with TQDM(total=total_items, desc="Writing to Zarr...") as pbar:
                while items_assembled < total_items:
                    idx, data = results_queue.get()
                    buffer[idx] = data
                    pbar.set_postfix_str(f'QS: {results_queue.qsize()}, BS: {len(buffer)}')

                    shard_idx = idx // shard_size
                    shard_start = shard_idx * shard_size
                    shard_end = min(shard_start + shard_size, total_items)
                    
                    # Check if the buffer is ready 
                    if (len(buffer) >= shard_size or (total_items - 1) in buffer) and all(i in buffer for i in range(shard_start, shard_end)):
                        indices = range(shard_start, shard_end)
                        shard_buf = [buffer.pop(i) for i in indices]
                        components = [[e[c] for e in shard_buf] for c in range(len(zarr_arrays))]
                        del shard_buf
                        
                        # Delegate the slow write operation to the writer pool
                        writer_semaphore.acquire()
                        writer_pool.submit(writer_task, indices, components)
                        
                        num_in_shard = len(indices)
                        pbar.update(num_in_shard)
                        items_assembled += num_in_shard

        assembler = Thread(target=assembler_task, daemon=True)
        assembler.start()

        with producer_pool as pool:
            pool.map(producer_task, enumerate(self.items))
                
        assembler.join()
        print("All shards prepared successfully. Writing the final shards in queue...")
        writer_pool.shutdown()

        store.close()
        print("Zarr disk cache built successfully.")


    def _load_from_zarr(self, i: int):
        if self._zarr_root is None:
            store = LocalStore(self.cache_path, read_only=True)
            self._zarr_root = zarr.open(store, mode='r', zarr_format=3)

        data_parts = []
        j = 0
        while f'data_{j}' in self._zarr_root:
            data_parts.append(
                torch.from_numpy(self._zarr_root[f'data_{j}'][i])
            )
            j += 1
        
        if self._disk_cache_is_single_array:
            return data_parts[0]
        return tuple(data_parts)

    def _cache_ram(self):
        if not self.items:
            self._ram_cache = torch.utils.data.TensorDataset() # Handle empty case
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
            raise TypeError(f"The provided function must return a tensor or a tuple/list of tensors, but got {type(first_item_processed)}")

        stacked_tensors = [
            torch.empty((len(self), *template.shape), dtype=template.dtype, device=template.device) 
            for template in templates
        ]

        max_workers = min(128, ((os.cpu_count() - 2) // 2)*2 or 1)
        batch_size = 256
        fetch_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fetcher")
        fetched_queue : Queue[tuple[int, torch.Tensor]] = Queue(max(32, batch_size * 4))
        insert_buffer : dict[int, torch.Tensor] = dict()
        insert_queue : Queue[tuple[int, torch.Tensor]] = Queue()

        def _fetch_one(idx_item):
            idx, item = idx_item
            data = self.func(item)
            fetched_queue.put((idx, data))

        def _contiguous_write(
                indexes : list[int], 
                data : list[torch.Tensor] | list[list[torch.Tensor]]
            ) -> None:
            """
            Args:
                idx: A list of contigous increasing indices for each corresponding torch.Tensor (element) in `data`.
                data: A list of torch.Tensor with the same length as `idx`.
            """
            if len(indexes) == 0:
                return
            slc = slice(indexes[0], indexes[-1]+1)
            # Insert data into slice along first dimension in dst (in-place)
            if self._ram_was_single_tensor:
                torch.stack(data, out=stacked_tensors[0][slc])
            else:
                for i, elements in enumerate(zip(*data)):
                    torch.stack(elements, out=stacked_tensors[i][slc])

        def _write():
            end_idx = len(self) - 1
            pbar = TQDM(range(len(self)), desc="Writing to CPU RAM cache...")
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

        fetch_pool.map(_fetch_one, enumerate(self.items))

        nxt_idx = 0
        for _ in range(len(self)):
            idx, data = fetched_queue.get()
            if idx == nxt_idx:
                insert_queue.put((idx, data))
                nxt_idx += 1
                while nxt_idx in insert_buffer:
                    insert_queue.put((nxt_idx, insert_buffer.pop(nxt_idx)))
                    nxt_idx += 1
            else:
                insert_buffer[idx] = data

        write_thread.join()
        fetch_pool.shutdown()

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
                    raise NotImplementedError(f'Indexing with {i} is not implemented. Only integer, slice, or list/np.ndarray/torch.Tensor of integers indexing is supported.')
                return [torch.stack(values) for values in zip(*elements)]
            case CACHE_MODE.DISK:
                return self._load_from_zarr(i)
            case CACHE_MODE.CPU | CACHE_MODE.CUDA:
                data = self._ram_cache[i]
                return data[0] if self._ram_was_single_tensor else data
            case _:
                raise RuntimeError(f'Invalid caching mode found {self._cache}, expected one of None, "disk", "cpu" or "cuda".')

class ImageLoader:
    def __init__(
            self, 
            size : int | tuple[int, int], 
            cache : str | None=None, 
            dtype : torch.dtype=torch.uint8
        ):
        self.dtype, self.device = dtype, torch.device("cpu")
        self.cache = cache
        self.converter = make_convert_dtype(self.dtype)
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : str | Iterable):
        if isinstance(x, str):
            img = Image.open(x).convert("RGB").resize(self.shape, Image.Resampling.NEAREST)
            proc_img = pil_to_tensor(img).to(self.device)
            proc_img = self.converter(proc_img)
            if len(proc_img.shape) == 4:
                proc_img = proc_img[0]
            return proc_img
        return LazyDataset(self, x, self.cache)
    
class ImageClassLoader:
    def __init__(
            self, 
            class_decoder, 
            item_splitter : Callable[[Any], tuple[str, Any]]=lambda x : (x, x),
            resize_size : int=256, 
            cache : str | None=None,
            dtype : torch.dtype=torch.uint8
        ):
        self.dtype, self.device = dtype, torch.device("cpu")
        self.cache = cache
        self.converter = make_convert_dtype(self.dtype)
        self.splitter = item_splitter
        self.class_decoder = class_decoder
        size = resize_size
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : str | Iterable):
        if isinstance(x, str) or isinstance(x, tuple) and len(x) == 2:
            p, c = self.splitter(x)
            img = Image.open(p).convert("RGB").resize(self.shape, Image.Resampling.NEAREST)
            proc_img = pil_to_tensor(img).to(self.device)
            proc_img = self.converter(proc_img)
            if len(proc_img.shape) == 4:
                proc_img = proc_img[0]
            cls = self.class_decoder(c)
            return proc_img, cls
        return LazyDataset(self, x, self.cache)
