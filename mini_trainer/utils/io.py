import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from tempfile import gettempdir
from threading import Thread, Semaphore
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import torch
import zarr
from PIL import Image
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import (InterpolationMode,
                                               pil_to_tensor, resize)
from tqdm.contrib.concurrent import thread_map
from zarr.storage import LocalStore

from mini_trainer import TQDM
from mini_trainer.utils import make_convert_dtype


def is_image(path: str) -> bool:
    if not os.path.exists(path):
        return False
    
    try:
        with open(path, "rb") as f:
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
    dtype: Union[torch.dtype, str],
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

class LazyDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            func : Callable[[Any], Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]]], 
            items : list[str],
            cache : Optional[str]=None
        ):
        self.func = func
        self.items = items
        self._zarr_root = None
        self.cache_dir = None
        self._cache_mode = cache # one of None (no caching), disk (precompute .npy files), ram (preload cpu tensor)
        self._init_cache()

    @staticmethod
    def _hash_item(item):
        if isinstance(item, str): return item.encode('utf-8')
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
        if self._cache_mode is None:
            print("On-the-fly data loading enabled (no cache).")
            return
        if self._cache_mode == "disk":
            cache_dir = os.path.join(gettempdir(), ".mini_trainer")
            self.cache_path = os.path.join(cache_dir, f"{self._get_cache_hash()}.zarr")
            self._cache_disk_zarr()
        elif self._cache_mode == "ram":
            self._cache_ram()
        else:
            raise ValueError(f"Invalid cache mode '{self._cache_mode}'. Choose from [None, 'disk', 'ram'].")

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
        if not self.items: return
        
        first_item_processed = _normalize_to_tuple(self.func(self.items[0]))
        self._disk_cache_is_single_array = len(first_item_processed) == 1
        
        store = LocalStore(self.cache_path, read_only=False)
        root = zarr.open(store, zarr_format=3)
        
        # Create a Zarr array for each component of the data
        zarr_arrays = []
        shard_size = 256
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
            zarr_arrays.append(arr)

        # We need two pools: one for CPU-bound producers, one for I/O-bound writers
        producer_workers = min(64, (os.cpu_count() - 1) or 1) # Can be high
        writer_workers = min(64, (os.cpu_count() // 2) or 1) # Lower, I/O-limited
        
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
                        components = [[buffer[i][c] for i in indices] for c in range(len(zarr_arrays))]
                        
                        # Delegate the slow write operation to the writer pool
                        writer_semaphore.acquire()
                        writer_pool.submit(writer_task, indices, components)
                        
                        num_in_shard = len(indices)
                        for i in indices:
                            del buffer[i]
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
            self._zarr_root = zarr.open(store, mode='r')

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

        def _proc_one(idx_item):
            idx, item = idx_item
            data = self.func(item)
            if self._ram_was_single_tensor:
                stacked_tensors[0][idx] = data
            else:
                for i, element in enumerate(data):
                    stacked_tensors[i][idx] = element

        thread_map(
            _proc_one,
            enumerate(self.items),
            tqdm_class=TQDM,
            total=len(self),
            desc="Caching dataset in RAM...",
            max_workers=min(64, os.cpu_count() or 1)
        )

        self._ram_cache = torch.utils.data.TensorDataset(*[t for t in stacked_tensors])
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        match self._cache_mode:
            case None:
                return self.func(self.items[i])
            case "disk":
                return self._load_from_zarr(i)
            case "ram":
                data = self._ram_cache[i]
                return data[0] if self._ram_was_single_tensor else data
            case _:
                raise RuntimeError(f'Invalid caching mode found {self._cache}, expected one of None, "disk" or "ram".')

class ImageLoader:
    def __init__(
            self, 
            size : Union[int, tuple[int, int]], 
            cache : Optional[str]=None, 
            dtype : torch.dtype=torch.uint8
        ):
        self.dtype, self.device = dtype, torch.device("cpu")
        self.cache = cache
        self.converter = make_convert_dtype(self.dtype)
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str):
            img = decode_image(x, ImageReadMode.RGB)
            ds = min([max(1, im_d // lo_d) for im_d, lo_d in zip(img.shape[-2:], self.shape)])
            if ds > 1:
                img = img[..., ::ds, ::ds]
            proc_img : torch.Tensor = resize(self.converter(img), self.shape, InterpolationMode.NEAREST).to(self.device)
            return proc_img
        return LazyDataset(self, x, self.cache)
    
class ImageClassLoader:
    def __init__(
            self, 
            class_decoder, 
            item_splitter : Callable[[Any], tuple[str, Any]]=lambda x : (x, x),
            resize_size : Optional[int]=None, 
            cache : Optional[str]=None,
            dtype : torch.dtype=torch.uint8
        ):
        self.dtype, self.device = dtype, torch.device("cpu")
        self.cache = cache
        self.converter = make_convert_dtype(self.dtype)
        self.splitter = item_splitter
        self.class_decoder = class_decoder
        size = resize_size
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str) or isinstance(x, tuple) and len(x) == 2:
            p, c = self.splitter(x)
            img = decode_image(p, ImageReadMode.RGB)
            ds = min([max(1, im_d // lo_d) for im_d, lo_d in zip(img.shape[-2:], self.shape)])
            if ds > 1:
                img = img[..., ::ds, ::ds]
            proc_img : torch.Tensor = resize(self.converter(img), self.shape, InterpolationMode.NEAREST)
            proc_img = proc_img.to(self.device)
            if len(proc_img.shape) == 4:
                proc_img = proc_img[0]
            cls = self.class_decoder(c)
            return proc_img, cls
        return LazyDataset(self, x, self.cache)
