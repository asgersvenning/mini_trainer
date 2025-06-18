import hashlib
import os
from tempfile import gettempdir, mkdtemp
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import (InterpolationMode,
                                               pil_to_tensor, resize)
from tqdm.contrib.concurrent import thread_map

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

class LazyDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            func : Callable[[str], Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]]], 
            items : list[str],
            cache : Optional[str]=None
        ):
        self.func = func
        self.items = items
        self._cache = cache # one of None (no caching), disk (precompute .npy files), ram (preload cpu tensor)
        self.cache()

    def cache(self):
        match self._cache:
            case None:
                print("NO CACHE")
                pass
            case "disk":
                self.cache_dir = os.path.join(gettempdir(), ".mini_trainer_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                self._cache_disk()
            case "ram":
                self._cache_ram()
            case torch.device():
                raise NotImplementedError(f'On-device caching not implemented yet.')
            case _:
                raise RuntimeError(f'Invalid caching mode found {self._cache}, expected one of None, "disk" or "ram".')
    
    def _cache_disk(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self._disk_paths = []
        def _store_one(item_str):
            hashed_name = hashlib.sha256(item_str.encode()).hexdigest()
            path = os.path.join(self.cache_dir, f"{hashed_name}.npz")
            self._disk_paths.append(path)

            if os.path.exists(path):
                return

            data = self.func(item_str)
            tensors_to_save = [data] if isinstance(data, torch.Tensor) else data
            save_dict = {str(i): t.numpy() for i, t in enumerate(tensors_to_save)}
            np.savez(path, **save_dict)
        thread_map(_store_one, self.items, tqdm_class=TQDM, desc="Caching dataset on disk...", leave=False, max_workers=min(64, max(1, os.cpu_count())))            

    def _cache_ram(self):
        if not self.items:
            self._ram_cache = torch.utils.data.TensorDataset() # Handle empty case
            self._ram_was_single_tensor = False
            return

        first_item_processed = self.func(self.items[0])

        if isinstance(first_item_processed, torch.Tensor):
            self._ram_was_single_tensor = True
        elif isinstance(first_item_processed, (tuple, list)):
            self._ram_was_single_tensor = False
        else:
            raise TypeError(f"The provided function must return a tensor or a tuple/list of tensors, but got {type(first_item_processed)}")

        tensors_transposed = thread_map(
            self.func,
            self.items,
            tqdm_class=TQDM,
            desc="Caching dataset in RAM...",
            leave=False,
            max_workers=min(64, max(1, os.cpu_count()))
        )
        tensors_stacked = [torch.stack(tensors) for tensors in zip(*tensors_transposed)]

        self._ram_cache = torch.utils.data.TensorDataset(*[t.pin_memory() for t in tensors_stacked])


    def _read_disk_cache(self, i: int) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Loads a single processed item from its .npz file."""
        path = self._disk_paths[i]
        with np.load(path) as npz_file:
            sorted_keys = sorted(npz_file.keys(), key=int)
            tensors = [torch.from_numpy(npz_file[key]) for key in sorted_keys]
        return tensors[0] if len(tensors) == 1 else tensors
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        match self._cache:
            case None:
                return self.func(self.items[i])
            case "disk":
                return self._read_disk_cache(i)
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
