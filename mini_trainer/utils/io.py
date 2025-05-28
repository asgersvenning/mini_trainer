from typing import Iterable, Optional, Union, Callable, Any

import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize

from mini_trainer.utils import convert2bf16, convert2fp16, convert2fp32


def is_image(path: str) -> bool:
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
        mode : ImageReadMode, 
        size : tuple[int, int], 
        device : torch.types.Device,
        dtype : Union[torch.dtype, str], 
        **kwargs
    ):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f'Specified data type "{dtype}" is not a known valid torch data type')
    match dtype:
        case torch.float16:
            converter = convert2fp16
        case torch.float32:
            converter = convert2fp32
        case torch.bfloat16:
            converter = convert2bf16
        case _:
            raise ValueError("Only fp16 supported for now.")
    def read_and_resize(path):
        return resize(converter(decode_image(path, mode)), size=size, **kwargs).to(device)
    return read_and_resize

class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, func : Callable[[str], torch.Tensor], items : list[str]):
        self.func = func
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.func(self.items[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ImageLoader:
    def __init__(self, preprocessor, dtype, device):
        self.dtype, self.device = dtype, device
        self.preprocessor = preprocessor
        match self.dtype:
            case torch.float16:
                self.converter = convert2fp16
            case torch.float32:
                self.converter = convert2fp32
            case torch.bfloat16:
                self.converter = convert2bf16
            case _:
                raise ValueError("Only fp16 supported for now.")
        size = self.preprocessor.resize_size if hasattr(self.preprocessor, "resize_size") else 256
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str):
            proc_img : torch.Tensor = self.preprocessor(resize(self.converter(decode_image(x, ImageReadMode.RGB)), self.shape)).to(self.device)
            return proc_img
        return LazyDataset(self, x)
    
class ImageClassLoader:
    def __init__(
            self, 
            class_decoder, 
            item_splitter : Callable[[Any], tuple[str, Any]]=lambda x : (x, x),
            resize_size : Optional[int]=None, 
            preprocessor : Callable[[torch.Tensor], torch.Tensor]=lambda x : x, 
            dtype=torch.float32, 
            device=torch.device("cpu")
        ):
        self.dtype, self.device = dtype, device
        self.preprocessor = preprocessor
        self.splitter = item_splitter
        self.class_decoder = class_decoder
        match self.dtype:
            case torch.float16:
                self.converter = convert2fp16
            case torch.float32:
                self.converter = convert2fp32
            case torch.bfloat16:
                self.converter = convert2bf16
            case _:
                raise ValueError("Only fp16 supported for now.")
        size = resize_size if resize_size is not None else self.preprocessor.resize_size if hasattr(self.preprocessor, "resize_size") else resize_size
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str) or isinstance(x, tuple) and len(x) == 2:
            p, c = self.splitter(x)
            proc_img : torch.Tensor = self.preprocessor(resize(self.converter(decode_image(p, ImageReadMode.RGB)), self.shape)).to(self.device)
            if len(proc_img.shape == 4):
                proc_img = proc_img[0]
            cls = self.class_decoder(c)
            return proc_img, cls
        return LazyDataset(self, x)
