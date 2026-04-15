import os
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import cast

import numpy as np
import torch
from tqdm.auto import tqdm

from mini_trainer.builders import BaseBuilder
from mini_trainer.classifier import classification_module, predict, set_classification_mask
from mini_trainer.hierarchical.predict import cli as default_args
from mini_trainer.predict import main as mt_predict
from mini_trainer.utils.io import make_read_and_resize_fn


def download(url, dest=None):
    if not dest:
        dest = os.path.basename(urllib.parse.urlparse(url).path)
        if not dest:
            raise ValueError("Cannot determine filename from URL; please provide a destination.")
            
    tmp = dest + ".tmp"
    
    try:
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
            total = int(r.headers.get('Content-Length', 0))
            with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024, desc=f'Downloading "{dest}"') as bar:
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        os.replace(tmp, dest)
    except Exception as e:
        e.add_note(f'Error while attempting to download {url} to {tmp}')
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


DEFAULT_MODEL = "hierarchical_bioclip2_ft_reduced_v0.pt"
SRC_TEMPLATE = "https://anon.erda.au.dk/share_redirect/HE90eyuZCT/MAMBO/{}"
_tmp_dir = cast(str, tempfile.tempdir)
if _tmp_dir is None:
    _tmp_dir = os.path.expanduser("~/.mini_trainer_cache")


def ensure_weights(model : str | None=None, weight_dir : str | None=None):
    if weight_dir is None:
        weight_dir = _tmp_dir
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir, exist_ok=True)
    if not os.path.isdir(weight_dir):
        raise NotADirectoryError(f'Weight directory: {weight_dir}, is not a directory.')
    
    if model is None:
        model = DEFAULT_MODEL
    src = SRC_TEMPLATE.format(model)
    dst = os.path.join(weight_dir, model)

    if not os.path.exists(dst):
        download(src, dst)
    
    return dst


class Predictor:
    """A wrapped inference model class."""
    def __init__(
            self, 
            device : str="cuda", 
            class_mask : list[int] | list[str] | torch.Tensor | np.ndarray | int | None=None, 
            **kwargs
        ):
        """A wrapped inference model.
        
        Args:
            device: Model/Inference device, default="cuda".
            class_mask: Optional mask for possible classes in model output. 
                Set to `-1` to reset the mask; ensure that all classes are allowed.
            kwargs: Optional, can be used to specify a different model or directory 
                for storing/caching the weights locally.
        """
        self.device = torch.device(device)
        self.weights = ensure_weights(**kwargs)
        self.model, self.preproc = BaseBuilder.build_model(weights=self.weights)
        self.resize_size = classification_module(self.model).metadata.get("resize_size", None)
        if not isinstance(self.resize_size, int):
            raise RuntimeError(
                f'Failed to extract a valid input size for {type(self.model)=} with {self.weights=}, '
                f'found {self.resize_size}, but expected an integer.'
            )
        self.reader = make_read_and_resize_fn((self.resize_size, self.resize_size), self.device, torch.uint8)
        if class_mask is not None:
            self._apply_class_mask(class_mask)
        self.model.to(device=device)
        self.model.eval()

    def _apply_class_mask(self, class_mask : list[int] | list[str] | torch.Tensor | np.ndarray | int | None):
        if isinstance(class_mask, int):
            if class_mask == -1:
                class_mask = None
            else:
                raise ValueError(
                    f'`class_mask` was interpreted as an integer flag, but {class_mask=} is not known.'
                )
        if class_mask is not None:
            cls2idx = classification_module(self.model).metadata.get("cls2idx", None)
            if cls2idx is not None:
                _sp_cls2idx = cls2idx.get("0", cls2idx.get(0, None))
                if isinstance(_sp_cls2idx, dict) and _sp_cls2idx:
                    cls2idx = _sp_cls2idx
                if hasattr(class_mask, "__iter__") and not isinstance(class_mask, (torch.Tensor, np.ndarray)):
                    class_mask = [cls2idx[cls] if isinstance(cls, str) else cls for cls in class_mask]
        
        if isinstance(class_mask, list):
            assert not any(map(lambda x : isinstance(x, str), class_mask))
            class_mask = cast(list[int], class_mask)
        
        set_classification_mask(self.model, class_mask)
    
    def __call__(
            self, 
            x : str | np.ndarray | torch.Tensor | Iterable[str | np.ndarray | torch.Tensor], 
            **kwargs
        ):
        """Perform inference on one or more images.
        
        Computation is done as simply, efficiently and flexibly as possible, this means that
        all inputs are stacked into a single batch and inference is done in a single pass.
        
        If you need to predict on a larger number of images it may be beneficial to preload
        the images with multiple workers to avoid IO bottlenecks, and manually batch to avoid
        OOM errors.
        
        Args:
            x : Input images, supports various mixed formats, such as paths, NumPy arrays and torch.Tensors.
            kwargs : Passed to model prediction function.
        """
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            # Ensure tensor
            if isinstance(x, str):
                x = self.reader(x)
            elif hasattr(x, "__iter__") and not isinstance(x, (torch.Tensor, np.ndarray)):
                x = torch.stack([self.reader(xi) if isinstance(xi, str) else torch.as_tensor(xi) for xi in x])
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x)
            
            # Ensure proper shape
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.ndim == 3 and x.size(0) == 1:
                x = x.repeat(3, 1, 1)
            if x.ndim == 4 and x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            if x.ndim == 3:
                x = x.unsqueeze(0)

            # Preprocess and ensure device
            x = self.preproc(x).to(self.device)

            # Predict and return
            return predict(self.model, x, **kwargs)


def run():
    args = default_args()

    if args["weights"] is None:
        args["weights"] = ensure_weights()
    if args["name"] in [None, "predict"]:
        args["name"] = "results"
    
    mt_predict(**args)


if __name__ == "__main__":
    run()