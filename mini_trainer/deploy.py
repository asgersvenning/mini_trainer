import difflib
import os
import re
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar, cast

import numpy as np
import torch
from tqdm.auto import tqdm

from mini_trainer.builders import BaseBuilder
from mini_trainer.classifier import EmbeddingContext, classification_module, predict, set_classification_mask
from mini_trainer.hierarchical.predict import cli as default_args
from mini_trainer.predict import main as mt_predict
from mini_trainer.utils.io import make_read_and_resize_fn


def _download_chunk(url, start, end, tmp_file, bar):
    """Worker function to download a specific byte range."""
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0', 'Range': f'bytes={start}-{end}'}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        # Open in read-write-binary to seek to the correct offset
        with open(tmp_file, 'r+b') as f:
            f.seek(start)
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk)) # tqdm is generally thread-safe for basic updates


def download(url, dest=None, workers=4):
    """Downloads a file, using concurrent connections if supported by the server."""
    if not dest:
        dest = os.path.basename(urllib.parse.urlparse(url).path)
        if not dest:
            raise ValueError("Cannot determine filename.")
            
    dest_dir = os.path.dirname(dest)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    tmp = dest + ".tmp"
    
    # 1. Probe the server with a HEAD request to check capabilities
    head_req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(head_req, timeout=10) as r:
            total_size = int(r.headers.get('Content-Length', 0))
            supports_ranges = r.headers.get('Accept-Ranges') == 'bytes'
    except Exception as e:
        raise RuntimeError(f"Failed to probe URL {url}") from e

    try:
        # 2. Decide between Concurrent vs. Single-Thread
        # Fall back to 1 worker if the server rejects ranges or the file is small (< 1MB)
        if not supports_ranges or total_size < 1024 * 1024:
            workers = 1

        # 3. Pre-allocate the empty file on disk
        with open(tmp, 'wb') as f:
            f.truncate(total_size)

        # 4. Calculate byte ranges for each worker
        chunk_size = total_size // workers
        ranges = []
        for i in range(workers):
            start = i * chunk_size
            # The last worker grabs whatever bytes remain
            end = total_size - 1 if i == workers - 1 else (start + chunk_size - 1)
            ranges.append((start, end))

        # 5. Execute the download
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=f'Downloading "{dest}"') as bar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(_download_chunk, url, start, end, tmp, bar) 
                    for start, end in ranges
                ]
                
                for future in as_completed(futures):
                    future.result() 

        os.replace(tmp, dest)
        return dest

    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


V = TypeVar("V")


class SmartDict[V]:
    """Smart and flexible dictionary wrapper."""
    n : str = "item"

    def __init__(self, d : dict[str, V], default : str | None=None):  # noqa: D107
        self.default = default
        self.d, self._c = d, lambda s: re.sub(r'\W+', '', str(s).lower())
        self.m = {self._c(k): k for k in d}

    def __call__(self, q : str | None) -> V:
        if not q:
            if self.default is None:
                raise ValueError("Please supply a query, no default query specified.")
            q = self.default
        nq = self._c(q)
        if nq in self.m: 
            return self.d[self.m[nq]]
        if len(h := [k for k in self.m if k.startswith(nq)]) == 1:
            return self.d[self.m[h[0]]]
        if len(h) > 1:
            raise ValueError(f"Ambiguous {self.n} '{q}': {', '.join(self.m[x] for x in h)}")
        s = difflib.get_close_matches(q, self.d.keys(), n=1)
        raise KeyError(f"No {self.n} '{q}'." + (f" Did you mean '{s[0]}'?" if s else ""))


MODELS : dict[str, str] = {
    "source" : "hierarchical_bioclip2_ft_v0.pt",
    "old" : "hierarchical_bioclip2_ft_reduced_v0.pt",
    "europe" : "hierarchical_bioclip2_ft_eu_v1.pt",
    "full" : "hierarchical_bioclip2_ft_v1.pt",
    "north_europe" : "hierarchical_bioclip2_ft_neu_v1.pt"
}
DEFAULT_MODEL = "europe"
MODEL_TABLE = SmartDict(MODELS, DEFAULT_MODEL)

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
    
    if not model or not (model.endswith(".pt") or model.endswith(".pth")):
        model = MODEL_TABLE(model)
    
    if os.path.exists(model) and os.path.isfile(model):
        return model, model

    src = SRC_TEMPLATE.format(model)
    dst = os.path.join(weight_dir, model)

    if not os.path.exists(dst):
        download(src, dst)
    
    return dst, src


class Predictor:
    """A wrapped inference model class."""
    def __init__(  # noqa: D417
            self, 
            device : str="cuda", 
            model : str | None=None,
            weights : dict[str, Any | torch.Tensor] | None=None,
            class_mask : list[int] | list[str] | torch.Tensor | np.ndarray | int | None=None, 
            **kwargs
        ):
        """A wrapped inference model.
        
        Args:
            device: Model/Inference device, default="cuda".
            model: Optional name of model (or path to weights).
            weights: Optional state dict used for loading (instead of specifying a `model`).
            class_mask: Optional mask for possible classes in model output. 
                Set to `-1` to reset the mask; ensure that all classes are allowed.
            kwargs: Optional, can be used to specify a different model or directory 
                for storing/caching the weights locally.
        """
        self.device = torch.device(device)
        if model and weights:
            raise ValueError('Specifying both `model` and `weights` is ambigous, please use only one.')
        if weights is None:
            self.weights, self.source = ensure_weights(model=model, **kwargs)
        else:
            self.weights = weights
            self.source = None
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

    def __call__(self, x):
        return self.predict(x)

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
    
    def predict(
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

    def predict_with_embeddings(
            self, 
            x : str | np.ndarray | torch.Tensor | Iterable[str | np.ndarray | torch.Tensor], 
            **kwargs
        ):
        """Perform inference on one or more images, and include embeddings in return.
        
        Computation is done as simply, efficiently and flexibly as possible, this means that
        all inputs are stacked into a single batch and inference is done in a single pass.
        
        If you need to predict on a larger number of images it may be beneficial to preload
        the images with multiple workers to avoid IO bottlenecks, and manually batch to avoid
        OOM errors.
        
        Args:
            x : Input images, supports various mixed formats, such as paths, NumPy arrays and torch.Tensors.
            kwargs : Passed to model prediction function.
        """
        with EmbeddingContext():
            return self.predict(x, **kwargs), EmbeddingContext.get()


def run():
    args = default_args(
        model={
            None : "-M",
            "type" : str,
            "default" : False,
            "required" : False,
            "help" : "Convert GBIF IDs in output to scientific names via GBIF API."
        }
    )
    model = args.pop("model", None)
    if args["weights"] is None:
        args["weights"] = ensure_weights(model=model)[0]
    elif model is not None:
        raise ValueError('Specifying both weights and model is ambigous, please use only one or the other.')
    if args["name"] in [None, "predict"]:
        args["name"] = "results"
    
    mt_predict(**args)


if __name__ == "__main__":
    run()