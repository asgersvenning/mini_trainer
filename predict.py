import os
import tempfile
import urllib.parse
import urllib.request
from typing import cast

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from mini_trainer.builders import BaseBuilder
    from mini_trainer.classifier import predict, set_classification_mask
    from mini_trainer.hierarchical.predict import cli as default_args
    from mini_trainer.predict import main as mt_predict
except ImportError as e:
    e.add_note("`mini_trainer` does not seem to be installed, try `pip install -e .`.")
    raise e


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


DEFAULT_MODEL = "hierarchical_bioclip2_ft_v0.pt"
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
            class_mask : list[int] | torch.Tensor | np.ndarray | int | None=None, 
            **kwargs
        ):
        """A wrapped inference model.
        
        Args:
            device: Model/Inference device, default="cuda".
            class_mask: Optional mask for possible classes in model output.
            kwargs: Optional, can be used to specify a different model or directory 
                for storing/caching the weights locally.
        """
        self.device = torch.device(device)
        self.weights = ensure_weights(**kwargs)
        self.model, self.preproc = BaseBuilder.build_model(weights=self.weights)
        if class_mask is not None:
            if isinstance(class_mask, int):
                if class_mask == -1:
                    class_mask = None
                else:
                    raise ValueError(
                        f'`class_mask` was interpreted as an integer flag, but {class_mask=} is not known.'
                    )
            set_classification_mask(self.model, class_mask)
        self.model.to(device=device)
        self.model.eval()
    
    def __call__(self, x, **kwargs):
        with torch.inference_mode():
            x = self.preproc(x)
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