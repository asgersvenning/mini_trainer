import json
import os
import random
from glob import glob
from typing import Optional, Union

import numpy as np
import torch
from torchvision.io import ImageReadMode

from mini_trainer import TQDM
from tqdm.contrib.concurrent import thread_map
from mini_trainer.utils.io import is_image, make_read_and_resize_fn


def write_metadata(directory : str, classes : list[str], cls2idx : dict[str, int], dst : str, train_proportion : float=0.9):
    data = {
        "path" : [],
        "class" : [],
        "split" : []
    }
    for cls in classes:
        this_dir = os.path.join(directory, cls)
        for file in map(os.path.basename, os.listdir(this_dir)):
            data["path"].append(os.path.join(this_dir, file))
            data["class"].append(cls2idx[cls])
            data["split"].append("train" if random.random() < train_proportion else "validation")
    with open(dst, "w") as f:
        json.dump(data, f)

def get_image_data(path : str, check_integrity : bool=False):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Meta data file ({path}) for training split not found. Please provide a JSON with the following keys: "path", "class", "split".')
        with open(path, "rb") as f:
            _image_data = {k : np.array(v) for k, v in json.load(f).items()}
        if check_integrity:
            image_data = {
                k : v[
                    np.array(
                        thread_map(
                            is_image, 
                            _image_data["path"], 
                            tqdm_class=TQDM, 
                            desc="Filtering non-standardized images...", 
                            total=len(_image_data["path"])
                        )
                    )
                ]
                for k, v in _image_data.items()
            }
        else:
            image_data = _image_data
        train_image_data = {k : v[image_data["split"] == np.array("train")] for k, v in image_data.items()}
        val_image_data = {k : v[image_data["split"] == np.array("validation")] for k, v in image_data.items()}
        return train_image_data, val_image_data

def find_images(root: str) -> list[str]:
    paths = glob(os.path.join(root, "**"), recursive=True)
    flags = thread_map(is_image, paths, tqdm_class=TQDM, desc="Filtering non-standardized images...", total=len(paths))
    return [p for p, f in zip(paths, flags) if f]

def parse_class_index(path : Optional[str]=None, dir : Optional[str]=None):
    if not os.path.exists(path):
        if dir is None or not os.path.isdir(dir):
            raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not \'{dir}\'.')
        cls2idx = {cls : i for i, cls in enumerate(sorted([f for f in map(os.path.basename, os.listdir(dir)) if os.path.isdir(os.path.join(dir, f))]))}
        if path is not None:
            with open(path, "w") as f:
                json.dump(cls2idx, f)
    else:
        with open(path, "rb") as f:
            cls2idx = json.load(f)
    cls = list(cls2idx.keys())
    idx2cls = {v : k for k, v in cls2idx.items()}
    ncls = len(idx2cls)
    return {"num_classes" : ncls}, {"classes" : cls, "cls2idx" : cls2idx, "idx2cls" : idx2cls}

def prepare_split(
        paths : list[str], desc="Preprocessing images for split...", 
        resize_size : Union[int, tuple[int, int]]=256, 
        device=torch.device("cpu"), 
        dtype=torch.bfloat16
    ):
    shape = resize_size if not isinstance(resize_size, int) and len(resize_size) == 2 else (resize_size, resize_size)
    tensor = torch.zeros((len(paths), 3, *shape), device=device, dtype=dtype)
    reader = make_read_and_resize_fn(ImageReadMode.RGB, shape, device, dtype)
    def _read_and_resize_one_image(args):
        idx, path = args
        try:
            tensor[idx] = reader(path)
        except Exception as e:
            e.add_note(f'Path: {path}')
            raise e
    [_read_and_resize_one_image(v) for v in TQDM(enumerate(paths), total=len(paths), desc=desc)]
    return tensor