import glob
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from hierarchical.base.model import HierarchicalClassifier
from hierarchical.base.utils import create_hierarchy, mask_hierarchy
from torch import nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mini_trainer.classifier import get_model
from mini_trainer.utils import ImageClassLoader, get_image_data

def hierarchical_parse_class_index(
        path : Optional[str]=None, 
        dir : Optional[str]=None, 
        dir2comb_fn : Optional[Callable[[str], List[Tuple[str, ...]]]]=None
    ):
    """
    Accepts a path to a class index file or a directory with named subdirectories for each class.

    Alternatively `dir` can be used with a custom function. The custom function should return a topologically sorted list of all combinations of classes (root-to-leaf) as a tuple of strings.
    """
    if path is None or not os.path.exists(path):
        if dir2comb_fn is None:
            combinations = sorted(set(tuple(f.split(os.sep)[:-1]) for f in glob.glob("**", root_dir=dir, recursive=True) if not os.path.isdir(os.path.join(dir, f))))
        else:
            combinations = dir2comb_fn(dir)
        classes = {i : [] for i in range(len(next(iter(combinations))))}
        for e in combinations:
            for lvl, cls in enumerate(e):
                if cls not in classes[lvl]:
                    classes[lvl].append(cls)
        classes = {lvl : classes[lvl] for lvl in range(len(classes))}
        cls2idx = {lvl : {cls : idx for idx, cls in enumerate(classes[lvl])} for lvl in range(len(classes))}
        if path is not None:
            with open(path, "w") as f:
                json.dump({"cls2idx" : cls2idx, "combinations" : combinations}, f)
    else:
        with open(path, "rb") as f:
            c2i_comb = json.load(f)
            cls2idx = {int(lvl) : e for lvl, e in c2i_comb["cls2idx"].items()}
            combinations = c2i_comb["combinations"]
    cls = [list(cls2idx[lvl]) for lvl in range(len(cls2idx))]
    idx2cls = {lvl : {v : k for k, v in cls2idx[lvl].items()} for lvl in range(len(cls2idx))}
    ncls = [len(clvl) for clvl in cls]
    hierarchy = create_hierarchy(combinations, cls2idx)
    masks = mask_hierarchy(hierarchy, zero=-100)
    return {"num_classes" : ncls[-1], "masks" : list(reversed(masks))}, {"classes" : cls, "class2idx" : cls2idx, "idx2class" : idx2cls}

def hierarchical_model_builder(
        model : str,
        masks : List[torch.Tensor],
        weights : Optional[str],
        fine_tune : bool,
        device : torch.device,
        dtype : torch.dtype,
        num_classes : int,
        **kwargs : Any
    ) -> Tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if len(kwargs) != 0:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(f"my_fun() got unexpected keyword argument(s): {unexpected}")
    model, head_name, model_preprocess = get_model(model)
    model : nn.Module
    if not isinstance(model, nn.Module):
        raise TypeError(f"Unknown model type `{type(model)}`, expected `{nn.Module}`")
    head = getattr(model, head_name)
    num_embeddings = (head[1] if not isinstance(head, nn.Linear) else head).in_features
    if weights is not None:
        model = HierarchicalClassifier.load(
            model_type=model, 
            path=weights, 
            masks=[m.to(device, torch.float32) for m in masks], 
            device=device, 
            dtype=torch.float32
        )
    else:
        classifier = HierarchicalClassifier(
            in_features=num_embeddings, 
            out_features=num_classes, 
            masks=[m.to(device, torch.float32) for m in masks]
        )
        setattr(model, head_name, classifier)
        model.to(device, torch.float32)
    if fine_tune:
        for name, param in model.named_parameters():
            if param.requires_grad and not head_name in name:
                param.requires_grad_(False)
    return model, model_preprocess

def hierarchical_base_path2cls2idx_builder(class2idx):
    def path2cls2idx(path, cls2idx=class2idx, nlvl=len(class2idx)):
        return torch.tensor(list(reversed([cls2idx[lvl][cls] for lvl, cls in enumerate(path.split(os.sep)[:-1][-nlvl:])]))).long()
    return path2cls2idx

def multi_level_collate(batch):
    return tuple(torch.stack(v) for v in zip(*batch))

def hierarchical_dataloader_builder(
        data_index : Optional[str],
        input_dir : str,
        classes : List[str],
        class2idx : Dict[str, int],
        preprocess : Callable[[torch.Tensor], torch.Tensor],
        batch_size : int,
        device : torch.device,
        dtype = torch.dtype,
        resize_size : Optional[int]=None,
        train_proportion : float=0.9,
        idx2class : Optional[Dict[int, str]]=None,
        num_workers : Optional[int]=None,
        path2cls2idx_builder : Callable[[Any], Callable[[str], torch.Tensor]]=hierarchical_base_path2cls2idx_builder,
        path2cls2idx_builder_kwargs : Dict[str, Any]={}
    ):
    path2cls2idx = path2cls2idx_builder(class2idx=class2idx, **path2cls2idx_builder_kwargs)
    # Prepare datasets/dataloaders
    if data_index is None:
        all_files = [path for f in glob.glob("**", root_dir=input_dir, recursive=True) if not os.path.isdir(path := os.path.join(input_dir, f))]
        data = {
            "path" : [],
            "class" : [],
            "split" : []
        }
        for path in all_files:
            data["path"].append(path)
            data["class"].append(path2cls2idx(path).tolist())
            data["split"].append("train" if random.random() < train_proportion else "validation")
        data = {k : np.array(v) for k, v in data.items()}
        train_image_data = {k : v[data["split"] == np.array("train")] for k, v in data.items()}
        val_image_data = {k : v[data["split"] == np.array("validation")] for k, v in data.items()}
    else:
        train_image_data, val_image_data = get_image_data(data_index)
        
    resize_size = preprocess.resize_size if hasattr(preprocess, "resize_size") else resize_size

    if not isinstance(resize_size, int):
        if not (isinstance(resize_size, tuple) and len(resize_size) == 2 and all(map(lambda x : isinstance(x, int), resize_size))):
            raise TypeError(f'Invalid resize size passed, foun {resize_size}, but expected an integer or a tuple of two integers')
    print(f"Building datasets with image size {resize_size}")

    loader = ImageClassLoader(path2cls2idx, resize_size=resize_size, preprocessor=lambda x : x, dtype=dtype, device=torch.device("cpu"))
    train_dataset = loader(train_image_data["path"])
    val_dataset = loader(val_image_data["path"])

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    if num_workers is None:
        num_workers = os.cpu_count() - 1
        num_workers -= num_workers % 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=multi_level_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler, 
        collate_fn=multi_level_collate,
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader

from mini_trainer.utils import BaseResultCollector, confusion_matrix


class MultiLevelResultCollector(BaseResultCollector):
    def __init__(self, lvl : int, *args, **kwargs):
        self.level = lvl
        super().__init__(*args, **kwargs)

    def evaluate(self):
        if self.training_format:
            data = self.data
            data["gt"] = [f.split(os.sep)[-(1 + (3 - self.level))] for f in data["path"]]

            confusion_matrix(data, self.idx2class)