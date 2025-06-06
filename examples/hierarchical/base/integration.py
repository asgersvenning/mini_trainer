import glob
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from hierarchical.base.loss import MultiLevelWeightedCrossEntropyLoss, MultiLevelCrossEntropyLoss
from hierarchical.base.model import HierarchicalClassifier
from hierarchical.base.setup import names_or_ids_to_combinations, resolve_name_or_id
from hierarchical.base.utils import (create_hierarchy, leaf_to_parents,
                                     mask_hierarchy)
from torch import nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mini_trainer.builders import BaseBuilder
from mini_trainer.utils.data import get_image_data
from mini_trainer.utils.io import ImageClassLoader
from mini_trainer.utils.logging import BaseResultCollector

DEFAULT_HIERARCHY_LEVELS = ("species", "genus", "family")

# def hierarchical_base_path2cls2idx_builder(cls2idx):
#     def path2cls2idx(path, cls2idx=cls2idx, nlvl=len(cls2idx)):
#         return torch.tensor(list(reversed([cls2idx[lvl][cls] for lvl, cls in enumerate(path.split(os.sep)[:-1][-nlvl:])]))).long()
#     return path2cls2idx

# def hierarchical_path2cls2idx_w_classindex(*args, class_index : str, **kwargs):
#     with open(class_index, "r") as f:
#         class_index = json.load(f)
#     cls2idx = class_index["cls2idx"]
#     combinations = class_index["combinations"]
#     cls2idxs = {comb[0] : [cls2idx[str(lvl)][c] for lvl, c in enumerate(comb)] for comb in combinations}
#     def path2cls2idx(path):
#         return torch.tensor(cls2idxs[os.path.basename(os.path.dirname(path))])
#     return path2cls2idx

def multi_level_collate(batch):
    return tuple(torch.stack(v) for v in zip(*batch))

def hierarchical_class_index_to_standard(path : str):
    with open(path, "r") as f:
        data = json.load(f)
    return data["cls2idx"]["0"]

def hierarchy_to_combinations(path : str, levels : list[str]=DEFAULT_HIERARCHY_LEVELS):
    """
    Args:
        path (`str`): Path to a JSON file containing a key for each leaf class (most likely species), which should have a corresponding folder of images with the same name. 
            Each value should be a dictionary with, at least, the keys given by `levels`, with values being being a list of class labels for each level. 
            I will assume that the first class label in the list is to be used as the primary class label. This can be used for example to save both the GBIF ID, scientific name and vernacular name. 
        levels (`list[str]`): A list of names for the class hierarchy levels. The names should be ordered from highest (root) to lowest (leaf).

    Returns:
        combinations (`list[tuple[str, ...]]`): A topologically sorted list of all unique combinations of classes (root-to-leaf) as a tuple of strings.
    """
    with open(path, "rb") as f:
        return list(sorted(set([tuple([leaf[level][0] for level in levels]) for leaf in json.load(f).values()])))

class HierarchicalPathParser:
    def __init__(
            self, 
            class_index : str, 
            levels : list[str]=DEFAULT_HIERARCHY_LEVELS, 
            as_tensor : bool=False, 
            reversed : bool=False, 
            name2cls : Optional[dict[str, int]]=None,
            *args, 
            **kwargs
        ):
        self.levels = levels
        self.as_tensor = as_tensor
        self.reversed = reversed
        self.name2cls = name2cls
        with open(class_index, "rb") as f:
            _class_index_data = json.load(f)
        self.cls2idx = _class_index_data["cls2idx"]
        if len(self.levels) > len(self.cls2idx):
            raise ValueError(f'Unable to initialize path to hierarchical class label parser. More levels ({levels}) specified than the number of class-to-index dictionaries found in {class_index}.')
        self.combinations : list = _class_index_data["combinations"]
        comb_leafs = [c[0] for c in self.combinations]
        self.hierarchy = {leaf : {level : [cls] for level, cls in zip(self.levels, self.combinations[comb_leafs.index(leaf)])} for leaf in self.cls2idx["0"]}

    def __call__(self, path : str):
        cls = path_to_hierarchy(path, self.hierarchy, self.cls2idx, self.levels, self.name2cls)
        if self.reversed:
            cls = cls[::-1]
        if self.as_tensor:
            return torch.tensor(cls, dtype=torch.long)
        return cls

def path_to_hierarchy(
        path : str, 
        hierarchy : dict[str, dict[str, tuple[str, ...]]],
        cls2idx : Optional[dict[str, dict[str, int]]]=None, 
        levels : list[str]=DEFAULT_HIERARCHY_LEVELS,
        name2cls : Optional[dict[str, int]]=None
    ):
    if os.path.sep in path:
        parts = path.split(os.path.sep)
    else:
        parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f'Expected `path` to be an absolute path or a relative path stump containing at least one parent directory, not {path} which only contains {len(parts)} path components.')
    leaf = parts[-2]
    if name2cls is not None:
        leaf = str(name2cls[leaf])
    out = tuple([hierarchy[leaf][level][0] for level in levels])
    if cls2idx is not None:
        out = tuple([cls2idx[str(li)][cls] for li, cls in enumerate(out)])
    return out   

def parse_quality_control(
        path : str, 
        correct_class : str="1",
        confidence_threshold : float=0.8,
        max_per_cls : Optional[int]=None,
        path2cls_builder : Callable[[Any], Callable[[str], Any]]=HierarchicalPathParser, 
        **kwargs
    ):
    with open(path, "rb") as f:
        data = json.load(f)
    if max_per_cls is not None:
        cls_counter = defaultdict(lambda : 0)
    path2cls = path2cls_builder(**kwargs)
    paths, clss = [], []
    for path, pred, conf in zip(*[data[key] for key in ["paths", "preds", "confs"]]):
        if not (pred == correct_class and conf >= confidence_threshold):
            continue
        cls = path2cls(path)
        if max_per_cls is not None:
            if cls_counter[cls] < max_per_cls:
                cls_counter[cls] += 1
            else:
                continue
        paths.append(path)
        clss.append(cls)
    return paths, clss

def hierarchical_create_data_index(
        path : str,
        outpath : Optional[str],
        parse_fn : Callable[[str], tuple[list[str], list[Any]]]=parse_quality_control,
        split : tuple[float, ...]=(0.8, 0.1, 0.1),
        split_labels : tuple[str, ...]=("train", "validation", "test"),
        **kwargs
    ):
    paths, cls = parse_fn(path, **kwargs)
    n = len(paths)
    spl = random.sample([lab for prop, lab in zip(split, split_labels) for _ in range(math.ceil(n * prop))], n)
    data = {k : v for k, v in zip(["path", "class", "split"],[paths, cls, spl])}
    if outpath:
        if os.path.exists(outpath):
            os.remove(outpath)
        with open(outpath, "w") as f:
            json.dump(data, f)
    else:
        return data

class HierarchicalBuilder(BaseBuilder):
    @staticmethod
    def spec_model_dataloader(
            path : Optional[str]=None, 
            dir : Optional[str]=None, 
            dir2comb_fn : Optional[Callable[[str], list[tuple[str, ...]]]]=None
        ):
        """
        Accepts a path to a class index file or a directory with named subdirectories for each class.

        Alternatively `dir` can be used with a custom function. The custom function should return a topologically sorted list of all combinations of classes (root-to-leaf) as a tuple of strings.
        """
        if path is None or not os.path.exists(path):
            if dir2comb_fn is None:
                # This function can be used if the images are stored in a hierarchical directory structure:
                # combinations = sorted(set(tuple(f.split(os.sep)[:-1]) for f in glob.glob("**", root_dir=dir, recursive=True) if not os.path.isdir(os.path.join(dir, f))))
                combinations = names_or_ids_to_combinations(os.listdir(dir))
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
        return {"num_classes" : ncls, "masks" : list(masks)}, {"classes" : cls, "cls2idx" : cls2idx, "idx2cls" : idx2cls, "combinations" : combinations}

    @staticmethod
    def build_model(*args, cls=HierarchicalClassifier, **kwargs):
        return BaseBuilder.build_model(*args, cls=cls, **kwargs)

    @staticmethod
    def build_dataloader(
            data_index : Optional[str],
            input_dir : str,
            classes : list[str],
            cls2idx : dict[str, int],
            preprocess : Callable[[torch.Tensor], torch.Tensor],
            batch_size : int,
            device : torch.device,
            dtype = torch.dtype,
            resize_size : Optional[int]=None,
            train_proportion : float=0.9,
            idx2cls : Optional[dict[int, str]]=None,
            num_workers : Optional[int]=None,
            combinations : Optional[list[tuple[str, str, str]]]=None,
            subsample : Optional[int]=None
            # hierarchy : Optional[list[list[list[int]]]]=None,
            # path2cls2idx_builder : Optional[Callable[[Any], Callable[[str], torch.Tensor]]]=hierarchical_base_path2cls2idx_builder,
            # path2cls2idx_builder_kwargs : dict[str, Any]={}
        ):
        # Prepare datasets/dataloaders
        if data_index is None:
            # if path2cls2idx_builder is None:
            #     raise RuntimeError(f'If no data index is passed a function factory (higher order function) that generates a function which computes the class/label from the path must be passed.')
            # path2cls2idx = path2cls2idx_builder(cls2idx=cls2idx, **path2cls2idx_builder_kwargs)
            cls2comb = {comb[0] : comb for comb in combinations}
            all_files = [path for f in glob.glob("**", root_dir=input_dir, recursive=True) if not os.path.isdir(path := os.path.join(input_dir, f))]
            data = {
                "path" : [],
                "class" : [],
                "split" : []
            }
            for path in all_files:
                data["path"].append(path)
                data["class"].append([cls2idx[lvl][cls] for lvl, cls in enumerate(cls2comb[resolve_name_or_id(os.path.basename(os.path.dirname(path)))["species"][0]])])
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

        loader = ImageClassLoader(torch.tensor, lambda x : x, resize_size=resize_size, preprocessor=lambda x : x, dtype=dtype, device=torch.device("cpu"))

        if subsample is None or (isinstance(subsample, int) and subsample <= 1):
            train_labels = train_image_data["class"]
            train_dataset = loader(list(zip(train_image_data["path"], train_image_data["class"])))
            val_dataset = loader(list(zip(val_image_data["path"], val_image_data["class"])))
        else:
            train_labels = train_image_data["class"][::subsample]
            train_dataset = loader(list(zip(train_image_data["path"][::subsample], train_image_data["class"][::subsample])))
            val_dataset = loader(list(zip(val_image_data["path"][::subsample], val_image_data["class"][::subsample])))

        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

        if num_workers is None:
            num_workers = int(os.cpu_count() * 3 / 4)
            num_workers -= num_workers % 2
            num_workers = max(0, num_workers)

        pin_memory = False # True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=multi_level_collate,
            drop_last=True, # Ensures compatibility with batch normalization
            num_workers=num_workers,
            pin_memory=pin_memory,
            pin_memory_device=str(device),
            persistent_workers=pin_memory
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            sampler=val_sampler, 
            collate_fn=multi_level_collate,
            num_workers=num_workers, # min(2, num_workers), 
            pin_memory=False,
            pin_memory_device="",
            persistent_workers=False
        )

        return train_labels, train_loader, val_loader
    
    @staticmethod
    def build_criterion(
            *args, 
            weighted : bool=False,
            labels : Optional[np.ndarray]=None, 
            num_classes : Optional[list[int]], 
            device : Optional[torch.types.Device]=None,
            dtype : Optional[torch.dtype]=None,
            **kwargs
        ):
        if not weighted or labels is None or num_classes is None:
            return MultiLevelCrossEntropyLoss(*args, **kwargs)
        class_weights = []
        for lvl, ncls in enumerate(num_classes):
            counts = torch.ones((ncls, ))
            for cls_idx in labels:
                counts[cls_idx[lvl]] += 1
            # weights = torch.log(counts)
            weights = 1 / (counts + counts.mean())
            weights /= weights.mean()
            class_weights.append(weights)
        return MultiLevelWeightedCrossEntropyLoss(
            *args, 
            class_weights=class_weights, 
            device=device,
            dtype=dtype,
            **kwargs
        )

class MultiLevelResultCollector(BaseResultCollector):
    def __init__(self, lvl : int, cls2cls : Optional[dict[str, str]]=None, *args, **kwargs):
        self.level = lvl
        self.cls2cls = cls2cls
        super().__init__(*args, **kwargs)

    def collect(self, paths, *args, labels=None, **kwargs):
        if labels is not None:
            return super().collect(paths, *args, **kwargs, labels=labels)
        if self._training_format:
            leaf_labels = [os.path.basename(os.path.dirname(path)) for path in paths]
            labels = [self.cls2cls[ll] for ll in leaf_labels]
            return super().collect(paths, *args, **kwargs, labels=labels)
        return super().collect(paths, *args, **kwargs)

    def eval_label_fn(self, data : dict, prefix : str="", *args, **kwargs):
        if len(prefix) > 0 and not prefix.endswith("_"):
            prefix = prefix + "_"
        prefix = f'{prefix}level{self.level}_'
        return super().eval_label_fn(data=data, prefix=prefix, *args, **kwargs)

class HierarchicalResultCollector:
    def __init__(
            self, 
            levels : int, 
            idx2cls : dict[int, dict[int, str]], 
            combinations : list[tuple[int, int, int]], 
            *args, 
            **kwargs
        ):
        self.levels = levels
        self.idx2cls = idx2cls
        self.cls2cls = dict()
        for comb in combinations:
            for lvl, e in enumerate(comb):
                if lvl not in self.cls2cls:
                    self.cls2cls[lvl] = dict()
                self.cls2cls[lvl][comb[0]] = e
        self.collectors = tuple([
            MultiLevelResultCollector(lvl, idx2cls=self.idx2cls[lvl], cls2cls=self.cls2cls[lvl], *args, **kwargs) 
            for lvl in range(self.levels)
        ])

    def evaluate(self, outdir : Optional[str]=None, prefix : str="", level : Optional[Union[int, list[int]]]=None):
        if level is None:
            level = list(range(self.levels))
        if isinstance(level, int):
            level = [level]
        # results = {lvl : result for lvl in level if (result := self.collectors[lvl].evaluate()) is not None}
        results = dict()
        for lvl in level:
            result = self.collectors[lvl].evaluate(outdir=outdir, prefix=prefix)
            if result is not None:
                results[lvl] = result
        
        do_save = isinstance(outdir, str)
        if do_save and not os.path.isdir(outdir):
            raise IOError(f'Specified output directory (`{outdir}`) does not exist.')
        if results:
            if do_save:
                with open(os.path.join(outdir, f'{prefix}eval_results.json'), "w") as f:
                    json.dump(results, f)
            return results

    def collect(self, paths : list[str], predictions : list[torch.Tensor], level : Optional[Union[int, list[int]]]=None, **kwargs):
        if level is None:
            level = list(range(self.levels))
        if isinstance(level, int):
            level = [level]
        for lvl in level:
            self.collectors[lvl].collect(paths, predictions[lvl], **kwargs)

    @property
    def data(self):
        return {lvl : self.collectors[lvl].data for lvl in range(self.levels)}
