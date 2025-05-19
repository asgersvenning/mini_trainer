import json
import os
from collections import defaultdict
from math import ceil
from random import sample
from typing import Any, Callable, Optional

import torch

DEFAULT_HIERARCHY_LEVELS = ("species", "genus", "family")

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
    spl = sample([lab for prop, lab in zip(split, split_labels) for _ in range(ceil(n * prop))], n)
    data = {k : v for k, v in zip(["path", "class", "split"],[paths, cls, spl])}
    if outpath:
        if os.path.exists(outpath):
            os.remove(outpath)
        with open(outpath, "w") as f:
            json.dump(data, f)
    else:
        return data
    # train_image_data = {k : v[data["split"] == np.array("train")] for k, v in data.items()}
    # val_image_data = {k : v[data["split"] == np.array("validation")] for k, v in data.items()}
    # return train_image_data, val_image_data