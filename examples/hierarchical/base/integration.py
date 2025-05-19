import glob
import json
import os
import random
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from hierarchical.base.loss import MultiLevelCrossEntropyLoss
from hierarchical.base.model import HierarchicalClassifier
from hierarchical.base.utils import create_hierarchy, mask_hierarchy, leaf_to_parents
from torch import nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from mini_trainer.builders import BaseBuilder
from mini_trainer.utils.data import get_image_data
from mini_trainer.utils.io import ImageClassLoader
from mini_trainer.utils.logging import BaseResultCollector


def hierarchical_base_path2cls2idx_builder(cls2idx):
    def path2cls2idx(path, cls2idx=cls2idx, nlvl=len(cls2idx)):
        return torch.tensor(list(reversed([cls2idx[lvl][cls] for lvl, cls in enumerate(path.split(os.sep)[:-1][-nlvl:])]))).long()
    return path2cls2idx

def hierarchical_path2cls2idx_w_classindex(*args, class_index : str, **kwargs):
    with open(class_index, "r") as f:
        class_index = json.load(f)
    cls2idx = class_index["cls2idx"]
    combinations = class_index["combinations"]
    cls2idxs = {comb[0] : [cls2idx[str(lvl)][c] for lvl, c in enumerate(comb)] for comb in combinations}
    def path2cls2idx(path):
        return torch.tensor(cls2idxs[os.path.basename(os.path.dirname(path))])
    return path2cls2idx

def multi_level_collate(batch):
    return tuple(torch.stack(v) for v in zip(*batch))

def hierarchical_class_index_to_standard(path : str):
    with open(path, "r") as f:
        data = json.load(f)
    return data["cls2idx"]["0"]


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
        return {"num_classes" : ncls[0], "masks" : list(masks)}, {"classes" : cls, "cls2idx" : cls2idx, "idx2cls" : idx2cls, "hierarchy" : hierarchy}

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
            hierarchy : Optional[list[list[list[int]]]]=None,
            path2cls2idx_builder : Optional[Callable[[Any], Callable[[str], torch.Tensor]]]=hierarchical_base_path2cls2idx_builder,
            path2cls2idx_builder_kwargs : dict[str, Any]={}
        ):
        # Prepare datasets/dataloaders
        if data_index is None:
            if path2cls2idx_builder is None:
                raise RuntimeError(f'If no data index is passed a function factory (higher order function) that generates a function which computes the class/label from the path must be passed.')
            path2cls2idx = path2cls2idx_builder(cls2idx=cls2idx, **path2cls2idx_builder_kwargs)
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

        loader = ImageClassLoader(torch.tensor, lambda x : x, resize_size=resize_size, preprocessor=lambda x : x, dtype=dtype, device=torch.device("cpu"))
        train_dataset = loader(list(zip(train_image_data["path"], train_image_data["class"])))
        val_dataset = loader(list(zip(val_image_data["path"], val_image_data["class"])))

        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

        if num_workers is None:
            num_workers = int(os.cpu_count() / 2)
            num_workers -= num_workers % 2
            num_workers = max(0, num_workers)

        pin_memory = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=multi_level_collate,
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
            num_workers=min(2, num_workers), 
            pin_memory=False,
            pin_memory_device="",
            persistent_workers=False
        )

        return train_loader, val_loader
    
    @staticmethod
    def build_criterion(*args, **kwargs):
        return MultiLevelCrossEntropyLoss(*args, **kwargs)

class MultiLevelResultCollector(BaseResultCollector):
    def __init__(self, lvl : int, cls2cls : Optional[dict[str, str]]=None, *args, **kwargs):
        self.level = lvl
        self.cls2cls = cls2cls
        super().__init__(*args, **kwargs)

    def eval_label_fn(self, data : dict, prefix : str="", *args, **kwargs):
        if self.cls2cls is not None:
            data["labels"] = [self.cls2cls[cls] for cls in data["labels"]]
        if len(prefix) > 0 and not prefix.endswith("_"):
            prefix = prefix + "_"
        prefix = f'{prefix}level{self.level}_'
        return super().eval_label_fn(data=data, prefix=prefix, *args, **kwargs)

class HierarchicalResultCollector:
    def __init__(self, levels : int, idx2cls : dict[int, dict[int, str]], hierarchy : list[list[list[int]]], *args, **kwargs):
        self.levels = levels
        self.idx2cls = idx2cls
        l2p = leaf_to_parents(hierarchy)
        self.idx2idx = [{k : k for k in l2p[0].keys()}] + l2p
        self.cls2cls = [{idx2cls[0][k] : idx2cls[lvl][v] for k, v in m.items()} for lvl, m in enumerate(self.idx2idx)]
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

    def collect(self, paths : list[str], predictions : list[torch.Tensor], level : Optional[Union[int, list[int]]]=None):
        if level is None:
            level = list(range(self.levels))
        if isinstance(level, int):
            level = [level]
        for lvl in level:
            self.collectors[lvl].collect(paths, predictions[lvl])

    @property
    def data(self):
        return {lvl : self.collectors[lvl].data for lvl in range(self.levels)}
