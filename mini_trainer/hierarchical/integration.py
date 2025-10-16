import json
import os
from collections import OrderedDict
from collections.abc import Callable
from typing import Iterable

import torch

from mini_trainer.builders import BaseBuilder
from mini_trainer.hierarchical.gbif import (create_taxonomy,
                                            labels_from_taxonomy)
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.hierarchical.model import HierarchicalClassifier
from mini_trainer.utils.data import find_images
from mini_trainer.utils.logging import BaseResultCollector
from mini_trainer.utils.parquet import parquet_to_class_spec_hierarchical


def linnean_labels_from_directory(dir : str, levels="family", **kwargs):
    images = find_images(dir)
    dirnames = set([os.path.split(os.path.dirname(im))[1] for im in images])
    taxonomy = create_taxonomy(dirnames, levels=levels)
    return labels_from_taxonomy(taxonomy)

def default_labels_from_directory_structure(dir : str, **kwargs):
    images = find_images(dir)
    image_dirs = list(sorted(set([os.path.dirname(os.path.relpath(p, dir)) for p in images])))
    labels = OrderedDict([(d[-1], tuple(d.split(os.sep)[::-1])) for d in image_dirs])
    return labels

def parse_class_spec(
        path : str | None=None, 
        dir : str | None=None,
        levels : int | None=None,
        label_fn : Callable[[str], OrderedDict[str, tuple[str, ...]]]=default_labels_from_directory_structure,
        **kwargs
    ) -> dict[str, dict[str, dict[str, int]] | OrderedDict[str, tuple[str, ...]] | int]:
    """
    Construct class specification: 
    * class index (label string to index mapping)
    * hierarchical labels (tuple of label strings leaf->root)
    * number of (leaf) classes
    from a precalculated class specification or a directory structure.

    If constructed from a directory structure, the hierarchy is constructed based on the names
    and structure of the directories containing the training images.
    
    By default it assumed that labels can be parsed from the image path like:
        ```
        image_path = <"[dir]/[root_label]>/[...]/[leaf_label]/[image_filename]">
        label = [<"leaf_label">, ..., <"root_label">]
        labels = {<"[root_label]>/[...]/[leaf_label]"> : [<"leaf_label">, ..., <"root_label">] for image in images}
        ```
    However, this behaviour can be modified by passing a function to `label_fn` that takes the root
    directory containing all training images (and no other images), and computes an ordered dictionary
    of all labels for all valid images in the directory, where the key should be the parent directory of images
    with a given label.
    The labels should be sorted first by the root label and last by the leaf label.

    Args:
        path: Path to a precomputed class specification if it exists, otherwise one will be computed.
            If path is not None, but doesn't exist yet, the computed class specification
            will be stored in path for later use.
        levels: If an integer, the hierarchy is truncated to the number of levels specified.
            Otherwise all levels computed from ``label_fn`` are used.
        dir: Root directory containing all training images (and no other images).
        label_fn: A function which computes an ordered dictionary of labels for all images in ``dir``, 
            where the key should be the name of the directory containing all images which match a label.
        **kwargs: Additional arguments passed to ``label_fn``.
    
    Returns:
        (class specification): A dictionary containing information 
            used for constructing models and dataloaders. Structure:
            * "cls2idx": [dict[str, dict[str, int]]]
                * [str] <"hierarchy level">:
                    * [str] <"leaf label">: [int] <"leaf class index">
            * "labels": [OrderedDict[str, tuple[str, ...]]]
                * [str] (<"label 1 image directory">) : [tuple[str, ...]] (<"leaf 1 label">, ..., <"root 1 label">)
            * "num_classes": [int] <number of leaf classes>
    """
    if isinstance(levels, int):
        assert levels > 0
    else:
        assert levels is None
    if path is None or not os.path.exists(path):
        if dir is None or not os.path.isdir(dir):
            if isinstance(dir, str) and dir.endswith(".parquet"):
                # TODO: For now we will just assume that there are three levels 
                # if not specified with parquet, but this should be determined 
                # automatically as it is in the other code branch!
                if levels is None:
                    levels = 3
                retval = parquet_to_class_spec_hierarchical(dir, levels=levels)
            else:
                raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not \'{dir}\'.')
        else:
            labels = label_fn(dir, **kwargs)
            if levels is not None:
                for lab in labels.keys():
                    labels[lab] = labels[lab][:levels]
            nlvl = set([len(l) for l in labels.values()])
            if len(nlvl) != 1:
                raise RuntimeError('Varying hierarchy levels found in image directory structure:', list(sorted(nlvl)))
            nlvl = list(nlvl)[0]
            cls2idx : dict[int, dict[str, int]] = {str(lvl) : dict() for lvl in range(nlvl)}
            classes = {str(lvl) : set() for lvl in range(nlvl)}
            for lab in labels.values():
                for lvl, cls in enumerate(lab):
                    if cls in classes[str(lvl)]:
                        continue
                    classes[str(lvl)].add(cls)
                    cls2idx[str(lvl)][cls] = len(classes[str(lvl)]) - 1
            retval = {"cls2idx" : cls2idx, "labels" : labels, "num_classes" : len(labels)}
        if path is not None:
            with open(path, "w") as f:
                json.dump(retval, f)
        else:
            return retval
    with open(path, "rb") as f:
        retval = json.load(f)
        retval["labels"] = OrderedDict([(k, v) for k, v in retval["labels"].items()])
    if levels:
        retval["cls2idx"] = {str(lvl) : retval["cls2idx"][str(lvl)] for lvl in range(levels)}
        for lab in retval["labels"].keys():
            retval["labels"][lab] = retval["labels"][lab][:levels]
    return retval
    
def sparse_masks_from_labels(
        labels : OrderedDict[str, tuple[str, ...]], 
        cls2idx : dict[str, dict[str, int]]
    ):
    """
    Compute 'sparse masks' from labels (e.g. [species, genus, family]) and class indices.

    A sparse mask is an integer vector (1D tensor) with length equal to the number of classes
    at some level (e.g. number of species) that maps each class to it's parent class 
    (e.g. a species to a genus), encoded such that the value in the mask at the index of a class
    is the index of it's parent:
        ```
        mask[child_idx] = parent_idx
        ```

    Args:
        labels: Ordered dictionary of hierarchical labels (tuple of label strings leaf->root).
        cls2idx: Dictionary of dictionaries, keys to the outer dictionary are hierarchy levels (integer), 
            while the nested dictionaries are class label to index mappings for each level in the hierarchy.

    Returns:
        List of sparse masks for levels `{0, ..., N-2}` where `N` is the 
            number of layers in the hierarchy (e.g. 3 if [species, genus, family]).
    """
    ## Function logic
    nlvl = len(cls2idx)
    # Initialize masks with "empty" values (-1)
    masks = [[-1 for _ in range(len(cls2idx[str(lvl)]))] for lvl in range(nlvl-1)] 
    for lab in labels.values():
        idx = [cls2idx[str(lvl)][cls] for lvl, cls in enumerate(lab)]
        for mask_i, (child, parent) in enumerate(zip(idx, idx[1:])):
            if masks[mask_i][child] not in [-1, parent]:
                raise ValueError(
                    f'Conflicting labels detected at level {mask_i} class {child} '
                    f'which had parent {masks[mask_i][child]}, now found {parent}!'
                )
            masks[mask_i][child] = parent

    ## Output checking
    # Check that masks contain no empty values (-1)
    invalid = []
    for mask_i, mask in enumerate(masks):
        for element_i, element in enumerate(mask):
            if element == -1:
                invalid.append((mask_i, element_i))
    if len(invalid) > 0:
        err_msg = (
            'Unable to construct sparse masks (child-parent mappings) from labels and class index.\n'
            f'Found {len(invalid)} missing elements at:\n'
            '| mask | element |\n'
            '------------------\n'
        ) + '\n'.join([f'|{mask_i:^6}|{element_i:^9}|' for mask_i, element_i in invalid])
        raise ValueError(err_msg)
    # Check that all classes in last layer class index are used
    if (missing := set(cls2idx[str(nlvl-1)].values()) - set(masks[-1])):
        err_msg = f'Found {len(missing)} unused classes in top level: [{", ".join(map(str, missing))}]'
        raise ValueError(err_msg)
    
    ## Function return
    # Return masks converted to long tensors
    return [torch.tensor(mask, dtype=torch.long) for mask in masks]


class HierarchicalBuilder(BaseBuilder):
    @staticmethod
    def class_spec(
            path : str | None=None,
            dir : str | None=None,
            levels : int | None=None,
            species : bool=True,
            *args, 
            **kwargs
        ):
        """
        Returns:
            (extra_model_kwargs, extra_dataloader_kwargs): Extra keyword arguments for the model and dataloader building functions.
        """
        if species:
            if "label_fn" in kwargs:
                raise ValueError(f'`label_fn` passed to `BaseBuilder.spec_model_dataloader` when `{species=})`')
            kwargs["label_fn"] = linnean_labels_from_directory
        return parse_class_spec(path=path, dir=dir, levels=levels, **kwargs)
        

    @staticmethod
    def build_model(
            cls2idx : dict[int, dict[str, int]], 
            labels : list[tuple[str, ...]], 
            *args, 
            cls=HierarchicalClassifier, 
            **kwargs
        ):
        sparse_masks = sparse_masks_from_labels(labels, cls2idx)
        return BaseBuilder.build_model(*args, cls=cls, sparse_masks=sparse_masks, **kwargs)
    
    @staticmethod
    def build_dataloader(*args, **kwargs):
        if "multilabel" in kwargs:
            raise NotImplementedError("Do not supply `multilabel` to `build_dataloader` manually.")
        return BaseBuilder.build_dataloader(*args, multilabel=True, **kwargs)
    
    @staticmethod
    def build_criterion(
            *args, 
            weighted : bool=False,
            labels : Iterable[int] | None=None, 
            num_classes : list[int] | None=None, 
            device : torch.types.Device | None=None,
            dtype : torch.dtype | None=None,
            **kwargs
        ):
        if not weighted or labels is None or num_classes is None:
            return MultiLevelWeightedCrossEntropyLoss(*args, device=device, dtype=dtype, **kwargs)
        class_weights = []
        for lvl, ncls in enumerate(num_classes):
            counts = torch.ones((ncls, ))
            for cls_idx in labels:
                counts[cls_idx[lvl]] += 1
            # weights = torch.log(counts)
            weights = 1 / counts
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
    def __init__(self, lvl : int, cls2cls : dict[str, str] | None=None, *args, **kwargs):
        self.level = lvl
        self.cls2cls = cls2cls
        super().__init__(*args, **kwargs)

    def collect(self, paths, *args, labels=None, **kwargs):
        if labels is not None:
            return super().collect(paths, *args, **kwargs, labels=labels)
        if self._training_format:
            leaf_labels = [os.pathname(os.path.dirname(path)) for path in paths]
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

    def evaluate(self, outdir : str | None=None, prefix : str="", level : int | list[int] | None=None):
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
            raise OSError(f'Specified output directory (`{outdir}`) does not exist.')
        if results:
            if do_save:
                with open(os.path.join(outdir, f'{prefix}eval_results.json'), "w") as f:
                    json.dump(results, f)
            return results

    def collect(self, paths : list[str], predictions : list[torch.Tensor], level : int | list[int] | None=None, **kwargs):
        if level is None:
            level = list(range(self.levels))
        if isinstance(level, int):
            level = [level]
        for lvl in level:
            self.collectors[lvl].collect(paths, predictions[lvl], **kwargs)

    @property
    def data(self):
        return {lvl : self.collectors[lvl].data for lvl in range(self.levels)}
