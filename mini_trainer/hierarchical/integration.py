import json
import os
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from functools import lru_cache
from itertools import repeat
from typing import cast

import torch
from torch import nn

from mini_trainer.builders import BaseBuilder
from mini_trainer.classifier import classification_module
from mini_trainer.hierarchical.gbif import cls2idx_from_labels, create_taxonomy, id_to_name, labels_from_taxonomy
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.hierarchical.model import HierarchicalClassifier, HierarchicalPrediction
from mini_trainer.utils import write_csv_from_dict
from mini_trainer.utils.data import find_images
from mini_trainer.utils.logging import BaseResultCollector
from mini_trainer.utils.parquet import parquet_to_class_spec_hierarchical
from mini_trainer.utils.plot import named_confusion_matrix


def _freeze(d: dict[str, dict[str, int]]):
    return tuple((k, tuple(sorted(v.items()))) for k, v in sorted(d.items()))


@lru_cache
def _cls2idx_to_names(frozen):
    ids = [id_ for _, inner in frozen for id_, _ in inner]
    id2name = dict(zip(ids, id_to_name(ids)))
    return {lvl: {id2name[i]: idx for i, idx in inner} for lvl, inner in frozen}


def cls2idx_to_names(cls2idx: dict[str, dict[str, int]]):
    return _cls2idx_to_names(_freeze(cls2idx))


def linnean_labels_from_directory(dir : str, levels="family", **kwargs): # noqa: D103
    images = find_images(dir)
    dirnames = set([os.path.split(os.path.dirname(im))[1] for im in images])
    taxonomy = create_taxonomy(dirnames, levels=levels)
    return labels_from_taxonomy(taxonomy)


def default_labels_from_directory_structure(dir : str, **kwargs): # noqa: D103
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
    """Construct class specification:
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
                raise TypeError(
                    f'If `path` is not the path to a valid file, `dir` must be a valid directory, not "{dir}".'
                )
        else:
            labels = label_fn(dir, **kwargs)
            if levels is not None:
                for lab in labels.keys():
                    labels[lab] = labels[lab][:levels]
            cls2idx = cls2idx_from_labels(labels)
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
    """Compute 'sparse masks' from labels (e.g. [species, genus, family]) and class indices.

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
    nlvl = len(cls2idx)
    # Initialize masks with "empty" values (-1)
    masks = [[-1 for _ in range(len(cls2idx[str(lvl)]))] for lvl in range(nlvl - 1)] 
    for lab in labels.values():
        idx = [cls2idx[str(lvl)][cls] for lvl, cls in enumerate(lab)]
        for mask_i, (child, parent) in enumerate(zip(idx, idx[1:])):
            if masks[mask_i][child] not in [-1, parent]:
                raise ValueError(
                    f'Conflicting labels detected at level {mask_i} class {child} '
                    f'which had parent {masks[mask_i][child]}, now found {parent}!'
                )
            masks[mask_i][child] = parent

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
    if masks and (missing := set(cls2idx[str(nlvl - 1)].values()) - set(masks[-1])):
        err_msg = f'Found {len(missing)} unused classes in top level: [{", ".join(map(str, missing))}]'
        raise ValueError(err_msg)
    
    # Return masks converted to long tensors
    return [torch.tensor(mask, dtype=torch.long) for mask in masks]


class HierarchicalBuilder(BaseBuilder): # noqa: D101
    @staticmethod
    def class_spec(
            path : str | None=None,
            dir : str | None=None,
            levels : int | None=None,
            species : bool=True,
            *args, 
            **kwargs
        ):
        """TODO.

        Returns:
            (extra_model_kwargs, extra_dataloader_kwargs): 
                Extra keyword arguments for the model and dataloader building functions.
        """
        if species:
            if "label_fn" in kwargs:
                raise ValueError(f'`label_fn` passed to `BaseBuilder.spec_model_dataloader` when `{species=})`')
            kwargs["label_fn"] = linnean_labels_from_directory
        return parse_class_spec(path=path, dir=dir, levels=levels, **kwargs)
        
    @staticmethod
    def build_model(
            cls2idx : dict[int, dict[str, int]] | None=None, 
            labels : list[tuple[str, ...]] | None=None, 
            *args, 
            cls=HierarchicalClassifier, 
            **kwargs
        ):
        if labels is not None and cls2idx is not None:
            sparse_masks = sparse_masks_from_labels(labels, cls2idx)
        else:
            sparse_masks = None
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


class HierarchicalResultCollector(BaseResultCollector): # noqa: D101 TODO
    def __init__( # noqa: D107
            self, 
            model : nn.Module | None=None,
            idx2cls : dict[str, dict[int, str]] | None=None,
            cls2idx : dict[str, dict[str, int]] | None=None,
            scientific_names : bool=True,
            *args, 
            **kwargs
        ):
        if model is not None:
            model_metadata = classification_module(model).metadata
            cls2idx = model_metadata.get("cls2idx", None)
            if cls2idx is not None:
                idx2cls = None
        self.scientific_names = scientific_names
        self._sn_cache = defaultdict(str)
        if cls2idx is not None and self.scientific_names:
            cls2idx = cls2idx_to_names(cls2idx)
            idx2cls = None
        if idx2cls is None and cls2idx is not None:
            idx2cls = {lvl : {v : k for k, v in _cls2idx.items()} for lvl, _cls2idx in cls2idx.items()}
        if cls2idx is None and idx2cls is not None:
            cls2idx = {lvl : {v : k for k, v in _idx2cls.items()} for lvl, _idx2cls in idx2cls.items()}
        super().__init__(idx2cls=idx2cls, cls2idx=cls2idx, *args, **kwargs)
        self._levels = None

    def _collect_base_attributes(
            self, 
            paths : list[str], 
            predictions : list[torch.Tensor] | list[HierarchicalPrediction], 
            labels : list[tuple[str, ...]] | None=None
        ):
        """Override in subclasses!
        """
        self.paths.extend(paths)
        if isinstance(predictions, list) and all(isinstance(p, HierarchicalPrediction) for p in predictions):
            predictions = cast(list[HierarchicalPrediction], predictions)
            if self._levels is None and predictions:
                self._levels = len(predictions[0][0].label)
            predictions, confidences, indices = zip(*[(pred[0].label, pred[0].confidence, pred[0].index) for pred in predictions])
            self.preds.extend([[self.idx2cls[str(lvl)][i] for lvl, i in enumerate(idxs)] for idxs in indices])
            self.confs.extend(confidences)
        else:
            if self._levels is None:
                self._levels = len(predictions)
            self.preds.extend(list(zip(*[
                map(self.idx2cls[str(lvl)].get, p.argmax(1).tolist())
                for lvl, p in enumerate(predictions)
            ])))
            self.confs.extend(list(zip(*[p.softmax(1).max(1).values.tolist() for p in predictions])))
        if labels is not None:
            if self.scientific_names:
                labels = [
                    tuple(
                        self._sn_cache[e] if e in self._sn_cache else self._sn_cache.setdefault(e, id_to_name(e)) 
                        for e in label
                    ) for label in labels
                ]
            self.labels.extend(labels)

    def eval_label_fn(
            self,
            data : dict,
            outdir : str | None,
            save : bool,
            prefix : str="",
            plot_conf_mat : bool=False,
            **kwargs
        ):
        if kwargs:
            raise RuntimeError(
                f'Unknown arguments ([{", ".join(kwargs)}]) passed. '
                'Perhaps you forgot to implement the intended `eval_label_fn` in your subclass.'
            )
        if save and not isinstance(outdir, str):
            raise RuntimeError(
                'Attempted to save evaluated results against labels without specifying an output directory.'
            )
        if self._levels is None:
            raise RuntimeError(
                'Hierarchical result collector was unable to detect number of levels in the class hierarchy!'
            )
        return {
            level : named_confusion_matrix(
                results={k : v[level] if k in ["preds", "confs", "labels"] else v for k, v in data.items()}, 
                cls2idx=self.cls2idx[str(level)],
                verbose=self.verbose, 
                plot_conf_mat=(
                    plot_conf_mat and 
                    save and 
                    os.path.join(outdir, f"{prefix}confusion_matrix_level{level}.png")
                )
            ) for level in range(self._levels)
        }
    
    def save_mini_metric_csv(self, dst : str, threshold : float=0.0):
        SCHEMA = dict((
            ("instance_id", int),
            ("filename", str),
            ("level", int),
            ("label", str),
            ("prediction", str),
            ("confidence", float),
            ("threshold", float),
            ("known_label", int),
            ("prediction_made", int),
            ("correct", int)
        ))
        data = {
            k : list() for k in SCHEMA
        }
        labels = self.labels or repeat(tuple(["-1"] * self._levels))
        for i, (path, preds, labs, confs) in enumerate(zip(self.paths, self.preds, labels, self.confs)):
            for level in range(self._levels):
                label, pred, conf = labs[level], preds[level], confs[level]
                do_predict = int(conf >= threshold)
                row = {
                    "instance_id" : i,
                    "filename" : path,
                    "level" : level,
                    "label" : label,
                    "prediction" : pred,
                    "confidence" : conf,
                    "threshold" : float(threshold),
                    "known_label" : int(label in self.cls2idx[str(level)]),
                    "prediction_made" : do_predict,
                    "correct" : do_predict if do_predict == 0 else 1 if pred == label else -1
                }
                for k, v in row.items():
                    assert isinstance(v, SCHEMA[k]), f'Invalid data type in {k}, found {v}, but expected a {SCHEMA[k]}'
                    data[k].append(v)
        write_csv_from_dict(data, dst)