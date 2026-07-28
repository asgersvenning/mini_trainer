import json
import os
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from functools import lru_cache
from itertools import repeat
from typing import Concatenate, cast

import numpy as np
import torch
from torch import nn

from mini_trainer.builders import BaseBuilder
from mini_trainer.data import find_images
from mini_trainer.integrations import (
    cls2idx_from_labels,
    create_taxonomy,
    id_to_name,
    labels_from_taxonomy,
    parquet_to_class_spec_hierarchical,
)
from mini_trainer.logging import BaseResultCollector
from mini_trainer.training import EMLACrossEntropy, named_confusion_matrix
from mini_trainer.visualization import plot_heatmap

from .loss import MultiLevelWeightedCrossEntropyLoss
from .model import HierarchicalClassifier, HierarchicalPrediction


def _freeze(d: dict[str, dict[str, int]]):
    return tuple((k, tuple(sorted(v.items()))) for k, v in sorted(d.items()))


@lru_cache
def _cls2idx_to_names(frozen):
    ids = [id_ for _, inner in frozen for id_, _ in inner]
    id2name = dict(zip(ids, id_to_name(ids)))
    return {lvl: {id2name[i]: idx for i, idx in inner} for lvl, inner in frozen}


def cls2idx_to_names(cls2idx: dict[str, dict[str, int]]):
    return _cls2idx_to_names(_freeze(cls2idx))


def linnean_labels_from_directory(dir: str, levels: str | int | list[str | int] | None = None, **kwargs):  # noqa: D103
    images = find_images(dir)
    dirnames = set([os.path.split(os.path.dirname(im))[1] for im in images])
    taxonomy = create_taxonomy(dirnames, levels=levels)
    taxonomy = OrderedDict(sorted(taxonomy.items(), key=lambda kv: tuple(vv[0] for vv in kv[1].values())[::-1]))
    return labels_from_taxonomy(taxonomy)


def default_labels_from_directory_structure(dir: str, **kwargs):  # noqa: D103
    images = find_images(dir)
    image_dirs = list(sorted(set([os.path.dirname(os.path.relpath(p, dir)) for p in images])))
    labels = OrderedDict([(d[-1], tuple(d.split(os.sep)[::-1])) for d in image_dirs])
    return labels


def parse_class_spec(
    path: str | None = None,
    dir: str | None = None,
    levels: int | None = None,
    label_fn: Callable[Concatenate[str, ...], OrderedDict[str, tuple[str, ...]]] = default_labels_from_directory_structure,
    **kwargs,
) -> dict[str, dict[str, dict[str, int]] | OrderedDict[str, tuple[str, ...]] | list[int]]:
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
                raise TypeError(f'If `path` is not the path to a valid file, `dir` must be a valid directory, not "{dir}".')
        else:
            labels = label_fn(dir, levels=levels, **kwargs)
            if levels is not None:
                for lab in labels.keys():
                    labels[lab] = labels[lab][:levels]
            cls2idx = cls2idx_from_labels(labels)
            num_classes = [len(cls2idx[str(i)]) for i in range(len(cls2idx))]
            retval: dict[str, dict[str, dict[str, int]] | OrderedDict[str, tuple[str, ...]] | list[int]] = {
                "cls2idx": cls2idx,
                "labels": labels,
                "num_classes": num_classes,
            }
        if path is not None:
            with open(path, "w") as f:
                json.dump(retval, f)
        else:
            return retval
    with open(path, "rb") as f:
        data = json.load(f)
    cls2idx = cast(dict[str, dict[str, int]], data["cls2idx"])
    labels = cast(dict[str, tuple[str, ...]], data["labels"])
    labels = OrderedDict([(k, v) for k, v in sorted(labels.items(), key=lambda kv: cls2idx["0"][kv[1][0]])])
    num_classes = cast(list[int], data["num_classes"])
    if levels:
        cls2idx = {str(lvl): cls2idx[str(lvl)] for lvl in range(levels)}
        for lab in labels.keys():
            labels[lab] = labels[lab][:levels]
    retval: dict[str, dict[str, dict[str, int]] | OrderedDict[str, tuple[str, ...]] | list[int]] = {
        "cls2idx": cls2idx,
        "labels": labels,
        "num_classes": num_classes,
    }
    return retval


def sparse_masks_from_labels(labels: OrderedDict[str, tuple[str, ...]], cls2idx: dict[int | str, dict[str, int]]):
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
    cls2idx = {str(k): v for k, v in cls2idx.items()}
    nlvl = len(cls2idx)
    # Initialize masks with "empty" values (-1)
    masks = [[-1 for _ in range(len(cls2idx[str(lvl)]))] for lvl in range(nlvl - 1)]
    for lab in labels.values():
        idx = [cls2idx[str(lvl)][cls] for lvl, cls in enumerate(lab)]
        for mask_i, (child, parent) in enumerate(zip(idx, idx[1:])):
            if masks[mask_i][child] not in [-1, parent]:
                raise ValueError(
                    f"Conflicting labels detected at level {mask_i} class {child} "
                    f"which had parent {masks[mask_i][child]}, now found {parent}!"
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
            "Unable to construct sparse masks (child-parent mappings) from labels and class index.\n"
            f"Found {len(invalid)} missing elements at:\n"
            "| mask | element |\n"
            "------------------\n"
        ) + "\n".join([f"|{mask_i:^6}|{element_i:^9}|" for mask_i, element_i in invalid])
        raise ValueError(err_msg)
    # Check that all classes in last layer class index are used
    if masks and (missing := set(cls2idx[str(nlvl - 1)].values()) - set(masks[-1])):
        err_msg = f"Found {len(missing)} unused classes in top level: [{', '.join(map(str, missing))}]"
        raise ValueError(err_msg)

    # Return masks converted to long tensors
    return [torch.tensor(mask, dtype=torch.long) for mask in masks]


class HierarchicalBuilder(BaseBuilder):  # noqa: D101
    @staticmethod
    def build_class_spec(*args, path: str | None = None, dir: str | None = None, levels: int | None = None, species: bool = True, **kwargs):
        """TODO.

        Returns:
            (extra_model_kwargs, extra_dataloader_kwargs):
                Extra keyword arguments for the model and dataloader building functions.
        """
        if species:
            if "label_fn" in kwargs:
                raise ValueError(f"`label_fn` passed to `HierarchicalBuilder.spec_model_dataloader` when `{species=})`")
            kwargs["label_fn"] = linnean_labels_from_directory
        return parse_class_spec(path=path, dir=dir, levels=levels, **kwargs)

    @staticmethod
    def build_model(
        cls2idx: dict[int | str, dict[str, int]] | None = None,
        labels: OrderedDict[str, tuple[str, ...]] | None = None,
        *args,
        cls=HierarchicalClassifier,
        **kwargs,
    ):
        if labels is not None and cls2idx is not None:
            sparse_masks = sparse_masks_from_labels(labels, cls2idx)
        else:
            sparse_masks = None
        if cls2idx is not None:
            kwargs["cls2idx"] = cls2idx
        return BaseBuilder.build_model(*args, cls=cls, sparse_masks=sparse_masks, labels=labels, **kwargs)

    @staticmethod
    def build_dataloader(*args, **kwargs):
        if "multilabel" in kwargs:
            raise NotImplementedError("Do not supply `multilabel` to `build_dataloader` manually.")
        return BaseBuilder.build_dataloader(*args, multilabel=True, **kwargs)

    @staticmethod
    def build_criterion(
        *args,
        num_classes: list[int],
        device: torch.device,
        dtype: torch.dtype,
        weighted: bool = False,
        labels: Iterable[np.ndarray | torch.Tensor | list | tuple] | None = None,
        label_smoothing: float | None = None,
        **kwargs,
    ):
        if label_smoothing is None:
            label_smoothing = 1 / num_classes[0]
        if not weighted or labels is None:
            return MultiLevelWeightedCrossEntropyLoss(
                *args, label_smoothing=label_smoothing, num_classes=num_classes, device=device, dtype=dtype, **kwargs
            )
        labels_long = list(zip(*[labs.tolist() if not isinstance(labels, (list, tuple)) else labs for labs in labels]))
        counts = []
        for lvl, ncls in enumerate(num_classes):
            lvlc = Counter(labels_long[lvl])
            lvlc = [lvlc.get(i, 0) for i in range(ncls)]
            counts.append(lvlc)
        return MultiLevelWeightedCrossEntropyLoss(
            *args,
            class_frequencies=counts,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            device=device,
            dtype=dtype,
            loss_cls=EMLACrossEntropy,
            **kwargs,
        )


class HierarchicalResultCollector(BaseResultCollector):
    def __init__(
        self,
        model: nn.Module | None = None,
        idx2cls: dict[str, dict[int, str]] | None = None,
        cls2idx: dict[str, dict[str, int]] | None = None,
        scientific_names: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, idx2cls=idx2cls, cls2idx=cls2idx, scientific_names=scientific_names, *args, **kwargs)
        self._levels = None

    # --- Overridden Hooks for Hierarchical Operations ---
    def _cls2idx_to_scientific(self, cls2idx: dict) -> dict:
        return cls2idx_to_names(cls2idx)

    def _invert_mapping(self, mapping: dict) -> dict:
        return {lvl: {v: k for k, v in sub_map.items()} for lvl, sub_map in mapping.items()}

    def _is_known_label(self, label: str, level: int) -> bool:
        return label in self.cls2idx[str(level)]

    def _get_evaluation_rows(self):
        if self._levels is None:
            return
        labels = self.labels or repeat(tuple(["-1"] * self._levels))
        for i, (path, preds, labs, confs) in enumerate(zip(self.paths, self.preds, labels, self.confs)):
            for level in range(self._levels):
                yield i, path, level, labs[level], preds[level], confs[level]

    # --- Overridden Base Attribute Extraction ---
    def _collect_base_attributes(
        self, paths: list[str], predictions: list[torch.Tensor] | HierarchicalPrediction, labels: list[tuple[str, ...]] | None = None
    ):
        self.paths.extend(paths)

        if not isinstance(predictions, HierarchicalPrediction):
            predictions = HierarchicalPrediction(predictions, topk=1, cls2idx=self.cls2idx)
        if self._levels is None:
            self._levels = len(predictions[0].label)
        confidences, indices = predictions.confidence.squeeze(1), predictions.indices.squeeze(1)
        self.preds.extend([[self.idx2cls[str(lvl)][i.item()] for lvl, i in enumerate(idxs)] for idxs in indices])
        self.confs.extend([c.tolist() for c in confidences])

        if labels is not None:
            if self.scientific_names:
                labels = [tuple(self._get_scientific_name(e) for e in label) for label in labels]
            self.labels.extend(labels)

    def eval_label_fn(self, data: dict, outdir: str | None, save: bool, prefix: str = "", plot_conf_mat: bool = False, **kwargs):
        if kwargs:
            raise RuntimeError(
                f"Unknown arguments ([{', '.join(kwargs)}]) passed. "
                "Perhaps you forgot to implement the intended `eval_label_fn` in your subclass."
            )
        if save and not isinstance(outdir, str):
            raise RuntimeError("Attempted to save evaluated results against labels without specifying an output directory.")
        if self._levels is None:
            raise RuntimeError("Hierarchical result collector was unable to detect number of levels in the class hierarchy!")

        results = {}
        for level in range(self._levels):
            lvl_results = named_confusion_matrix(
                results={k: v[level] if k in ["preds", "confs", "labels"] else v for k, v in data.items()},
                cls2idx=self.cls2idx[str(level)],
                verbose=self.verbose,
            )
            results[level] = lvl_results

            if plot_conf_mat and save:
                assert outdir is not None
                dst = os.path.join(outdir, f"{prefix}confusion_matrix_level{level}.png")
                classes = [k for k, v in sorted(self.cls2idx[str(level)].items(), key=lambda x: x[1])]
                conf_mat = lvl_results["conf_mat"]

                conf_mat_arr = np.array([[conf_mat[g][p] for p in classes] for g in classes]).astype(np.float64)
                arr = plot_heatmap(conf_mat_arr, "magma", percent=False)
                from PIL.Image import fromarray

                fromarray(arr).save(dst)

        return results
