from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from .classifier import classification_module


def restrict_class_labels(model: nn.Module, labels: list[str]) -> dict:
    """Restrict current candidate labels, preserving original weight indices.

    For hierarchical heads, labels refer to the leaf level. Missing labels are
    reported rather than added; an empty intersection is an error.
    """
    classifier = classification_module(model)
    mapping = classifier.metadata["cls2idx"]
    if mapping and isinstance(next(iter(mapping.values())), dict):
        mapping = mapping["0"] if "0" in mapping else mapping[0]
    requested = set(labels)
    retained = sorted((label for label in mapping if label in requested), key=mapping.__getitem__)
    if not retained:
        raise ValueError("Class list has no overlap with the model's active vocabulary")
    indices = [mapping[label] for label in retained]
    if classifier.active_indices is not None:
        indices = classifier.active_indices[indices].tolist()
    classifier.set_active_features(indices)
    return {
        "requested_count": len(requested),
        "original_candidate_count": len(mapping),
        "retained_count": len(retained),
        "retained_labels": retained,
        "missing_labels": sorted(requested - set(mapping)),
        "excluded_labels": sorted(set(mapping) - requested),
    }


def set_classification_mask(model: nn.Module, indices: list[int] | torch.Tensor | np.ndarray | None = None):
    """Mask a selection of output features (classes).

    Args:
        model: A model created with `mini_trainer.classifier.Classifier.build()`.
        indices: Indices to (reversibly) mask in forward pass. If None the mask is disabled.
    """
    classification_module(model).set_active_features(indices)


@contextmanager
def mask_classifier(model: nn.Module, indices: list[int] | torch.Tensor | np.ndarray | None = None):
    """Mask a selection of output features (classes).

    Args:
        model: A model created with `mini_trainer.classifier.Classifier.build()`.
        indices: Indices to (reversibly) mask in forward pass. If None the mask is disabled.
    """
    classifier = classification_module(model)
    orig_indices = classifier.active_indices

    classifier.set_active_features(indices)

    try:
        yield
    finally:
        classifier.set_active_features(orig_indices)
