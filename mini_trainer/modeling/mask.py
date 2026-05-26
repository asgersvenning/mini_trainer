from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from .classifier import classification_module


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
