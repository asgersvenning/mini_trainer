from collections.abc import Callable
from typing import Any

import torchvision

from . import core


def _load_blacklist() -> set[str]:
    return set(core.BLACKLIST.get("torchvision", []))


def get_torchvision_model(
    model: str,
    default_transform: Callable | None = None,
    weights: Any = "DEFAULT",
    pretrained: bool = True,
    **kwargs: Any,
):
    """Load a pretrained torchvision model.

    Args:
        model: The name of the model to load.
        default_transform: The default transform to use for the model.
        weights: The weights to use for the model.
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        The loaded model.
    """
    if model in _load_blacklist():
        raise ValueError(f"The model {model} is not supported.")

    weight_enum = torchvision.models.get_model_weights(model)

    if isinstance(weights, str):
        weights = getattr(weight_enum, weights.upper())

    if not pretrained:
        weights = None

    if default_transform is None:
        if weights is not None:
            default_transform = weights.transforms()
        else:
            default_transform = weight_enum.DEFAULT.transforms()

    return torchvision.models.get_model(model, weights=weights, **kwargs), default_transform
