from collections.abc import Callable
from typing import Any

import torchvision


def get_torchvision_model(
    model: str,
    default_transform: Callable | None = None,
    weights: Any = "DEFAULT",
    pretrained: bool = True,
    resize_size: int | None = None,
    **kwargs: Any,
):
    """Load a pretrained torchvision model.

    Args:
        model: The name of the model to load.
        default_transform: The default transform to use for the model.
        weights: The weights to use for the model.
        pretrained: Whether to load pretrained weights.
        resize_size: Target size to resize/crop images.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        The loaded model.
    """
    weight_enum = torchvision.models.get_model_weights(model)

    if isinstance(weights, str):
        weights = getattr(weight_enum, weights.upper())

    if not pretrained:
        weights = None

    if default_transform is None:
        transform_kwargs = {}
        if resize_size is not None:
            transform_kwargs["crop_size"] = [resize_size]
            transform_kwargs["resize_size"] = [int(resize_size * (256 / 224))]
        if weights is not None:
            default_transform = weights.transforms(**transform_kwargs)
        else:
            default_transform = weight_enum.DEFAULT.transforms(**transform_kwargs)

    backbone_model = torchvision.models.get_model(model, weights=weights, **kwargs)

    return backbone_model, default_transform, None

