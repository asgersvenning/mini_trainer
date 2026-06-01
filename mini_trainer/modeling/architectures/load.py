from functools import partial
from typing import Any

import torch
from torch import nn

from mini_trainer.utils import import_class, make_convert_dtype

from .bioclip import get_bioclip_model
from .core import preprocess
from .torchvision import get_torchvision_model


def get_dynamic_model(model_class_path: str, transform, **model_args):
    return import_class(model_class_path)(**model_args), transform


def get_model(
    backbone_model: str | nn.Module,
    model_args: dict = {},
    classifier_name: str | list[str] = ["classifier", "fc", "heads", "head"],
    preprocess_dtype: torch.dtype | None = None,
    transform: Any = None,
):
    """Get torchvision model and preprocessing function by name."""
    default_transform = transform
    if isinstance(backbone_model, str):
        if "bioclip" in backbone_model.lower().strip():
            backbone_model, default_transform = get_bioclip_model(backbone_model, default_transform, **model_args)
        elif "." in backbone_model:
            backbone_model, default_transform = get_dynamic_model(backbone_model, default_transform, **model_args)
        else:
            backbone_model, default_transform = get_torchvision_model(backbone_model, default_transform, **model_args)
    if not isinstance(backbone_model, nn.Module):
        raise ValueError("backbone_model must be a string or a torch.nn.Module")

    if default_transform is None:
        for attr in ("transforms", "default_transform", "preprocess_transform", "transform"):
            if hasattr(backbone_model, attr):
                val = getattr(backbone_model, attr)
                if callable(val):
                    try:
                        default_transform = val()
                    except Exception:
                        default_transform = val
                else:
                    default_transform = val
                break

    backbone_classifier_name = None
    if isinstance(classifier_name, str):
        classifier_name = [classifier_name]
    for name, module in backbone_model.named_modules():
        if name in classifier_name:
            backbone_classifier_name = name
            break
    if backbone_classifier_name is None:
        raise AttributeError(f"No classifier found with names {classifier_name}")
    return (
        backbone_model,
        backbone_classifier_name,
        partial(
            preprocess,
            transform=default_transform,
            func=preprocess_dtype if preprocess_dtype is None else make_convert_dtype(preprocess_dtype),
        ),
    )
