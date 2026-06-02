from functools import partial
from typing import Any, NamedTuple

import torch
from torch import nn

from mini_trainer.utils import import_class, make_convert_dtype

from .bioclip import get_bioclip_model
from .core import preprocess
from .timm import get_timm_model
from .torchvision import get_torchvision_model
from .transformers import get_transformers_model


class BackboneInfo(NamedTuple):
    """Data container for a supported backbone model.

    Provides statically typed, attribute-based access (e.g. info.model)
    and guaranteed ordering as a named tuple.
    """

    model: str
    backend: str
    availability: bool


def get_dynamic_model(model_class_path: str, transform, **model_args):
    return import_class(model_class_path)(**model_args), transform


def resolve_backbone_getter(backbone_model: str) -> tuple[Any, str]:
    """Resolves a model name to the appropriate getter function and clean model name."""
    if "bioclip" in backbone_model.lower().strip():
        return get_bioclip_model, backbone_model
    if backbone_model.startswith(("timm:", "timm/")):
        if backbone_model.startswith("timm:"):
            actual_model_name = backbone_model.split(":", 1)[-1]
        else:
            actual_model_name = backbone_model.split("/", 1)[-1]
        return get_timm_model, actual_model_name
    if backbone_model.startswith(("transformers:", "transformers/", "hf:", "hf-hub:")):
        if backbone_model.startswith(("transformers:", "hf:", "hf-hub:")):
            actual_model_name = backbone_model.split(":", 1)[-1]
        else:
            actual_model_name = backbone_model.split("/", 1)[-1]
        return get_transformers_model, actual_model_name
    if "." in backbone_model:
        return get_dynamic_model, backbone_model

    # Auto-detection
    try:
        import torchvision

        _ = torchvision.models.get_model_weights(backbone_model)
        return get_torchvision_model, backbone_model
    except (ValueError, AttributeError):
        pass

    try:
        import timm

        if timm.is_model(backbone_model):
            return get_timm_model, backbone_model
    except ImportError:
        pass

    if "/" in backbone_model:
        return get_transformers_model, backbone_model

    return get_torchvision_model, backbone_model


def get_model(
    backbone_model: str | nn.Module,
    model_args: dict = {},
    classifier_name: str | list[str] = ["classifier", "fc", "heads", "head", "logits"],
    preprocess_dtype: torch.dtype | None = None,
    transform: Any = None,
):
    """Get torchvision, timm, transformers, or bioclip model and preprocessing function by name."""
    default_transform = transform
    if isinstance(backbone_model, str):
        getter, clean_name = resolve_backbone_getter(backbone_model)
        backbone_model, default_transform = getter(clean_name, default_transform, **model_args)

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
    if hasattr(backbone_model, "get_classifier") and callable(backbone_model.get_classifier):
        try:
            timm_classifier = backbone_model.get_classifier()
            for name, child in backbone_model.named_children():
                if child is timm_classifier or any(m is timm_classifier for m in child.modules()):
                    backbone_classifier_name = name
                    break
        except Exception:
            pass

    if backbone_classifier_name is None:
        if isinstance(classifier_name, str):
            classifier_name = [classifier_name]
        for name, child in backbone_model.named_children():
            if name in classifier_name:
                backbone_classifier_name = name
                break
        if backbone_classifier_name is None:
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


def list_supported_backbones() -> list[BackboneInfo]:
    """Generates a list of supported models, showing backend and availability status."""
    rows = []

    # 1. Torchvision
    import torchvision

    tv_models = []
    try:
        from .torchvision import _UNSUPPORTED_MODELS
    except ImportError:
        _UNSUPPORTED_MODELS = []

    if hasattr(torchvision.models, "list_models"):
        all_tv = torchvision.models.list_models()
        tv_models = [m for m in all_tv if m not in _UNSUPPORTED_MODELS]
    for m in tv_models:
        rows.append(BackboneInfo(model=m, backend="torchvision", availability=True))

    # 2. BioCLIP
    has_open_clip = False
    try:
        import open_clip  # noqa: F401

        has_open_clip = True
    except ImportError:
        pass
    rows.append(
        BackboneInfo(
            model="bioclip-2",
            backend="bioclip",
            availability=has_open_clip,
        )
    )

    # 3. Timm
    has_timm = False
    try:
        import timm

        has_timm = True
    except ImportError:
        pass

    if has_timm:
        timm_models = timm.list_models()
        for m in timm_models:
            rows.append(BackboneInfo(model=f"timm:{m}", backend="timm", availability=True))
    else:
        # Static list of some popular timm models as placeholders
        popular_timm = ["vit_tiny_patch16_224", "resnet10t", "efficientnet_b0", "convnext_tiny"]
        for m in popular_timm:
            rows.append(BackboneInfo(model=f"timm:{m}", backend="timm", availability=False))

    # 4. Transformers
    has_transformers = False
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING

        has_transformers = True
    except ImportError:
        pass

    popular_transformers = {
        "vit": "google/vit-base-patch16-224",
        "swin": "microsoft/swin-tiny-patch4-window7-224",
        "convnext": "facebook/convnext-tiny-224",
        "resnet": "microsoft/resnet-50",
        "deit": "facebook/deit-tiny-patch16-224",
        "dinov2": "facebook/dinov2-base",
    }

    if has_transformers:
        supported_types = []
        for model_type, config_cls in CONFIG_MAPPING.items():
            if config_cls in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING:
                supported_types.append(model_type)
        for m in sorted(supported_types):
            repo = popular_transformers.get(m, f"google/{m}-base-patch16-224")
            rows.append(
                BackboneInfo(
                    model=f"hf-hub:{repo}",
                    backend="transformers",
                    availability=True,
                )
            )
    else:
        for m, repo in sorted(popular_transformers.items()):
            rows.append(
                BackboneInfo(
                    model=f"hf-hub:{repo}",
                    backend="transformers",
                    availability=False,
                )
            )

    return rows
