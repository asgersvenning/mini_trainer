import json
import os
from typing import Any, NamedTuple

import torch
from torch import nn

from mini_trainer.utils import import_class, make_convert_dtype

from . import core
from .bioclip import get_bioclip_model, get_bioclip_models
from .core import Preprocess, infer_size_from_transform, resolve_embedding_dim
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
    blacklisted: bool = False


def load_blacklist() -> dict[str, list[str]]:
    """Loads the model blacklist from blacklist.json."""
    return core.BLACKLIST


def save_blacklist(blacklist: dict[str, list[str]]) -> None:
    """Saves the model blacklist to blacklist.json."""
    core.BLACKLIST.clear()
    core.BLACKLIST.update(blacklist)
    json_path = os.path.join(os.path.dirname(__file__), "blacklist.json")
    with open(json_path, "w") as f:
        json.dump(blacklist, f, indent=4)


def _infer_size_from_transform(transform: Any) -> int:
    return infer_size_from_transform(transform, fallback=256, warn_on_fallback=True)


def get_dynamic_model(model_class_path: str, transform, resize_size: int | None = None, **model_args):
    backbone_model = import_class(model_class_path)(**model_args)
    return backbone_model, transform, _infer_size_from_transform(transform)


def resolve_backbone_getter(backbone_model: str) -> tuple[Any, str]:
    """Resolves a model name to the appropriate getter function and clean model name."""
    if ":" in backbone_model:
        source, actual_model_name = backbone_model.split(":", 1)
        source_map = {
            "bioclip": get_bioclip_model,
            "timm": get_timm_model,
            "transformers": get_transformers_model,
            "hf": get_transformers_model,
            "hf-hub": get_transformers_model,
        }
        if source in source_map:
            return source_map[source], actual_model_name

        getter_name = f"get_{source}_model"
        if getter_name in globals():
            return globals()[getter_name], actual_model_name

    if "." in backbone_model:
        return get_dynamic_model, backbone_model

    if "bioclip" in backbone_model.lower():
        return get_bioclip_model, backbone_model

    if ":" in backbone_model:
        source, _ = backbone_model.split(":", 1)
        raise ValueError(f"Unknown source '{source}' in backbone model identifier '{backbone_model}'")

    return get_torchvision_model, backbone_model


def get_model(
    backbone_model: str | nn.Module,
    model_args: dict = {},
    classifier_name: str | list[str] = ["classifier", "fc", "heads", "head", "logits"],
    preprocess_dtype: torch.dtype | None = None,
    transform: Any = None,
    device: torch.device | None = None,
):
    """Get torchvision, timm, transformers, or bioclip model and preprocessing function by name."""
    default_transform = transform
    preferred_size = None
    if isinstance(backbone_model, str):
        getter, clean_name = resolve_backbone_getter(backbone_model)

        backend_map = {
            get_torchvision_model: "torchvision",
            get_timm_model: "timm",
            get_transformers_model: "transformers",
            get_bioclip_model: "bioclip",
        }
        backend = backend_map.get(getter, "custom")
        blacklist = load_blacklist()
        if clean_name in blacklist.get(backend, []):
            raise ValueError(f"Model '{backbone_model}' (backend '{backend}') is blacklisted due to compatibility issues.")

        clean_args = dict(model_args)
        if getter is not get_transformers_model:
            clean_args.pop("local_files_only", None)
        backbone_model, default_transform, preferred_size = getter(clean_name, default_transform, **clean_args)

    if not isinstance(backbone_model, nn.Module):
        raise ValueError("backbone_model must be a string or a torch.nn.Module")

    # Resolve Transform
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

    # Build the exact preprocess pipeline
    preprocess_pipeline = Preprocess(
        transform=default_transform,
        func=preprocess_dtype if preprocess_dtype is None else make_convert_dtype(preprocess_dtype),
    )

    # Resolve exact classifier name (using named_modules for exact paths like "head.fc")
    backbone_classifier_name = None

    # 1. Try timm's native method first
    if hasattr(backbone_model, "get_classifier") and callable(backbone_model.get_classifier):
        try:
            timm_classifier = backbone_model.get_classifier()
            # Search all modules to find the exact match
            for name, module in backbone_model.named_modules():
                if module is timm_classifier:
                    backbone_classifier_name = name
                    break
        except Exception:
            pass

    # 2. Fallback to name matching
    if backbone_classifier_name is None:
        if isinstance(classifier_name, str):
            classifier_name = [classifier_name]

        # Prioritize exact top-level children first
        for name, child in backbone_model.named_children():
            if name in classifier_name:
                backbone_classifier_name = name
                break

        # If not found at top level, search deeply
        if backbone_classifier_name is None:
            for name, module in backbone_model.named_modules():
                # We split by '.' so we match the local name (e.g., 'fc' in 'head.fc')
                if name.split(".")[-1] in classifier_name:
                    backbone_classifier_name = name
                    break

    if backbone_classifier_name is None:
        raise AttributeError(f"No classifier found matching names {classifier_name}")

    # Calculate embedding dimension using our robust resolver
    embedding_dim = resolve_embedding_dim(
        model=backbone_model, head_name=backbone_classifier_name, preprocess=preprocess_pipeline, device=device
    )

    if preferred_size is None:
        preferred_size = _infer_size_from_transform(default_transform)

    return backbone_model, backbone_classifier_name, preprocess_pipeline, embedding_dim, preferred_size


def list_supported_backbones() -> list[BackboneInfo]:
    """Generates a list of supported models, showing backend and availability status."""
    rows = []
    blacklist = load_blacklist()

    # 1. Torchvision
    import torchvision

    tv_models = []
    if hasattr(torchvision.models, "list_models"):
        tv_models = torchvision.models.list_models()
    for m in tv_models:
        rows.append(
            BackboneInfo(
                model=m,
                backend="torchvision",
                availability=True,
                blacklisted=m in blacklist.get("torchvision", []),
            )
        )

    # 2. BioCLIP
    has_open_clip = False
    try:
        import open_clip  # noqa: F401

        has_open_clip = True
    except ImportError:
        pass
    bioclip_versions = get_bioclip_models()
    for version in bioclip_versions:
        rows.append(
            BackboneInfo(
                model=f"bioclip:{version}",
                backend="bioclip",
                availability=has_open_clip,
                blacklisted=version in blacklist.get("bioclip", []),
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
            rows.append(
                BackboneInfo(
                    model=f"timm:{m}",
                    backend="timm",
                    availability=True,
                    blacklisted=m in blacklist.get("timm", []),
                )
            )
    else:
        popular_timm = ["vit_tiny_patch16_224", "resnet10t", "efficientnet_b0", "convnext_tiny"]
        for m in popular_timm:
            rows.append(
                BackboneInfo(
                    model=f"timm:{m}",
                    backend="timm",
                    availability=False,
                    blacklisted=m in blacklist.get("timm", []),
                )
            )

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
                    blacklisted=repo in blacklist.get("transformers", []),
                )
            )
    else:
        for m, repo in sorted(popular_transformers.items()):
            rows.append(
                BackboneInfo(
                    model=f"hf-hub:{repo}",
                    backend="transformers",
                    availability=False,
                    blacklisted=repo in blacklist.get("transformers", []),
                )
            )

    return rows
