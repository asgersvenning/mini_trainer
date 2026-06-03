from typing import Any

import torch


class TimmTransformWrapper:
    """Wrapper to handle float conversion for timm transforms if input is a uint8 tensor."""

    def __init__(self, transform: Any):
        self.transform = transform

    def __call__(self, image: Any) -> Any:
        if isinstance(image, torch.Tensor):
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
        return self.transform(image)


def get_timm_model(
    model: str,
    default_transform: Any = None,
    pretrained: bool = True,
    resize_size: int | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, int]:
    """Load timm model and resolve its default transform."""
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as e:
        e.add_note(
            "The `timm` module was not found in the current Python environment. Please install with `pip install mini-trainer[timm]`."
        )
        raise

    backbone_model = timm.create_model(model, pretrained=pretrained, **kwargs)

    if default_transform is None:
        config = resolve_model_data_config(backbone_model)
        if resize_size is not None:
            config["input_size"] = (3, resize_size, resize_size)
        timm_transform = create_transform(**config, is_training=False)
        default_transform = TimmTransformWrapper(timm_transform)

    preferred_size = _infer_preferred_size(backbone_model)

    return backbone_model, default_transform, preferred_size


def _infer_preferred_size(backbone_model: Any) -> int | None:
    """Infer the preferred input size from the timm model default configuration."""
    cfg = getattr(backbone_model, "default_cfg", None)
    if isinstance(cfg, dict):
        input_size = cfg.get("input_size")
        if isinstance(input_size, (list, tuple)) and len(input_size) >= 2:
            return input_size[-1]
    return None

