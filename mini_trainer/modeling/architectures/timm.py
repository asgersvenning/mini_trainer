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


def get_timm_model(model: str, default_transform: Any = None, **kwargs: Any) -> tuple[Any, Any]:
    """Load timm model and resolve its default transform."""
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ImportError as e:
        e.add_note("The `timm` module was not found in the current Python environment. Please install with `pip install timm`.")
        raise

    if "pretrained" not in kwargs:
        kwargs["pretrained"] = True

    backbone_model = timm.create_model(model, **kwargs)

    if default_transform is None:
        config = resolve_model_data_config(backbone_model)
        timm_transform = create_transform(**config, is_training=False)
        default_transform = TimmTransformWrapper(timm_transform)

    return backbone_model, default_transform
