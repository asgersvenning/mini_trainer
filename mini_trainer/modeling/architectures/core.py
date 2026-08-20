import json
import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from mini_trainer import get_logger


def _read_blacklist():
    json_path = os.path.join(os.path.dirname(__file__), "blacklist.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"torchvision": [], "timm": [], "transformers": [], "bioclip": []}


BLACKLIST = _read_blacklist()


def resolve_embedding_dim(
    model: nn.Module, head_name: str, preprocess: Callable[[torch.Tensor], torch.Tensor], device: torch.device | None = None
) -> int:
    """
    Attempts to resolve the embedding dimension using fast structural checks.
    Falls back to a dummy forward pass only if all structural checks fail.
    """

    # timm standard
    if hasattr(model, "num_features") and isinstance(model.num_features, int):
        return model.num_features

    # OpenCLIP / standard ViT attribute
    if hasattr(model, "embed_dim") and isinstance(model.embed_dim, int):
        return model.embed_dim

    # torchvision VisionTransformer
    if hasattr(model, "hidden_dim") and isinstance(model.hidden_dim, int):
        return model.hidden_dim

    head_module = getattr(model, head_name)

    for m in head_module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
        if isinstance(m, nn.Conv2d):
            return m.in_channels

    get_logger().debug("Could not structurally infer embedding dimension. Falling back to dummy forward pass.")

    return _infer_via_dummy_pass(model, preprocess, device)


def _infer_via_dummy_pass(model: nn.Module, preprocess: Callable[[torch.Tensor], torch.Tensor], device: torch.device | None) -> int:
    """The robust dummy pass fallback."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    was_training = model.training
    model.eval()
    dummy_input = preprocess(torch.zeros(3, 224, 224, dtype=torch.uint8)).unsqueeze(0).to(device)

    with torch.inference_mode():
        try:
            output = model(dummy_input)
        except Exception as e:
            model.train(was_training)
            raise RuntimeError(f"Dummy pass failed. You may need to provide a specific `fallback_input_shape` for this model. Error: {e}")

    model.train(was_training)

    if isinstance(output, tuple):
        output = output[0]
    elif isinstance(output, dict):
        output = next(iter(output.values()))

    if output.ndim >= 2:
        return output.shape[1]

    raise ValueError(f"Unexpected output shape from dummy pass: {output.shape}")


class Preprocess:
    def __init__(self, transform=None, func=None):
        """Hook torchvision preprocessing function with load image from file to tensor."""
        self.transform = transform
        self.func = func

    def __call__(self, item):
        if isinstance(item, str):
            path = str(item)
            if not os.path.exists(path):
                raise FileNotFoundError("Unable to find image: " + path)
            from torchvision.io import ImageReadMode, decode_image

            image = decode_image(path, ImageReadMode.RGB)
        elif isinstance(item, torch.Tensor):
            image = item
        else:
            raise TypeError(f"'item' must be of type `str` or `torch.Tensor`, not {type(item)}")
        if self.transform is not None:
            image = self.transform(image)
        if self.func is not None:
            image = self.func(image)
        return image

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.transform} + {self.func})"


def module_output_dim(module: nn.Module):
    """Finds the output dimension by looking for the last parameter and returning the right-most size."""
    for param in reversed(list(module.parameters())):
        if param.ndim > 0:
            return param.shape[0]

    raise ValueError(f"Could not determine output dimension for {type(module)}")


class WrappedEncoder(nn.Module):
    """Barebones encoder wrapper."""

    def __init__(self, encoder: nn.Module, encoder_method: str | None = None):  # noqa: D107
        super().__init__()
        self.encoder = encoder
        self.encoder_method = encoder_method

        self._is_trainable = any(p.requires_grad for p in self.encoder.parameters())

    def requires_grad_(self, requires_grad: bool = True):
        """Override to update our internal cache when the user freezes/unfreezes."""
        super().requires_grad_(requires_grad)
        self._is_trainable = any(p.requires_grad for p in self.encoder.parameters())
        return self

    def get_extra_state(self):
        """Standard PyTorch hook to save non-tensor state."""
        return {"encoder_method": self.encoder_method}

    def set_extra_state(self, state):
        """Standard PyTorch hook to load non-tensor state."""
        if "encoder_method" in state:
            encoder_method = state["encoder_method"]
        else:
            get_logger().debug("No `encoder_method` found in state, assuming None.")
            encoder_method = None
        self.encoder_method = encoder_method

    def forward(self, x):
        def _inner():
            if self.encoder_method is not None:
                return getattr(self.encoder, self.encoder_method)(x)
            return self.encoder(x)

        if self._is_trainable:
            return _inner()
        with torch.inference_mode():
            return _inner()


class BackboneModel(nn.Module):
    """A barebones wrapper for arbitrary encoder-only modules."""

    def __init__(self, encoder: nn.Module, encoder_method: str | None = None):  # noqa: D107
        super().__init__()
        self.backbone = WrappedEncoder(encoder, encoder_method)
        self.classifier = nn.Linear(in_features=module_output_dim(self.backbone), out_features=10)

    def forward(self, x):
        x = self.backbone(x)
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(
                f"Output of encoder of type {type(self.backbone)} was of type {type(x)}, "
                "but it should be a torch.Tensor."
                "\nPerhaps you forgot to pass the relevant `encoder_method` to `BackboneModel`?"
            )
        return self.classifier(x)


def infer_size_from_transform(transform: Any, fallback: int = 256, warn_on_fallback: bool = True) -> int:
    if transform is not None:
        # Unwrap common wrapper attributes if present
        for attr in ("transform", "processor"):
            if hasattr(transform, attr):
                transform = getattr(transform, attr)

        if hasattr(transform, "crop_size"):
            cs = getattr(transform, "crop_size")
            if isinstance(cs, (list, tuple)) and len(cs) > 0:
                return cs[0]
            if isinstance(cs, int):
                return cs
        elif hasattr(transform, "size"):
            sz = getattr(transform, "size")
            if isinstance(sz, int):
                return sz
            elif isinstance(sz, dict):
                for key in ["height", "width", "shortest_edge"]:
                    if key in sz:
                        return sz[key]
            elif isinstance(sz, (list, tuple)) and len(sz) > 0:
                return sz[0]
        elif hasattr(transform, "transforms"):
            for op in transform.transforms:
                op_sz = infer_size_from_transform(op, fallback=-1, warn_on_fallback=False)
                if op_sz != -1:
                    return op_sz
    if warn_on_fallback:
        get_logger().debug(f"Could not infer preferred input size from transform. Falling back to default size of {fallback}.")
    return fallback
