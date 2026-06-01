import os
import warnings

import torch
from torch import nn
from torchvision.io import ImageReadMode, decode_image


def preprocess(item, transform, func=None):
    """Hook torchvision preprocessing function with load image from file to tensor."""
    if isinstance(item, str):
        path = str(item)
        if not os.path.exists(path):
            raise FileNotFoundError("Unable to find image: " + path)
        image = decode_image(path, ImageReadMode.RGB)
    elif isinstance(item, torch.Tensor):
        image = item
    else:
        raise TypeError(f"'item' must be of type `str` or `torch.Tensor`, not {type(item)}")
    if transform is not None:
        image = transform(image)
    if func:
        image = func(image)
    return image


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
            warnings.warn("No `encoder_method` found in state, assuming None.", UserWarning)
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
