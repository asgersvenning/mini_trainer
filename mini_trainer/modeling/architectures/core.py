import json
import os
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn

from mini_trainer import get_logger
from mini_trainer.utils import affine_resample, read_square_crop


def _read_blacklist():
    """Use empty backend lists when the local blacklist is absent or unreadable."""
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
    """Infer feature width from model/head structure, falling back to a dummy pass."""

    # timm standard
    if hasattr(model, "num_features") and isinstance(model.num_features, int):
        return model.num_features

    # OpenCLIP / standard ViT attribute
    if hasattr(model, "embed_dim") and isinstance(model.embed_dim, int):
        return model.embed_dim

    # torchvision VisionTransformer
    if hasattr(model, "hidden_dim") and isinstance(model.hidden_dim, int):
        return model.hidden_dim

    head_module = model.get_submodule(head_name)  # dotted paths: e.g. timm's head.fc

    for m in head_module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
        if isinstance(m, nn.Conv2d):
            return m.in_channels

    get_logger().debug("Could not structurally infer embedding dimension. Falling back to dummy forward pass.")

    return _infer_via_dummy_pass(model, preprocess, device)


def _infer_via_dummy_pass(model: nn.Module, preprocess: Callable[[torch.Tensor], torch.Tensor], device: torch.device | None) -> int:
    """Infer feature width from a preprocessed 224-pixel dummy image."""
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
    def __init__(self, transform=None, func=None, reader=None):
        """Decode image paths (with ``reader`` when given) and apply the optional transform and postprocessor."""
        self.transform = transform
        self.func = func
        self.reader = reader

    def __call__(self, item):
        if isinstance(item, str):
            path = str(item)
            if not os.path.exists(path):
                raise FileNotFoundError("Unable to find image: " + path)
            from torchvision.io import ImageReadMode, decode_image

            image = self.reader(path) if self.reader is not None else decode_image(path, ImageReadMode.RGB)
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResampleNormalize(nn.Module):
    """Model input for square-crop reading: resample to ``size`` when needed, then normalise.

    Loaders read squares at ``read_size``. Evaluation batches are resampled to ``size`` on an identity
    grid (bilinear, reflection padding, ``align_corners=True``), which is fastai's validation path;
    training batches already arrive at ``size`` from ``RandomAffineResize``, which samples the affine and
    the resize in one step. Any other input size is rejected rather than silently resampled twice.
    uint8 input is scaled by 1/255; float input is taken to be in ``[0, 1]``. Returns float32.
    """

    def __init__(self, size: int, read_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD):  # noqa: D107
        super().__init__()
        self.size, self.read_size = int(size), int(read_size)
        self.mean, self.std = tuple(float(m) for m in mean), tuple(float(s) for s in std)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        single = images.ndim == 3
        x = images.unsqueeze(0) if single else images
        x = x.float().div(255) if x.dtype == torch.uint8 else x.float()
        if tuple(x.shape[-2:]) != (self.size, self.size):
            if tuple(x.shape[-2:]) != (self.read_size, self.read_size):
                raise ValueError(
                    f"Expected images of {self.read_size}x{self.read_size} (read) or {self.size}x{self.size} (augmented), "
                    f"got {tuple(x.shape[-2:])}."
                )
            x = affine_resample(x, None, self.size)
        mean = torch.tensor(self.mean, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, -1, 1, 1)
        x = (x - mean) / std
        return x[0] if single else x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, read_size={self.read_size}, mean={self.mean}, std={self.std})"


def square_crop_preprocess(preprocess: "Preprocess", size: int | tuple[int, int] | list[int], read_size: int | None = None) -> "Preprocess":
    """Model preprocessing for square-crop reading at ``read_size`` (default ``size``) and a ``size`` model input.

    Normalisation statistics and the dtype postprocessor are taken from ``preprocess`` when available.
    Image paths are read as centred squares, as the evaluation loader reads them.
    """
    if not isinstance(size, int):
        if len(size) != 2 or size[0] != size[1]:
            raise ValueError(f"Square-crop preprocessing needs a square model size, not {size}.")
        size = int(size[0])
    read_size = int(read_size or size)
    transform = getattr(preprocess, "transform", None)
    mean, std = getattr(transform, "mean", IMAGENET_MEAN), getattr(transform, "std", IMAGENET_STD)
    return Preprocess(
        transform=ResampleNormalize(size, read_size, mean, std),
        func=getattr(preprocess, "func", None),
        reader=partial(read_square_crop, size=read_size, position=(0.5, 0.5)),
    )


def module_output_dim(module: nn.Module):
    """Infer output width from the first dimension of the last non-scalar parameter."""
    for param in reversed(list(module.parameters())):
        if param.ndim > 0:
            return param.shape[0]

    raise ValueError(f"Could not determine output dimension for {type(module)}")


class WrappedEncoder(nn.Module):
    """Wrap an encoder method, caching trainability for inference-mode selection."""

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
        """Retain the selected encoder method in checkpoints."""
        return {"encoder_method": self.encoder_method}

    def set_extra_state(self, state):
        """Restore the encoder method, defaulting older checkpoints to forward."""
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
    """Wrap an encoder with a replaceable linear classification head."""

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
    """Infer preferred size from wrapper/crop/resize metadata, or return fallback."""
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
