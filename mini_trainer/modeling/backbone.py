import os
import warnings
from functools import partial
from typing import Any, cast

import torch
import torchvision
from torch import nn
from torchvision.io import ImageReadMode, decode_image

from mini_trainer.utils._core.misc import make_convert_dtype

_UNSUPPORTED_MODELS = [
    "squeezenet1_0",
    "squeezenet1_1",
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "keypointrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
    "mvit_v1_b",
    "mvit_v2_s",
    "raft_large",
    "raft_small",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
    "swin3d_b",
    "swin3d_s",
    "swin3d_t",
]


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


def get_bioclip2_encoder(version: str = "bioclip-2"):
    try:
        # pyrefly: ignore [missing-import]
        import open_clip
    except ImportError as e:
        e.add_note(
            "The `open_clip` module was not found in the current Python environment. Please install with `pip install open_clip_torch`."
        )
        raise

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(f"hf-hub:imageomics/{version}")
    model = cast(open_clip.model.CLIP, model)
    preprocess_train = cast(torchvision.transforms.transforms.Compose, preprocess_train)
    preprocess_val = cast(torchvision.transforms.transforms.Compose, preprocess_val)
    tokenizer = open_clip.get_tokenizer(f"hf-hub:imageomics/{version}")
    tokenizer = cast(open_clip.tokenizer.SimpleTokenizer, tokenizer)

    return model, preprocess_train, preprocess_val, tokenizer


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
            encoder, _, bioclip_preprocess, tokenizer = get_bioclip2_encoder(backbone_model.lower().strip())
            encoder.compile(mode="reduce-overhead")
            if default_transform is None:
                default_transform = torchvision.transforms.transforms.Compose(
                    [torchvision.transforms.transforms.ConvertImageDtype(dtype=torch.float32), bioclip_preprocess]
                )
            backbone_model = BackboneModel(encoder=encoder, encoder_method="encode_image")
        elif "." in backbone_model:
            from mini_trainer.utils import import_class
            cls = import_class(backbone_model)
            backbone_model = cls(**model_args)
        else:
            if backbone_model in _UNSUPPORTED_MODELS:
                raise ValueError(f"The model {backbone_model} is not supported.")
            default_weights = torchvision.models.get_model_weights(backbone_model).DEFAULT
            if default_transform is None:
                try:
                    default_transform = default_weights.transforms(antialias=True)
                except TypeError as e:
                    if "unexpected keyword argument 'antialias'" not in str(e):
                        raise
                    default_transform = default_weights.transforms()
            backbone_model = torchvision.models.get_model(backbone_model, weights=default_weights, **model_args)
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
