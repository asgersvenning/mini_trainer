from typing import cast

import torch
import torchvision

from .core import BackboneModel


def get_bioclip_encoder(version: str = "bioclip-2"):
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


def get_bioclip_model(version: str = "bioclip-2", default_transform=None, **kwargs):
    encoder, _, bioclip_preprocess, tokenizer = get_bioclip_encoder(version.lower().strip())
    encoder.compile(mode="reduce-overhead")
    if default_transform is None:
        default_transform = torchvision.transforms.transforms.Compose(
            [torchvision.transforms.transforms.ConvertImageDtype(dtype=torch.float32), bioclip_preprocess]
        )
    return BackboneModel(encoder=encoder, encoder_method="encode_image"), default_transform
