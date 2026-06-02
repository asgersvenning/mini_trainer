from typing import cast

import torch
import torchvision

from .core import BackboneModel


def get_bioclip_encoder(version: str = "bioclip-2", pretrained: bool = True):
    try:
        # pyrefly: ignore [missing-import]
        import open_clip
    except ImportError as e:
        e.add_note(
            "The `open_clip` module was not found in the current Python environment. Please install with `pip install open_clip_torch`."
        )
        raise

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        f"hf-hub:imageomics/{version}",
        load_weights=pretrained,
    )
    model = cast(open_clip.model.CLIP, model)
    preprocess_train = cast(torchvision.transforms.transforms.Compose, preprocess_train)
    preprocess_val = cast(torchvision.transforms.transforms.Compose, preprocess_val)
    tokenizer = open_clip.get_tokenizer(f"hf-hub:imageomics/{version}")
    tokenizer = cast(open_clip.tokenizer.SimpleTokenizer, tokenizer)

    return model, preprocess_train, preprocess_val, tokenizer


def get_bioclip_model(version: str = "bioclip-2", default_transform=None, pretrained: bool = True, **kwargs):
    encoder, _, bioclip_preprocess, tokenizer = get_bioclip_encoder(version.lower().strip(), pretrained=pretrained)
    encoder.compile(mode="reduce-overhead")
    if default_transform is None:
        default_transform = torchvision.transforms.transforms.Compose(
            [torchvision.transforms.transforms.ConvertImageDtype(dtype=torch.float32), bioclip_preprocess]
        )
    return BackboneModel(encoder=encoder, encoder_method="encode_image"), default_transform


def get_bioclip_models() -> list[str]:
    """Dynamically fetches the list of BioCLIP model versions from Hugging Face."""
    fallback = ["bioclip", "bioclip-2", "bioclip-2.5-vith14", "bioclip-vit-b-16-inat-only"]
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(author="imageomics")
        bioclip_models = []
        for m in models:
            if "bioclip" in m.id.lower():
                name = m.id.split("/")[-1]
                bioclip_models.append(name)
        if bioclip_models:
            return sorted(bioclip_models)
    except Exception as e:
        import warnings

        warnings.warn(
            f"Failed to dynamically retrieve BioCLIP models from Hugging Face Hub (error: {e}). "
            f"Falling back to the static known model list: {fallback}",
            UserWarning,
        )
    return fallback
