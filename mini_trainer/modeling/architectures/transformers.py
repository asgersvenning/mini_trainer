from typing import Any

import torch


class TransformersProcessorTransform:
    """Wrapper to adapt Hugging Face AutoImageProcessor to torchvision-like transform."""

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, image: Any) -> Any:
        from torchvision.transforms.functional import to_pil_image

        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values.squeeze(0)
        return pixel_values


class TransformersBackboneWrapper(torch.nn.Module):
    """Wrapper for Hugging Face image classification models to output a raw logits tensor."""

    def __init__(self, hf_model: Any, classifier_name: str):
        super().__init__()
        self.hf_model = hf_model
        self.classifier_name = classifier_name
        self.add_module(classifier_name, getattr(hf_model, classifier_name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_head = getattr(self, self.classifier_name)
        if getattr(self.hf_model, self.classifier_name) is not current_head:
            self.hf_model.add_module(self.classifier_name, current_head)
        outputs = self.hf_model(pixel_values=x)
        return outputs.logits


def get_transformers_model(
    model: str, default_transform: Any = None, pretrained: bool = True, **kwargs: Any,
) -> tuple[Any, Any]:
    """Load Hugging Face transformers classification model and resolve its default transform."""
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as e:
        e.add_note(
            "The `transformers` module was not found in the current Python environment. Please install with `pip install transformers`."
        )
        raise

    # Extract hub kwargs that are applicable to both model/config loading and processor loading
    hub_kwargs = {}
    for key in ["local_files_only", "revision", "cache_dir", "force_download", "proxies", "token"]:
        if key in kwargs:
            hub_kwargs[key] = kwargs[key]

    if pretrained:
        hf_model = AutoModelForImageClassification.from_pretrained(model, **kwargs)
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, **hub_kwargs)
        hf_model = AutoModelForImageClassification.from_config(config)

    classifier_name = None
    for name in ["classifier", "logits", "head"]:
        if hasattr(hf_model, name):
            classifier_name = name
            break

    if classifier_name is None:
        for name, child in hf_model.named_children():
            if isinstance(child, torch.nn.Linear):
                classifier_name = name
                break

    if classifier_name is None:
        raise AttributeError(f"Could not determine classification head name for Hugging Face model {model}")

    backbone_model = TransformersBackboneWrapper(hf_model, classifier_name)

    if default_transform is None:
        processor = AutoImageProcessor.from_pretrained(model, **hub_kwargs)
        default_transform = TransformersProcessorTransform(processor)

    return backbone_model, default_transform
