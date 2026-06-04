import warnings
from typing import Any

import torch


class TransformersPreprocessor:
    def __init__(self, preprocessor):
        try:
            from transformers import TorchvisionBackend
        except ImportError as e:
            e.add_note(
                "The `transformers` module was not found in the current Python environment. "
                "Please install with `pip install mini-trainer[transformers]`."
            )
            raise
        assert isinstance(preprocessor, TorchvisionBackend)
        self.preprocessor = preprocessor

    def __call__(self, x) -> torch.Tensor:
        return self.preprocessor(x, return_tensors="pt")["pixel_values"]

    def __repr__(self):
        return f"({self.preprocessor}) -> torch.Tensor"

    def __getattr__(self, item):
        return getattr(self.preprocessor, item)


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
    model: str,
    default_transform: Any = None,
    pretrained: bool = True,
    resize_size: int | None = None,
    **kwargs: Any,
) -> tuple[Any, Any, int | None]:
    """Load Hugging Face transformers classification model and resolve its default transform."""
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
    except ImportError as e:
        e.add_note(
            "The `transformers` module was not found in the current Python environment. "
            "Please install with `pip install mini-trainer[transformers]`."
        )
        raise

    # 1. DRY Fallback with strict exception handling and state immutability
    def _load_with_fallback(hf_class: Any, **load_kwargs: Any) -> Any:
        try:
            return hf_class.from_pretrained(model, **load_kwargs)
        except OSError as e:  # Hugging Face maps network reachability issues to OSError
            if not load_kwargs.get("local_files_only", False):
                warnings.warn(f"Network unreachable for {hf_class.__name__} ('{model}'). Forcing local cache.")

                # Copy kwargs to prevent side-effects on the original reference
                offline_kwargs = load_kwargs.copy()
                offline_kwargs["local_files_only"] = True
                return hf_class.from_pretrained(model, **offline_kwargs)

            # If we are already offline and it throws OSError, the cache is missing/corrupted
            raise e

    # Extract hub kwargs applicable to config and processor loading
    hub_keys = {"local_files_only", "revision", "cache_dir", "force_download", "proxies", "token"}
    hub_kwargs = {k: v for k, v in kwargs.items() if k in hub_keys}

    # --- 1. Model / Config Loading ---
    if pretrained:
        hf_model = _load_with_fallback(AutoModelForImageClassification, **kwargs)
    else:
        config = _load_with_fallback(AutoConfig, **hub_kwargs)
        hf_model = AutoModelForImageClassification.from_config(config)

    # --- 2. Classification Head Resolution ---
    classifier_name = None

    # Semantic Search: Use tuples instead of lists for faster instantiation
    for name in ("classifier", "logits", "head"):
        if hasattr(hf_model, name):
            classifier_name = name
            break

    # Structural Search: Reverse iterate to guarantee we grab the final layer, not an intermediate one
    if classifier_name is None:
        for name, child in reversed(list(hf_model.named_children())):
            if isinstance(child, torch.nn.Linear):
                classifier_name = name
                break

    if classifier_name is None:
        raise AttributeError(f"Could not structurally determine the classification head for {model}.")

    # Assuming TransformersBackboneWrapper is defined elsewhere
    backbone_model = TransformersBackboneWrapper(hf_model, classifier_name)

    # --- 3. Processor Loading ---
    if default_transform is None:
        default_transform = _load_with_fallback(AutoImageProcessor, backend="torchvision", **hub_kwargs)

        # Safe structural type checking
        if resize_size is not None and getattr(default_transform, "size", None):
            if isinstance(default_transform.size, dict):
                for key in ("height", "width", "shortest_edge"):
                    if key in default_transform.size:
                        default_transform.size[key] = resize_size

        # Assuming TransformersPreprocessor is defined elsewhere
        default_transform = TransformersPreprocessor(default_transform)

    # --- 4. Preferred Size Resolution ---
    cfg = getattr(hf_model, "config", None)
    preferred_size = getattr(cfg, "image_size", None) if cfg is not None else None

    if isinstance(preferred_size, (list, tuple)) and len(preferred_size) > 0:
        preferred_size = preferred_size[-1]

    assert isinstance(preferred_size, int) or preferred_size is None

    return backbone_model, default_transform, preferred_size
