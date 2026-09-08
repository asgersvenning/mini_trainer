"""Opt-in CUDA INT8 weight/activation training for ordinary linear modules.

Prepare before constructing an optimizer. Biases, normalization, convolutions,
and unselected weights remain floating point and are reported explicitly.
"""

import torch
from torch import nn


def _backend():
    try:
        from . import _quantized_training
    except ImportError as error:
        raise ImportError("CUDA INT8 training requires mini_trainer[quantization] and a compatible CUDA/Triton installation.") from error
    return _quantized_training


class IntegerLinear:
    """Lazy access to the backend autograd function for numerical validation."""

    @staticmethod
    def apply(inputs, weight):
        return _backend().IntegerLinear.apply(inputs, weight)


def prepare_quantized_training(model: nn.Module, *, module_names=None) -> dict:
    """Replace eligible linear weights in place and return the coverage recipe.

    No floating master weights are retained. Recreate optimizers after this call.
    CPU preparation supports checkpoint inspection; execution requires CUDA.
    Explicitly selected unsupported modules raise instead of silently skipping.
    Parametrized weights and weights shared with unselected operations remain
    floating point until their quantized training contracts are implemented.
    """
    backend = _backend()
    modules = dict(model.named_modules(remove_duplicate=False))
    requested = None if module_names is None else set([module_names] if isinstance(module_names, str) else module_names)
    if requested is not None and requested - modules.keys():
        raise ValueError(f"Unknown quantized training modules: {sorted(requested - modules.keys())}")
    candidates, skipped = {}, {}
    for name, module in modules.items():
        if requested is not None and name not in requested:
            continue
        reason = None
        if not isinstance(module, nn.Linear):
            if requested is not None or isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                reason = "integer training is currently implemented for Linear"
            else:
                continue
        elif nn.utils.parametrize.is_parametrized(module, "weight"):
            reason = "parametrized weights require a separate quantized gradient contract"
        elif not isinstance(module.weight, nn.Parameter) or module.weight.numel() == 0:
            reason = "requires a nonempty weight Parameter"
        elif module.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            reason = "requires float16, bfloat16 or float32 compute metadata"
        if reason:
            skipped[name] = reason
        else:
            candidates[name] = (module, module.weight)
    selected_owners = {(id(module), "weight") for module, _ in candidates.values()}
    owners = {}
    for module in modules.values():
        for name, parameter in module.named_parameters(recurse=False):
            owners.setdefault(id(parameter), set()).add((id(module), name))
    for name, (_, parameter) in list(candidates.items()):
        if owners[id(parameter)] - selected_owners:
            skipped[name] = "weight is shared with an unselected operation"
            del candidates[name]
    if requested is not None and skipped:
        raise ValueError(f"Unsupported quantized training selection: {skipped}")
    if not candidates:
        raise ValueError(f"No eligible Linear weights for quantized training. Unsupported modules: {skipped}")
    # Validate before mutating the caller's model.
    for _, parameter in candidates.values():
        values = parameter.dequantize() if isinstance(parameter, backend.TrainingWeight) else parameter
        if not torch.isfinite(values).all():
            raise ValueError("Quantized training requires finite initial weights.")
    replacements = {}
    for module, parameter in candidates.values():
        if id(parameter) not in replacements:
            replacements[id(parameter)] = (
                parameter
                if isinstance(parameter, backend.TrainingWeight)
                else nn.Parameter(backend.TrainingWeight.from_float(parameter), requires_grad=parameter.requires_grad)
            )
        module.weight = replacements[id(parameter)]
    for module in modules.values():
        invalidate = getattr(module, "_on_quantized_training_prepared", None)
        if invalidate is not None:
            invalidate()
    report = {
        "schema_version": 1,
        "backend": "cuda-int8-linear",
        "quantized_modules": sorted(candidates),
        "skipped_modules": skipped,
        "floating_parameter_names": [
            name for name, parameter in model.named_parameters() if not isinstance(parameter, backend.TrainingWeight)
        ],
        "weight_bits": 8,
        "saved_linear_input_bits": 8,
        "floating_master_weights": False,
        "quantized_weight_bytes": sum(
            weight.int_data.numel() + weight.scale.numel() * weight.scale.element_size() for weight in replacements.values()
        ),
        "reference_weight_bytes": sum(weight.numel() * weight.element_size() for weight in replacements.values()),
    }
    model._quantized_training_recipe = report
    if not getattr(model, "_quantized_training_hooks", False):
        model.register_state_dict_post_hook(_store_recipe)
        model.register_load_state_dict_pre_hook(_consume_recipe)
        model._quantized_training_hooks = True
    return report


def _store_recipe(module, state_dict, prefix, local_metadata):
    import copy

    state_dict[prefix + "_quantized_training"] = copy.deepcopy(module._quantized_training_recipe)


def _consume_recipe(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    recipe = state_dict.pop(prefix + "_quantized_training", None)
    expected = module._quantized_training_recipe
    if recipe is None or any(recipe.get(key) != expected[key] for key in ("schema_version", "backend", "quantized_modules")):
        error_msgs.append("Quantized training checkpoint recipe does not match the prepared model.")


def restore_quantized_training(model: nn.Module, state_dict: dict) -> None:
    """Restore quantized parameter types before loading a model state dictionary."""
    for key, recipe in state_dict.items():
        if key == "_quantized_training" or key.endswith("._quantized_training"):
            if not isinstance(recipe, dict) or recipe.get("schema_version") != 1 or recipe.get("backend") != "cuda-int8-linear":
                raise ValueError("Unsupported quantized training checkpoint recipe.")
            target = model if key == "_quantized_training" else model.get_submodule(key.removesuffix("._quantized_training"))
            prepare_quantized_training(target, module_names=recipe["quantized_modules"])


def load_training_weights(path, *, map_location="cpu"):
    """Load ordinary or INT8 model weights using a scoped known-class allowlist."""
    import pickle

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as error:
        name = "mini_trainer.modeling._quantized_training.TrainingWeight"
        if name not in str(error):
            raise
        with torch.serialization.safe_globals([_backend().TrainingWeight]):
            return torch.load(path, map_location=map_location, weights_only=True)
