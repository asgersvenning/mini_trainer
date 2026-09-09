"""Opt-in CUDA INT8 weight/activation training for linear modules.

Prepare before constructing an optimizer. Biases, activation normalization, convolutions,
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
    Row-wise weight normalization retains a floating magnitude and quantizes its
    direction. Other parametrizations and weights shared with unselected
    operations remain floating point.
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
        owner, attribute = module, "weight"
        if not isinstance(module, nn.Linear):
            if requested is not None or isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                reason = "integer training is currently implemented for Linear"
            else:
                continue
        elif nn.utils.parametrize.is_parametrized(module, "weight"):
            parametrizations = module.parametrizations.weight
            if (
                len(parametrizations) == 1
                and type(parametrizations[0]) is nn.utils.parametrizations._WeightNorm
                and parametrizations[0].dim == 0
            ):
                owner, attribute = parametrizations, "original1"
            else:
                reason = "only row-wise weight normalization is supported among parametrized weights"
        if reason is None and (not isinstance(getattr(owner, attribute), nn.Parameter) or getattr(owner, attribute).numel() == 0):
            reason = "requires a nonempty weight Parameter"
        elif reason is None and getattr(owner, attribute).dtype not in (torch.float16, torch.bfloat16, torch.float32):
            reason = "requires float16, bfloat16 or float32 compute metadata"
        if reason:
            skipped[name] = reason
        else:
            candidates[name] = (owner, attribute, getattr(owner, attribute))
    selected_owners = {(id(owner), attribute) for owner, attribute, _ in candidates.values()}
    owners = {}
    for module in modules.values():
        for name, parameter in module.named_parameters(recurse=False):
            owners.setdefault(id(parameter), set()).add((id(module), name))
    for name, (_, _, parameter) in list(candidates.items()):
        if owners[id(parameter)] - selected_owners:
            skipped[name] = "weight is shared with an unselected operation"
            del candidates[name]
    if requested is not None and skipped:
        raise ValueError(f"Unsupported quantized training selection: {skipped}")
    if not candidates:
        raise ValueError(f"No eligible Linear weights for quantized training. Unsupported modules: {skipped}")
    # Validate before mutating the caller's model.
    for owner, attribute, parameter in candidates.values():
        values = parameter.dequantize() if isinstance(parameter, backend.TrainingWeight) else parameter
        if not torch.isfinite(values).all():
            raise ValueError("Quantized training requires finite initial weights.")
        if attribute == "original1":
            if not torch.isfinite(owner.original0).all() or (values.float().norm(dim=1) == 0).any():
                raise ValueError("Quantized weight normalization requires finite magnitudes and nonzero directions.")
    replacements = {}
    for owner, attribute, parameter in candidates.values():
        if id(parameter) not in replacements:
            replacements[id(parameter)] = (
                parameter
                if isinstance(parameter, backend.TrainingWeight)
                else nn.Parameter(backend.TrainingWeight.from_float(parameter), requires_grad=parameter.requires_grad)
            )
        setattr(owner, attribute, replacements[id(parameter)])
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
            restored = prepare_quantized_training(target, module_names=recipe["quantized_modules"])
            # Explicit restoration selects only recorded weights, so keep the
            # original reasons why other modules were not selected.
            restored["skipped_modules"] = dict(recipe.get("skipped_modules", {}))


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


def materialize_quantized_training_state(state_dict: dict, *, dtype=torch.float32):
    """Create independent floating deployment weights from native INT8 state.

    This removes dynamic activation quantization and is not a training-resume
    conversion or a promise of native forward parity. Normalized directions use
    signed integer codes; zero scales require zero magnitudes. Incompatible tied
    parameter roles fail explicitly. Caller tensors and recipes remain untouched.
    """
    import copy
    from collections import OrderedDict

    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("Materialization requires a floating dtype")
    recipes = {key: value for key, value in state_dict.items() if key == "_quantized_training" or key.endswith("._quantized_training")}
    weights = {key: value for key, value in state_dict.items() if getattr(value, "_is_quantized_training", False)}
    if not recipes or not weights:
        raise ValueError("Expected native INT8 training weights and their recipes")
    backend = _backend()
    selected = set()
    for key, recipe in recipes.items():
        if not isinstance(recipe, dict) or recipe.get("schema_version") != 1 or recipe.get("backend") != "cuda-int8-linear":
            raise ValueError("Unsupported quantized training recipe for materialization")
        prefix = key.removesuffix("_quantized_training")
        for module in recipe["quantized_modules"]:
            base = prefix + module + ("." if module else "")
            matches = {base + "weight", base + "parametrizations.weight.original1"} & weights.keys()
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous native weight for {base}")
            selected.update(matches)
    if selected != weights.keys() or any(not isinstance(value, backend.TrainingWeight) for value in weights.values()):
        raise ValueError("Native weights do not match the recorded training recipes")
    result = OrderedDict((key, copy.deepcopy(value)) for key, value in state_dict.items() if key not in recipes and key not in weights)
    if hasattr(state_dict, "_metadata"):
        result._metadata = copy.deepcopy(state_dict._metadata)
    converted, touched = [], set(weights)
    for key, weight in weights.items():
        codes, scales = weight.int_data, weight.scale
        if (
            codes.ndim != 2
            or not codes.numel()
            or codes.dtype != torch.int8
            or scales.shape != (codes.shape[0],)
            or not torch.isfinite(scales).all()
        ):
            raise ValueError(f"Invalid native weight representation: {key}")
        values = codes.detach().to(dtype=dtype)
        normalized = key.endswith("parametrizations.weight.original1")
        if normalized:
            magnitude_key = key.removesuffix("original1") + "original0"
            magnitude = result.get(magnitude_key)
            if not isinstance(magnitude, torch.Tensor) or magnitude.shape != (codes.shape[0], 1) or not torch.isfinite(magnitude).all():
                raise ValueError(f"Invalid normalization magnitude: {magnitude_key}")
            if (torch.linalg.vector_norm(values.float(), dim=1) == 0).any():
                raise ValueError(f"Native normalization has an undefined zero-code direction: {key}")
            signs = scales.sign().to(dtype=dtype)
            values.mul_(torch.where(signs == 0, 1, signs).view(-1, 1))
            result[magnitude_key] = magnitude.to(dtype=dtype) * (signs != 0).view(-1, 1)
            touched.add(magnitude_key)
            if not torch.isfinite(result[magnitude_key]).all():
                raise ValueError(f"Normalization magnitude overflows the materialization dtype: {key}")
        else:
            values.mul_(scales.to(dtype=dtype).view(-1, 1))
        if not torch.isfinite(values).all():
            raise ValueError(f"Represented weights overflow the materialization dtype: {key}")
        result[key] = values
        converted.append({"name": key, "shape": list(values.shape), "normalized": normalized, "source_compute_dtype": str(weight.dtype)})
    # State loading can re-tie parameters through the architecture constructor.
    # Different converted values for the same source storage would silently make
    # the last loaded alias override another operation's intended representation.
    aliases = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or not value.numel():
            continue
        tensors = (value.int_data, value.scale) if key in weights else (value,)
        storage = tuple((str(t.device), t.untyped_storage().data_ptr()) for t in tensors)
        aliases.setdefault(storage, []).append(key)
    for group in aliases.values():
        if len(group) < 2 or not touched.intersection(group):
            continue
        first = group[0]
        for key in group[1:]:
            a, b = state_dict[first], state_dict[key]
            views = (a.int_data, b.int_data) if first in weights and key in weights else (a, b)
            if (
                views[0].shape != views[1].shape
                or views[0].stride() != views[1].stride()
                or views[0].storage_offset() != views[1].storage_offset()
                or not torch.equal(result[first], result[key])
            ):
                raise ValueError(f"Materialization cannot preserve tied parameter roles/views: {group}")
    return result, {
        "schema_version": 1,
        "source_backend": "cuda-int8-linear",
        "target": "floating_weights_for_deployment_calibration",
        "dtype": str(dtype),
        "normalization_parameterization": "signed codes; zero magnitude for zero scales",
        "converted_weights": converted,
        "source_recipes": copy.deepcopy(recipes),
        "dynamic_activation_quantization_preserved": False,
        "training_resume_supported": False,
    }
