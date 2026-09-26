import copy
import hashlib
import os
from collections import OrderedDict

import torch
from torch import nn


def average_checkpoints(inputs, map_location=None, weights_only=True):
    """Average tensor entries under each checkpoint's ``model`` key.

    Inputs must be a nonempty path sequence with identical ordered model keys.
    Floating entries use arithmetic means; integer entries use floor division.
    Other checkpoint fields, including optimizer state, come from the first input.
    Non-tensor model metadata is not supported. Loading defaults to CPU;
    map_location and weights_only are forwarded to torch.load.

    Based on:
    https://github.com/pytorch/fairseq/blob/a48f235636557b8d3bc4922a6fa90f3a0fa57955/scripts/average_checkpoints.py#L16
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    if map_location is None:
        map_location = lambda s, _: torch.serialization.default_restore_location(s, "cpu")  # noqa: E731
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(f, map_location=map_location, weights_only=weights_only)
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(f"For checkpoint {f}, expected list of params: {params_keys}, but found: {model_params_keys}")
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """Validate checkpoint weights against a copy of model and save its state dict.

    Load checkpoint_path on CPU with weights_only=True, selecting checkpoint_key
    (default ``model``) and forwarding strict to load_state_dict. For ``model_ema``,
    remove the averaging counter and ``module.`` prefix before loading.

    Return the absolute path to ``weights-<sha256[:8]>.pth`` beside the checkpoint.
    The caller's model is unchanged. With strict=False, missing parameters retain
    the supplied model's values.
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: float | None = None,
    norm_classes: list[type] | None = None,
    custom_keys_weight_decay: list[tuple[str, float]] | None = None,
):
    """Return trainable parameter groups with per-group weight decay.

    Custom keys take precedence over normalization-module and default decay.
    A key containing a dot matches a full parameter path; other keys match local
    parameter names. The first matching custom key wins.
    """
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups


def copy_bn_buffers(src: nn.Module, dst: nn.Module) -> None:  # noqa: D103
    for ms, md in zip(src.modules(), dst.modules()):
        if isinstance(ms, nn.modules.batchnorm._BatchNorm):
            md.running_mean.copy_(ms.running_mean)
            md.running_var.copy_(ms.running_var)
            if hasattr(ms, "num_batches_tracked") and hasattr(md, "num_batches_tracked"):
                md.num_batches_tracked.copy_(ms.num_batches_tracked)
