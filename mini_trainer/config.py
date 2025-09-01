import importlib
import inspect
import os
import re
import warnings
from copy import deepcopy
from functools import reduce
from typing import Any

import torch


def _nullify(d: dict[str, Any]):
    """
    Recursively replaces all values in a dictionary with None.
    Recurses on nested dictionaries. 
    """
    for k, v in list(d.items()):
        if isinstance(v, dict):
            _nullify(v)
        else:
            d[k] = None
    return d


def _prune_nones(d: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively drop keys with value ``None`` and empty dicts.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = _prune_nones(v)
        if not (v is None or isinstance(v, dict) and len(v) == 0):
            out[k] = v
    return out


def _stringify_types(obj: Any) -> Any:
    """Recursively convert values to YAML/JSON-friendly primitives.

    - torch.device -> str
    - torch.dtype -> canonical lowercase str without 'torch.' prefix
    - type/class -> 'module.QualName'
    - sequences/mappings -> converted recursively
    - arbitrary objects -> class path string
    """

    if isinstance(obj, dict):
        return {k: _stringify_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = [_stringify_types(v) for v in obj]
        return type(obj)(seq) if not isinstance(obj, set) else seq
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj).removeprefix("torch.").strip().lower()
    if isinstance(obj, type):
        return f"{obj.__module__}.{getattr(obj, '__qualname__', getattr(obj, '__name__', 'UnknownType'))}"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback for arbitrary objects (e.g., SummaryWriter): represent by class path
    cls = obj.__class__
    return f"{cls.__module__}.{getattr(cls, '__qualname__', getattr(cls, '__name__', 'UnknownObject'))}"


def dump_resolved_config(
    output_dir: str,
    fn: Any,
    local_vars: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> None:
    """
    Dump a YAML (or JSON fallback) config derived from function parameters.

    Args:
        output_dir: Destination directory for the config file.
        fn: Function whose signature determines which locals are captured.
        local_vars: The locals() mapping from the function at the capture point.
        overrides: Optional mapping to override selected keys (e.g., resolved paths).
    """
    params = inspect.signature(fn).parameters
    cfg: dict[str, Any] = {}
    for name in params:
        if name not in local_vars:
            continue
        cfg[name] = local_vars[name]

    # Apply optional overrides (e.g., resolved paths/objects to strings)
    if overrides:
        cfg.update(overrides)

    # Normalize to dump-friendly forms
    cfg = _stringify_types(cfg)

    # Remove empty arguments
    cfg = _prune_nones(cfg)

    # Write YAML preferred, JSON fallback (then exit)
    os.makedirs(output_dir, exist_ok=True)
    try:
        import yaml
        path_yaml = os.path.join(output_dir, f"{cfg["name"]}_config.yaml")
        with open(path_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return
    except Exception as e:
        import json
        path_json = os.path.join(output_dir, f"{cfg["name"]}_config.json")
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        raise SystemExit(
            f"YAML dump failed ({e!s}). Wrote JSON fallback at: {path_json}"
        )
    
def save_yaml_template(path: str) -> str:
    """
    Write a YAML template for the given function to "path".

    Returns the absolute path written.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "PyYAML is required for YAML config support. Install with 'pip install pyyaml'"
        ) from e

    cfg = _stringify_types(defaults_from_function(getattr(importlib.import_module("mini_trainer.train"), "main")))
    
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """
    Load a YAML config file into a dict of keyword arguments.

    Supports optional "builder" as an import path string.
    """
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "PyYAML is required for YAML config support. Install with 'pip install pyyaml'"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML in {path!r} must be a mapping/object.")

    # Attempt to deserialize any dotted import strings anywhere in the config
    dotted_path = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]+)+$")

    def _resolve_any(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: rv for k, v in obj.items() if (rv := _resolve_any(v)) is not None}
        if isinstance(obj, list):
            return [rv for v in obj if (rv := _resolve_any(v)) is not None]
        if isinstance(obj, tuple):
            return tuple(rv for v in obj if (rv := _resolve_any(v)) is not None)
        if isinstance(obj, set):
            return {rv for v in obj if (rv := _resolve_any(v)) is not None}
        if isinstance(obj, str) and dotted_path.match(obj):
            mod, _, attr = obj.rpartition(".")
            try:
                module = importlib.import_module(mod)
                return getattr(module, attr)
            except Exception as e:
                warnings.warn(f"Config deserialization: could not import '{obj}': {e}")
                return None
        return obj

    out = _resolve_any(data)
    if not isinstance(out, dict):
        warnings.warn(f'! OBS: CONFIG SKIPPED !\nConfig not properly deserialized: {out}')
        return {}
    return out


def defaults_from_function(fn: Any) -> dict[str, Any]:
    """
    Return a dict mapping parameter names to their default values for ``fn``.

    The structure matches the function signature exactly and preserves types
    (e.g., the ``builder`` remains a class/type if that is the default).
    """
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        default = param.default if param.default is not inspect._empty else None
        # Deepcopy containers to avoid accidental mutation across uses
        try:
            out[name] = deepcopy(default)
        except Exception:
            out[name] = default
    return out


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge multiple dictionaries left-to-right. Later dicts win.
    """
    out: dict[str, Any] = {}
    for d in dicts:
        for k, v in d.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = merge_dicts(out[k], v)
            else:
                out[k] = v
    return out


def restructure_cli_args(args: dict[str, Any], fn: Any) -> dict[str, Any]:
    """
    Map flat CLI args into the nested structure expected by ``main``.

    Rules:
    - Skip keys with value ``None`` (treated as not provided).
    - For boolean flags, include only if ``True``.
    - Restructure known flat keys into nested ``*_builder_kwargs`` blocks.
    """
    # Start from the function defaults to inherit the target shape, then null them
    base = defaults_from_function(fn)
    _nullify(base)

    # Top-level keys that map directly
    top_keys = {
        "input", "output", "checkpoint", "class_index", "epochs", "name",
        "device", "dtype", "seed", "builder",
        "spec_model_dataloader_kwargs", "augmentation_builder_kwargs",
        "regularizer_builder_kwargs", "logger_builder_kwargs",
    }
    base.update({k : v for k in top_keys if (v := args.get(k, None)) is not None})

    # Helpers for setting nested values
    def _inset(d : dict, loc : list[str], value : Any):
        sub_d = d
        for k in loc[:-1]:
            sub_d.setdefault(k, {})
            sub_d = sub_d[k]
        sub_d[loc[-1]] = value

    def set_nested(path: list[str] | str, *keys: dict[str, str] | list[str]):
        if isinstance(path, str):
            path = [path]
        keys = reduce(
            lambda d1, d2 : {**d1, **d2}, 
            [{k : k for k in ks} if isinstance(ks, (list, set, tuple)) else ks for ks in keys]
        )
        
        for k, v in [(k, args.get(_k, None)) for _k, k in keys.items()]:
            if v is None:
                continue
            _inset(base, path + [k], v)

    # Mappings for nested kwargs
    nested_keys = [
        ("model_builder_kwargs", {"model" : "model_type"}, ["weights", "fine_tune"]),
        ("dataloader_builder_kwargs", ["data_index", "batch_size", "subsample", "num_workers", "cache"]),
        ("optimizer_builder_kwargs", ["lr"]),
        ("criterion_builder_kwargs", ["label_smoothing"], {"class_weighted" : "weighted"}),
        ("logger_builder_kwargs", ["verbose"])
    ]
    [set_nested(*nk) for nk in nested_keys]
    
    return _prune_nones(base)
