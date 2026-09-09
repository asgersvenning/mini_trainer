"""Opt-in model and optimizer compilation."""

from functools import partial, wraps

import torch

from .muon import Muon, MuonAuxAdamW

MODEL_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


def model_compile_options(enabled: bool, mode: str | None) -> dict:
    """Validate explicit model modes while preserving ordinary compile defaults."""
    if mode is None:
        return {}
    if not enabled:
        raise ValueError("compile_mode requires compile=True (--compile).")
    if mode not in MODEL_COMPILE_MODES:
        raise ValueError(f"Unknown model compile mode {mode!r}; choose from {MODEL_COMPILE_MODES}.")
    return {"mode": mode}


def validate_optimizer_compilation(enabled: bool, cudagraphs: bool, device=None) -> None:
    if cudagraphs and not enabled:
        raise ValueError("optimizer_cudagraphs requires compile_optimizer=True (--compile-optimizer).")
    if cudagraphs and device is not None and torch.device(device).type != "cuda":
        raise ValueError("Optimizer CUDA graphs require CUDA.")


def _saved_rate_dtype(group):
    name = group.get("_mini_trainer_lr_dtype", "float64")
    dtype = getattr(torch, name, None) if isinstance(name, str) else None
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Invalid saved learning-rate dtype: {name!r}.")
    return dtype


def _tensor_learning_rates(optimizer, *, cudagraphs=False):
    for group in optimizer.param_groups:
        if not isinstance(group["lr"], torch.Tensor):
            # Python floats are double precision. Keep that value unchanged,
            # while letting compilation treat scheduler updates as inputs.
            group["lr"] = torch.tensor(group["lr"], dtype=_saved_rate_dtype(group))
        group.pop("_mini_trainer_lr_dtype", None)
        if cudagraphs and group["params"]:
            # Preserve explicitly supplied tensor precision. Only numeric rates
            # need the float64 construction above; graph inputs stay on device.
            group["lr"] = group["lr"].to(group["params"][0].device)


def _portable_learning_rates(optimizer, state):
    for group in state["param_groups"]:
        if isinstance(group["lr"], torch.Tensor):
            if group["lr"].dtype != torch.float64:
                group["_mini_trainer_lr_dtype"] = str(group["lr"].dtype).removeprefix("torch.")
            else:
                group.pop("_mini_trainer_lr_dtype", None)
            group["lr"] = group["lr"].item()
    return state


def _compile_after_initial_call(optimizer, options, *, cudagraphs=False):
    step = optimizer.step
    compiled = torch.compile(step, **options)
    initialized = False
    normalize_rates = partial(_tensor_learning_rates, cudagraphs=cudagraphs)

    @wraps(step)
    def update(*args, **kwargs):
        nonlocal initialized
        if not initialized:
            # Initialize lazy momentum/state during a real call. Tracing that
            # mutation across multiple groups can fail in Dynamo; never insert
            # a fake update or bypass GradScaler to initialize it.
            # A restored explicitly typed rate must already have its original
            # representation during this real eager update, not only afterward.
            if any("_mini_trainer_lr_dtype" in group for group in optimizer.param_groups):
                normalize_rates(optimizer)
            result = step(*args, **kwargs)
            normalize_rates(optimizer)
            optimizer.register_load_state_dict_post_hook(normalize_rates)
            initialized = True
            return result
        return compiled(*args, **kwargs)

    return update


def compile_optimizer(optimizer, *, backend=None, cudagraphs=False):
    """Compile updates after scheduler construction and checkpoint restoration.

    Keep hooks, scaler overflow decisions and scheduler calls in their usual
    order. Composite Muon counters stay outside compiled child updates. Numeric
    learning rates become scalar tensor inputs, but checkpoint groups retain
    scalar rates so ordinary eager resume does not require this option.
    Optional CUDA graphs move learning-rate tensors to the parameter device;
    capture eligibility and benefits depend on the optimizer and workload.
    """
    targets = [getattr(optimizer, name) for name in optimizer.optimizers] if isinstance(optimizer, MuonAuxAdamW) else [optimizer]
    if cudagraphs:
        devices = {p.device for target in targets for group in target.param_groups for p in group["params"]}
        if backend is not None or len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("Optimizer CUDA graphs require the default Inductor backend and parameters on one CUDA device.")
    for target in targets:
        for group in target.param_groups:
            _saved_rate_dtype(group)
        if getattr(target, "_mini_trainer_compiled", False) and getattr(target, "_mini_trainer_optimizer_cudagraphs", False) != cudagraphs:
            raise ValueError("Optimizer is already compiled with a different CUDA graph setting.")
        if (
            cudagraphs
            and isinstance(target, (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW))
            and any(
                group.get("fused")
                and not (
                    isinstance(group["lr"], torch.Tensor)
                    and group["lr"].dtype == torch.float32
                    or not isinstance(group["lr"], torch.Tensor)
                    and group.get("_mini_trainer_lr_dtype") == "float32"
                )
                for group in target.param_groups
            )
        ):
            raise ValueError(
                "Native fused optimizer CUDA graphs require an explicit float32 tensor learning rate. "
                "Supply that rate when constructing the optimizer, or leave optimizer CUDA graphs disabled; "
                "numeric/float64 rates are not silently rounded."
            )
        if isinstance(target, (torch.optim.Adam, torch.optim.AdamW)) and any(
            group.get("foreach") and not group.get("capturable") for group in target.param_groups
        ):
            raise ValueError(
                "Optimizer compilation with explicit foreach Adam/AdamW requires capturable=True; "
                "use foreach=False (or its default) for non-capturable updates."
            )
    for target in targets:
        if getattr(target, "_mini_trainer_compiled", False):
            continue
        target.register_state_dict_post_hook(_portable_learning_rates)
        options = {} if backend is None else {"backend": backend}
        if backend is None and isinstance(target, Muon):
            # Newton-Schulz intentionally rounds intermediate values to BF16.
            # Preserve those casts instead of silently changing its iteration.
            options["options"] = {"emulate_precision_casts": True}
        if cudagraphs:
            options.setdefault("options", {})["triton.cudagraphs"] = True
        target.step = _compile_after_initial_call(target, options, cudagraphs=cudagraphs)
        target._mini_trainer_compiled = True
        target._mini_trainer_optimizer_cudagraphs = cudagraphs
    return optimizer
