"""Opt-in optimizer compilation with stable learning-rate inputs."""

from functools import wraps

import torch

from .muon import Muon, MuonAuxAdamW


def _tensor_learning_rates(optimizer):
    for group in optimizer.param_groups:
        if not isinstance(group["lr"], torch.Tensor):
            # Python floats are double precision. Keep that value unchanged,
            # while letting compilation treat scheduler updates as inputs.
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float64)


def _portable_learning_rates(optimizer, state):
    for group in state["param_groups"]:
        if isinstance(group["lr"], torch.Tensor):
            group["lr"] = group["lr"].item()
    return state


def _compile_after_initial_call(optimizer, options):
    step = optimizer.step
    compiled = torch.compile(step, **options)
    initialized = False

    @wraps(step)
    def update(*args, **kwargs):
        nonlocal initialized
        if not initialized:
            # Initialize lazy momentum/state during a real call. Tracing that
            # mutation across multiple groups can fail in Dynamo; never insert
            # a fake update or bypass GradScaler to initialize it.
            result = step(*args, **kwargs)
            _tensor_learning_rates(optimizer)
            optimizer.register_load_state_dict_post_hook(_tensor_learning_rates)
            initialized = True
            return result
        return compiled(*args, **kwargs)

    return update


def compile_optimizer(optimizer, *, backend=None):
    """Compile updates after scheduler construction and checkpoint restoration.

    Keep hooks, scaler overflow decisions and scheduler calls in their usual
    order. Composite Muon counters stay outside compiled child updates. Numeric
    learning rates become scalar tensor inputs, but checkpoint groups retain
    scalar rates so ordinary eager resume does not require this option.
    """
    targets = [getattr(optimizer, name) for name in optimizer.optimizers] if isinstance(optimizer, MuonAuxAdamW) else [optimizer]
    for target in targets:
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
        target.step = _compile_after_initial_call(target, options)
        target._mini_trainer_compiled = True
    return optimizer
