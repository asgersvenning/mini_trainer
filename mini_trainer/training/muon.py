"""Muon and a composite Muon/AdamW optimizer.

Adapted from https://github.com/pytorch/pytorch/blob/main/torch/optim/_muon.py.
"""

import math
from collections.abc import Callable, MutableMapping, Sequence
from itertools import chain
from typing import Any

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer, ParamsT, _disable_dynamo_if_unsupported


def _to_scalar(x: float | torch.Tensor):
    """Squeeze tensor hyperparameters; leave non-tensors unchanged."""
    if isinstance(x, torch.Tensor) and x.dim() != 0:
        return x.squeeze()
    else:
        return x


__all__ = ["Muon"]

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
# github permlink: https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float) -> Tensor:
    """Approximate the matrix polar factor with a BF16 quintic iteration.

    The default finite iteration does not force every singular value to one.
    Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py,
    with suggestions by @jxbz, @leloykun and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(Optimizer):
    """Optimize dense real 2D parameters with momentum and Newton-Schulz updates.

    The momentum buffer interpolates toward the gradient with weight
    ``1 - momentum``. Nesterov mode blends the current gradient with that buffer
    before orthogonalization. Decoupled weight decay uses the base learning rate;
    the orthogonalized update uses the shape-adjusted rate.

    ``adjust_lr_fn=None`` or ``"original"`` scales the update rate by
    ``sqrt(max(1, rows / columns))``; ``"match_rms_adamw"`` uses
    ``0.2 * sqrt(max(rows, columns))``. ``ns_coefficients`` and ``ns_steps`` control
    the quintic iteration, and ``eps`` floors its normalization denominator.
    Defaults are in the constructor signature. Use a separate optimizer for
    non-matrix parameters, or ``MuonAuxAdamW`` for automatic routing.

    References:
        https://kellerjordan.github.io/posts/muon/
        https://arxiv.org/pdf/2502.16982
    """

    def __init__(  # noqa: D107
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in [
            "original",
            "match_rms_adamw",
        ]:
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters whereas we found a parameter with size: {p.size()}")

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            if torch.is_complex(p):
                raise RuntimeError("Muon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            muon_momentum_bufs.append(state["momentum_buffer"])

        return False  # has_complex

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                muon_momentum_bufs,
            )

            muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                eps=group["eps"],
                ns_steps=group["ns_steps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)  # type: ignore
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_muon)
def muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    foreach: bool | None = None,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
):
    """Apply this module's Muon update; foreach execution is unsupported."""
    if foreach is not None and foreach:
        raise RuntimeError("Foreach is not supported for Muon yet")

    _single_tensor_muon(
        params,
        grads,
        muon_momentum_bufs,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
        adjust_lr_fn=adjust_lr_fn,
        has_complex=has_complex,
    )


class MuonAuxAdamW(Optimizer):
    """Route named groups to Muon for matrices and AdamW for other parameters.

    A group name containing ``nomuon`` forces all its parameters onto AdamW.
    Exposed parameter groups reference the child groups, allowing schedulers to
    update both optimizers. Checkpoints contain child states keyed by optimizer
    name; the outer step counter tracks successful composite calls, not resume
    history. Default Muon rate adjustment is ``match_rms_adamw``.
    """

    def __init__(self, params: ParamsT, **kwargs):
        self._init = True
        self.opt_args = {"muon": {"adjust_lr_fn": "match_rms_adamw", "momentum": 0.95}, "adamw": {"betas": (0.9, 0.999)}}
        self.opt_cls = {"muon": Muon, "adamw": AdamW}
        super().__init__(params=params, defaults=kwargs)
        orig_groups = list(self.param_groups)
        self.param_groups = []
        self.muon = self.adamw = None
        self._init = False
        self._step_count = 0
        for g in orig_groups:
            self.add_param_group(g)

    @property
    def optimizers(self):
        if self.muon is not None:
            yield "muon"
        if self.adamw is not None:
            yield "adamw"

    @property
    def state(self):
        unified_state = {}
        for opt in self.optimizers:
            unified_state.update(getattr(self, opt).state)
        return unified_state

    @state.setter
    def state(self, value):
        pass

    def _refresh_param_groups(self):
        self.param_groups = list(chain.from_iterable(getattr(self, opt).param_groups for opt in self.optimizers))

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            getattr(self, opt).zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable[[], float | Tensor] | None = None):
        loss = closure() if closure is not None else None
        for opt in self.optimizers:
            getattr(self, opt).step()

        self._step_count += 1

        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._init:
            return super().add_param_group(param_group=param_group)
        # Force AdamW for parameter groups with "nomuon" in name
        if "nomuon" in param_group["name"]:
            grps = {"adamw": param_group}
        else:
            params: Sequence[Tensor] = param_group.pop("params", [])
            if not isinstance(params, Sequence) or len(params) == 0:
                raise ValueError("param_group['params'] must be a non-empty sequence")
            base = {k: v for k, v in param_group.items() if k != "params"}

            grps = {
                "muon": {"params": [p for p in params if p.ndim == 2], **base},
                "adamw": {"params": [p for p in params if p.ndim != 2], **base},
            }
        for name, grp in grps.items():
            if len(grp["params"]) == 0:
                continue
            opt: Optimizer | None = getattr(self, name, None)
            if opt is None:
                setattr(self, name, self.opt_cls[name]([grp], **{**self.opt_args[name], **self.defaults}))  # ty: ignore[invalid-argument-type]
            else:
                opt.add_param_group(grp)
        self._refresh_param_groups()

    def state_dict(self) -> dict[str, Any]:
        return {opt: getattr(self, opt).state_dict() for opt in self.optimizers}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for k, v in state_dict.items():
            opt = getattr(self, k, None)
            if opt is not None:
                opt.load_state_dict(v)
        self._refresh_param_groups()
