"""Optional CUDA INT8 training backend. Import through quantized_training.

Changes are local to this subclass, never TorchAO's global dispatch.
"""

import hashlib
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight, quantize_int8_rowwise

from ._quantized_matmul import scaled_int8_mm as _native_scaled_int8_mm
from ._quantized_normalization import int8_weight_norm_backward
from ._quantized_update import quantize_int8_rows, update_int8_rows_

# Tensor-subclass dispatch hides custom autograd bodies from AOT's ordinary
# graph key. Include the backend and kernel implementations so changing hidden
# backward math or operator decomposition cannot reuse an earlier graph.
# Compute once when importing.
_IMPLEMENTATION_HASH = hashlib.sha256(
    b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in ("_quantized_training.py", "_quantized_matmul.py", "_quantized_update.py", "_quantized_normalization.py")
    )
).hexdigest()


class TrainingWeight(Int8QuantizedTrainingLinearWeight):
    """INT8 storage with integer linear arithmetic and ordinary optimizer updates."""

    _is_quantized_training = True

    def dequantize(self):
        return _Dequantize.apply(self)

    def _stable_hash_for_caching(self):
        metadata = [
            (tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device), value.requires_grad)
            for value in (self, self.int_data, self.scale)
        ]
        return hashlib.sha256(repr((_IMPLEMENTATION_HASH, metadata)).encode()).hexdigest()


@TrainingWeight.implements_torch_function(torch.nn.functional.linear)
def linear(func, types, args, kwargs):
    inputs = args[0] if args else kwargs["input"]
    weight = args[1] if len(args) > 1 else kwargs["weight"]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    if inputs.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        inputs = inputs.to(torch.get_autocast_dtype("cuda"))
    output = IntegerLinear.apply(inputs.reshape(-1, inputs.shape[-1]), weight)
    output = output.reshape(*inputs.shape[:-1], weight.shape[0])
    return output if bias is None else output + bias.to(output.dtype)


@TrainingWeight.implements([torch.ops.aten.detach.default, torch.ops.aten.clone.default])
def preserve_type(func, types, args, kwargs):
    original = args[0]
    if func == torch.ops.aten.detach.default:
        # Detach aliases the original version counter. Constructing its wrapper
        # under inference_mode would discard that counter for normal weights.
        with torch.inference_mode(original.is_inference()):
            out = TrainingWeight(func(original.int_data, **kwargs), func(original.scale, **kwargs))
    else:
        out = TrainingWeight(func(original.int_data, **kwargs), func(original.scale, **kwargs))
    return return_and_correct_aliasing(func, args, kwargs, out)


@TrainingWeight.implements(torch.ops.aten._to_copy.default)
def to_copy(func, types, args, kwargs):
    original = args[0]
    integer_kwargs = {key: value for key, value in kwargs.items() if key != "dtype"}
    out = TrainingWeight(func(original.int_data, **integer_kwargs), func(original.scale, **kwargs))
    return return_and_correct_aliasing(func, args, kwargs, out)


@TrainingWeight.implements(torch.ops.aten.add.Tensor)
def add(func, types, args, kwargs):
    # Coupled weight decay (SGD) adds the dequantized parameter to its gradient.
    return func(*(value.dequantize() if isinstance(value, TrainingWeight) else value for value in args), **kwargs)


@TrainingWeight.implements(torch.ops.aten.copy_.default)
def copy_weight(func, types, args, kwargs):
    destination, source = args[:2]
    if not isinstance(destination, Int8QuantizedTrainingLinearWeight):
        destination.copy_(source.dequantize(), **kwargs)
    elif isinstance(source, Int8QuantizedTrainingLinearWeight):
        destination.int_data.copy_(source.int_data, **kwargs)
        destination.scale.copy_(source.scale, **kwargs)
    elif (
        destination.device.type == "cuda"
        and source.device == destination.device
        and source.shape == destination.shape
        and source.dtype == destination.dtype
        and source.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and 0 < source.shape[1] <= 16384
    ):
        codes, scales = quantize_int8_rows(source)
        destination.int_data.copy_(codes, **kwargs)
        destination.scale.copy_(scales, **kwargs)
    else:
        codes, scales = quantize_int8_rowwise(source, stochastic_rounding=True)
        destination.int_data.copy_(codes, **kwargs)
        destination.scale.copy_(scales, **kwargs)
    return destination


@TrainingWeight.implements(torch.ops.prims.fma.default)
def fused_multiply_add(func, types, args, kwargs):
    # Dynamo lowers add_/addcdiv_ with tensor learning rates to fma + copy_.
    # The out-of-place result is floating; copy_ performs the ordinary single
    # stochastic requantization. Leaving fma unsupported breaks optimizer loops
    # into per-weight frames and eventually exhausts the compilation cache.
    return func(*(value.dequantize() if isinstance(value, TrainingWeight) else value for value in args), **kwargs)


@TrainingWeight.implements(torch.ops.aten.mul_.Tensor)
def multiply_inplace(func, types, args, kwargs):
    original, multiplier = args
    # AdamW's decoupled decay is a scalar rescale. Preserve the integer codes
    # exactly instead of adding a second stochastic rounding to every update.
    if isinstance(multiplier, (float, int)) or isinstance(multiplier, torch.Tensor) and multiplier.ndim == 0:
        original.scale.mul_(multiplier)
        return original
    return original.copy_(original.dequantize() * multiplier)


@TrainingWeight.implements(torch.ops.aten._foreach_add.List)
def foreach_add(func, types, args, kwargs):
    return [torch.add(left, right, **kwargs) for left, right in zip(*args, strict=True)]


@TrainingWeight.implements(torch.ops.aten._foreach_add_.List)
def foreach_add_inplace(func, types, args, kwargs):
    for left, right in zip(*args, strict=True):
        left.add_(right, **kwargs)
    return None


@TrainingWeight.implements(torch.ops.aten._foreach_mul_.Scalar)
def foreach_mul_inplace(func, types, args, kwargs):
    for value in args[0]:
        value.mul_(args[1])
    return None


@TrainingWeight.implements([torch.ops.aten._foreach_addcdiv_.Scalar, torch.ops.aten._foreach_addcdiv_.ScalarList])
def foreach_addcdiv_inplace(func, types, args, kwargs):
    factors = args[3] if len(args) > 3 else 1
    for index, (target, numerator, denominator) in enumerate(zip(*args[:3], strict=True)):
        factor = factors[index] if isinstance(factors, (tuple, list)) else factors
        target.addcdiv_(numerator, denominator, value=factor)
    return None


class IntegerLinear(torch.autograd.Function):
    """Row-scaled INT8 GEMMs, including approximate input and weight gradients."""

    @staticmethod
    def forward(ctx, inputs, weight):
        quantize, mm = quantize_int8_rowwise, scaled_int8_mm
        if inputs.device.type != "cuda":
            raise ValueError("INT8 training linear execution requires CUDA.")
        quantized, scale = quantize(inputs.float())
        ctx.weight_dtype = weight.dtype
        ctx.save_for_backward(quantized, scale, weight.int_data, weight.scale.float())
        return mm(quantized.contiguous(), weight.int_data.T, scale.contiguous(), weight.scale.float().contiguous()).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        quantize, mm = quantize_int8_rowwise, scaled_int8_mm
        inputs, input_scale, weight, weight_scale = ctx.saved_tensors
        ones = torch.ones(weight.shape[1], device=grad_output.device, dtype=torch.float32)
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Weight scales lie along the contraction axis: absorb them into
            # dY before its row quantization, not into the result columns.
            quantized_grad, scale = quantize(grad_output.float() * weight_scale.float())
            grad_input = mm(quantized_grad.contiguous(), weight.contiguous(), scale.contiguous(), ones)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # Similarly absorb saved activation scales into dY.T for dW.
            quantized_grad, scale = quantize(grad_output.T.float() * input_scale.float())
            grad_weight = mm(quantized_grad.contiguous(), inputs.contiguous(), scale.contiguous(), ones)
        return (
            grad_input.to(grad_output.dtype) if grad_input is not None else None,
            grad_weight.to(ctx.weight_dtype) if grad_weight is not None else None,
        )


def scaled_int8_mm(left, right, row_scale, column_scale):
    # Validate row scales without squeeze, which would reject a single sample.
    if row_scale.shape != (left.shape[0],) or column_scale.numel() not in (1, right.shape[1]):
        raise ValueError("Invalid INT8 matrix scales.")
    if left.shape[1] != right.shape[0] or row_scale.dtype != column_scale.dtype:
        raise ValueError("Incompatible INT8 matrix shapes or scale dtypes.")
    if left.dtype != torch.int8 or right.dtype != torch.int8:
        raise ValueError("INT8 matrix products require integer inputs.")
    if not row_scale.is_contiguous() or not column_scale.is_contiguous():
        raise ValueError("INT8 matrix scales must be contiguous.")
    return _native_scaled_int8_mm(left, right, row_scale, column_scale)


@TrainingWeight.implements(torch.ops.aten.index_select.default)
def select_weight_rows(func, types, args, kwargs):
    weight, dimension, indices = args
    if dimension not in (0, -2):
        raise ValueError("Quantized training weights support row selection only.")
    return TrainingWeight(weight.int_data.index_select(0, indices), weight.scale.index_select(0, indices))


class _Dequantize(torch.autograd.Function):
    """Expose represented values to floating-point auxiliary losses with STE."""

    @staticmethod
    def forward(ctx, weight):
        return weight.int_data * weight.scale.view(-1, 1)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


@TrainingWeight.implements_torch_function(torch._weight_norm)
def weight_norm(func, types, args, kwargs):
    direction = args[0] if args else kwargs["v"]
    magnitude = args[1] if len(args) > 1 else kwargs["g"]
    dimension = args[2] if len(args) > 2 else kwargs.get("dim", 0)
    if dimension != 0:
        raise ValueError("INT8 weight normalization supports Linear output rows (dim=0) only.")
    return _WeightNorm.apply(direction, magnitude)


class _WeightNorm(torch.autograd.Function):
    """Normalize represented directions without retaining floating weight matrices.

    w = g * v / ||v||. The codes are unchanged; only their row scales change.
    Backward applies the ordinary normalization Jacobian to the approximate dW
    supplied by IntegerLinear. Float intermediates are transient, never masters.
    """

    @staticmethod
    def forward(ctx, direction, magnitude):
        codes, scales = direction.int_data, direction.scale
        norm = torch.linalg.vector_norm(codes.float(), dim=1)
        ctx.save_for_backward(codes, scales, magnitude, norm)
        return TrainingWeight(codes, (scales.sign() * magnitude.flatten().float() / norm).to(scales.dtype))

    @staticmethod
    def backward(ctx, gradient):
        codes, scales, magnitude, norm = ctx.saved_tensors
        if codes.device.type == "cuda" and codes.shape[1] <= 16384 and not torch.is_grad_enabled():
            return int8_weight_norm_backward(codes, scales, magnitude, norm, gradient)
        unit = codes.float() * (scales.sign() / norm).unsqueeze(1)
        projection = (gradient.float() * unit).sum(dim=1, keepdim=True)
        direction_gradient = (gradient.float() - projection * unit) * (magnitude.float() / (scales.float().abs() * norm).unsqueeze(1))
        return direction_gradient.to(scales.dtype), projection.to(magnitude.dtype)


@TrainingWeight.implements_torch_function(torch.Tensor.add_)
def add_update(func, types, args, kwargs):
    original = args[0]
    update = args[1] if len(args) > 1 else kwargs["other"]
    alpha = kwargs.get("alpha", 1)
    # Tensor learning rates must stay tensor inputs. Passing them through the
    # aten alpha scalar overload can specialize dispatch on Python objects.
    return _apply_weight_update(original, update, alpha)


@TrainingWeight.implements_torch_function(torch.Tensor.addcdiv_)
def addcdiv_update(func, types, args, kwargs):
    original = args[0]
    numerator = args[1] if len(args) > 1 else kwargs["tensor1"]
    denominator = args[2] if len(args) > 2 else kwargs["tensor2"]
    value = kwargs.get("value", 1)
    return _apply_weight_update(original, numerator, value, denominator)


def _apply_weight_update(original, update, alpha, denominator=None):
    supported = (
        original.device.type == "cuda"
        and original.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and 0 < original.shape[1] <= 16384
        and isinstance(update, torch.Tensor)
        and update.shape == original.shape
        and update.dtype == original.dtype
        and update.device == original.device
        and (
            denominator is None
            or denominator.shape == original.shape
            and denominator.dtype == original.dtype
            and denominator.device == original.device
        )
    )
    # Expose update math and final copies to compiled graphs. An opaque
    # in-place update can carry CPU scalar inputs across CUDA graph partitions
    # and hides the storage dependencies the compiler needs to schedule safely.
    if is_fake(original) or not supported or torch.is_grad_enabled() and original.requires_grad:
        change = update if denominator is None else update / denominator
        return original.copy_(original.dequantize() + change * alpha)
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.tensor(alpha, dtype=torch.float64)
    update_int8_rows_(original.int_data, original.scale, update, alpha, denominator)
    # The kernel mutates storage tensors directly; also advance the wrapper's
    # version counter for cache invalidation and saved-tensor safety.
    torch.autograd.graph.increment_version(original)
    return original
