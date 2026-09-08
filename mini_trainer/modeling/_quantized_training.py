"""Optional CUDA INT8 training backend. Import through quantized_training.

Changes are local to this subclass, never TorchAO's global dispatch.
"""

import hashlib
from pathlib import Path

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight, quantize_int8_rowwise
from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm as _native_scaled_int8_mm

# Tensor-subclass dispatch hides custom autograd bodies from AOT's ordinary
# graph key. Include the backend implementation so changing backward math cannot
# reuse a graph compiled for an earlier version. Compute once when importing.
_IMPLEMENTATION_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


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


@TrainingWeight.implements(torch.ops.aten.mul_.Tensor)
def multiply_inplace(func, types, args, kwargs):
    original, multiplier = args
    # AdamW's decoupled decay is a scalar rescale. Preserve the integer codes
    # exactly instead of adding a second stochastic rounding to every update.
    if isinstance(multiplier, (float, int)):
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
    # TorchAO's Python validation squeezes the row scale, rejecting M=1 even
    # though its native kernel supports it. Validate that case without squeeze.
    if left.shape[0] == 1:
        if row_scale.shape != (1,) or column_scale.shape != (right.shape[1],):
            raise ValueError("Invalid INT8 matrix scales.")
        if left.shape[1] != right.shape[0] or row_scale.dtype != column_scale.dtype:
            raise ValueError("Incompatible INT8 matrix shapes or scale dtypes.")
        return torch.ops.torchao.scaled_int8_mm(left, right, row_scale, column_scale)
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
