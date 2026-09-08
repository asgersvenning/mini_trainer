"""Experimental INT8 parameter dispatch for the CUDA training probe.

Kept outside the runtime package until optimizer, checkpoint and model coverage
are established. Changes are local to this subclass, never TorchAO's dispatch.
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight


class TrainingWeight(Int8QuantizedTrainingLinearWeight):
    """INT8 storage with integer linear arithmetic and ordinary optimizer updates."""


@TrainingWeight.implements_torch_function(torch.nn.functional.linear)
def linear(func, types, args, kwargs):
    from .quantized_training import IntegerLinear

    inputs = args[0] if args else kwargs["input"]
    weight = args[1] if len(args) > 1 else kwargs["weight"]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    output = IntegerLinear.apply(inputs.reshape(-1, inputs.shape[-1]), weight)
    output = output.reshape(*inputs.shape[:-1], weight.shape[0])
    return output if bias is None else output + bias


@TrainingWeight.implements([torch.ops.aten.detach.default, torch.ops.aten.clone.default])
def preserve_type(func, types, args, kwargs):
    original = args[0]
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
