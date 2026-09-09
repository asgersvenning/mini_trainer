"""CUDA normalization gradients without full floating direction intermediates."""

import torch
import triton
import triton.language as tl


@triton.jit
def _backward_rows(
    codes,
    scales,
    magnitude,
    norm,
    gradient,
    direction_gradient,
    magnitude_gradient,
    columns,
    code_row_stride,
    code_column_stride,
    scale_stride,
    magnitude_stride,
    norm_stride,
    gradient_row_stride,
    gradient_column_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < columns
    code = tl.load(codes + row * code_row_stride + column * code_column_stride, valid, 0).to(tl.float32)
    scale = tl.load(scales + row * scale_stride).to(tl.float32)
    length = tl.load(norm + row * norm_stride)
    gain = tl.load(magnitude + row * magnitude_stride).to(tl.float32)
    grad = tl.load(gradient + row * gradient_row_stride + column * gradient_column_stride, valid, 0).to(tl.float32)
    sign = tl.where(scale > 0, 1.0, tl.where(scale < 0, -1.0, 0.0))
    unit = code * (sign / length)
    projection = tl.sum(grad * unit, 0)
    result = (grad - projection * unit) * (gain / (tl.abs(scale) * length))
    tl.store(direction_gradient + row * columns + column, result, valid)
    tl.store(magnitude_gradient + row, projection)


@torch.library.custom_op("mini_trainer::int8_weight_norm_backward", mutates_args=())
def int8_weight_norm_backward(
    codes: torch.Tensor, scales: torch.Tensor, magnitude: torch.Tensor, norm: torch.Tensor, gradient: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the row normalization Jacobian, allocating only the returned gradients."""
    direction_gradient = torch.empty(codes.shape, device=codes.device, dtype=scales.dtype)
    magnitude_gradient = torch.empty(magnitude.shape, device=magnitude.device, dtype=magnitude.dtype)
    with torch.cuda.device(codes.device):
        _backward_rows[(codes.shape[0],)](
            codes,
            scales,
            magnitude,
            norm,
            gradient,
            direction_gradient,
            magnitude_gradient,
            codes.shape[1],
            *codes.stride(),
            scales.stride(0),
            magnitude.stride(0),
            norm.stride(0),
            *gradient.stride(),
            BLOCK=triton.next_power_of_2(codes.shape[1]),
            enable_fp_fusion=False,
        )
    return direction_gradient, magnitude_gradient


@int8_weight_norm_backward.register_fake
def _fake_backward(codes, scales, magnitude, norm, gradient):
    return torch.empty(codes.shape, device=codes.device, dtype=scales.dtype), torch.empty(
        magnitude.shape, device=magnitude.device, dtype=magnitude.dtype
    )
