"""CUDA row-wise INT8 updates without a floating master weight matrix."""

import torch
import triton
import triton.language as tl


@triton.jit
def _update_rows(
    codes,
    scales,
    update,
    denominator,
    seed,
    alpha,
    columns,
    code_row_stride,
    code_column_stride,
    scale_stride,
    update_row_stride,
    update_column_stride,
    denominator_row_stride,
    denominator_column_stride,
    DIVIDE: tl.constexpr,
    COPY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < columns
    code_offset = row * code_row_stride + column * code_column_stride
    dtype = scales.dtype.element_ty
    change = tl.load(update + row * update_row_stride + column * update_column_stride, valid, 0).to(tl.float32)
    if COPY:
        values = change
    else:
        old_codes = tl.load(codes + code_offset, valid, 0).to(tl.float32)
        old_scale = tl.load(scales + row * scale_stride)
        represented = (old_codes * old_scale.to(tl.float32)).to(dtype).to(tl.float32)
        if DIVIDE:
            divisor = tl.load(denominator + row * denominator_row_stride + column * denominator_column_stride, valid, 1).to(tl.float32)
            change = (change / divisor).to(dtype).to(tl.float32)
        # Muon produces BF16 updates for FP32 parameters. Match the eager
        # update * alpha rounding before promotion in the weight addition.
        change = (change * alpha).to(update.dtype.element_ty).to(tl.float32)
        values = (represented + change).to(dtype).to(tl.float32)
    maximum = tl.max(tl.where(valid, tl.abs(values), 0), 0)
    next_scale = (maximum / 127).to(dtype)
    inverse = 1.0 / tl.maximum(next_scale.to(tl.float32), 1.0e-12)
    random = tl.rand(tl.load(seed), row * columns + column)
    next_codes = tl.floor(values * inverse + random)
    next_codes = tl.minimum(tl.maximum(next_codes, -128), 127).to(tl.int8)
    tl.store(codes + code_offset, next_codes, valid)
    tl.store(scales + row * scale_stride, next_scale)


@torch.library.custom_op("mini_trainer::update_int8_rows_", mutates_args=("codes", "scales"))
def update_int8_rows_(
    codes: torch.Tensor, scales: torch.Tensor, update: torch.Tensor, alpha: torch.Tensor, denominator: torch.Tensor | None
) -> None:
    """Mutate represented weights; fake execution never launches a CUDA kernel."""
    # Optimizer learning rates normally live on CPU. Pass their numeric value
    # as a runtime kernel argument, without a device allocation per weight.
    coefficient = alpha.item()
    _launch(codes, scales, update, coefficient, denominator, copy=False)


def _launch(codes, scales, update, coefficient, denominator, *, copy):
    seed = torch.randint(0, 2**31, (), device=codes.device, dtype=torch.int64)
    divisor = update if denominator is None else denominator
    with torch.cuda.device(codes.device):
        _update_rows[(codes.shape[0],)](
            codes,
            scales,
            update,
            divisor,
            seed,
            coefficient,
            codes.shape[1],
            *codes.stride(),
            scales.stride(0),
            *update.stride(),
            *divisor.stride(),
            DIVIDE=denominator is not None,
            COPY=copy,
            BLOCK=triton.next_power_of_2(codes.shape[1]),
            enable_fp_fusion=False,
        )


@update_int8_rows_.register_fake
def _fake_update_int8_rows_(codes, scales, update, alpha, denominator):
    return None


@torch.library.custom_op("mini_trainer::quantize_int8_rows", mutates_args=(), tags=(torch.Tag.nondeterministic_seeded,))
def quantize_int8_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize updates without hiding mutations from the graph scheduler."""
    codes = torch.empty(values.shape, device=values.device, dtype=torch.int8)
    scales = torch.empty(values.shape[0], device=values.device, dtype=values.dtype)
    _launch(codes, scales, values, 0.0, None, copy=True)
    return codes, scales


@quantize_int8_rows.register_fake
def _fake_quantize_int8_rows(values):
    return torch.empty(values.shape, device=values.device, dtype=torch.int8), torch.empty(
        values.shape[0], device=values.device, dtype=values.dtype
    )
