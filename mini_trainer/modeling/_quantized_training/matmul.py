"""Local INT8 matrix kernel tuning without Triton's large L2-flush allocation."""

import torch
import triton
import triton.language as tl
from torchao.prototype.quantized_training.int8_mm import _scaled_int8_mm_kernel as _upstream_kernel
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources, PTXASError
from triton.testing import do_bench_cudagraph


def _benchmark(kernel, quantiles):
    try:
        return do_bench_cudagraph(kernel, rep=5, quantiles=quantiles)
    except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as error:
        # Triton rejects these candidates with infinite timing. Release their
        # traceback frames here: they can retain temporary tensors until GC,
        # beyond the lifetime tracked by the surrounding model CUDA graph.
        error.__traceback__ = None
        return [float("inf")] * len(quantiles)


# Reuse TorchAO's kernel and candidate configurations, but create a separate
# tuner. Never alter TorchAO's global operator, tuner, or Triton benchmark hooks.
# CUDA graph timing measures repeated device execution without allocating the
# default benchmark's 256 MiB cache-flushing buffer. Persist selected configs so
# a new process need not retune every matrix shape.
_kernel = triton.autotune(configs=_upstream_kernel.configs, key=_upstream_kernel.keys, do_bench=_benchmark, cache_results=True)(
    _upstream_kernel.fn
)


@triton.jit
def _wide_kernel(
    A,
    B,
    C,
    ROW,
    COL,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SCALAR_COL: tl.constexpr,
    BM: tl.constexpr = 16,
    BN_TILE: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 128,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN_TILE + tl.arange(0, BN_TILE)
    inner = tl.arange(0, BLOCK_K)
    total = tl.zeros((BM, BN_TILE), tl.int64)
    partial = tl.zeros((BM, BN_TILE), tl.int32)
    # A product is at most (-128)*(-128). Flush before INT32 can saturate.
    safe_blocks: tl.constexpr = ((2**31 - 1) // (128 * 128)) // BLOCK_K
    for block in range(tl.cdiv(K, BLOCK_K)):
        k = block * BLOCK_K + inner
        a = tl.load(A + rows[:, None] * AM + k[None, :] * AK, (rows[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + k[:, None] * BK + cols[None, :] * BN, (k[:, None] < K) & (cols[None, :] < N), 0)
        partial = tl.dot(a, b, partial, out_dtype=tl.int32)
        if (block + 1) % safe_blocks == 0:
            total += partial.to(tl.int64)
            partial = tl.zeros((BM, BN_TILE), tl.int32)
    total += partial.to(tl.int64)
    row_scale = tl.load(ROW + rows, rows < M, 0)
    col_scale = tl.load(COL + (tl.zeros((BN_TILE,), tl.int32) if SCALAR_COL else cols), cols < N, 0)
    result = total.to(tl.float32) * row_scale[:, None].to(tl.float32) * col_scale[None, :].to(tl.float32)
    tl.store(C + rows[:, None] * N + cols[None, :], result, (rows[:, None] < M) & (cols[None, :] < N))


@torch.library.custom_op("mini_trainer::scaled_int8_mm", mutates_args=())
def scaled_int8_mm(left: torch.Tensor, right: torch.Tensor, row_scale: torch.Tensor, column_scale: torch.Tensor) -> torch.Tensor:
    """Compute a scaled INT8 matrix product using locally tuned CUDA kernels."""
    rows, contraction = left.shape
    columns = right.shape[1]
    output = torch.empty((rows, columns), dtype=row_scale.dtype, device=left.device)
    if contraction > (2**31 - 1) // (128 * 128):
        with torch.cuda.device(left.device):
            _wide_kernel[(triton.cdiv(rows, 16), triton.cdiv(columns, 64))](
                left,
                right,
                output,
                row_scale,
                column_scale,
                rows,
                columns,
                contraction,
                *left.stride(),
                *right.stride(),
                SCALAR_COL=column_scale.numel() == 1,
            )
        return output

    def grid(meta):
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),)

    with torch.cuda.device(left.device):
        _kernel[grid](
            left,
            right,
            output,
            row_scale,
            column_scale,
            rows,
            columns,
            contraction,
            *left.stride(),
            *right.stride(),
            *output.stride(),
            COL_SCALE_SCALAR=column_scale.numel() == 1,
        )
    return output


@scaled_int8_mm.register_fake
def _fake_scaled_int8_mm(left, right, row_scale, column_scale):
    return torch.empty((left.shape[0], right.shape[1]), dtype=row_scale.dtype, device=left.device)
