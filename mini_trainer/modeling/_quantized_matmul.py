"""Local INT8 matrix kernel tuning without Triton's large L2-flush allocation."""

import torch
import triton
from torchao.prototype.quantized_training.int8_mm import _scaled_int8_mm_kernel as _upstream_kernel
from triton.testing import do_bench_cudagraph


def _benchmark(kernel, quantiles):
    return do_bench_cudagraph(kernel, rep=5, quantiles=quantiles)


# Reuse TorchAO's kernel and candidate configurations, but create a separate
# tuner. Never alter TorchAO's global operator, tuner, or Triton benchmark hooks.
# CUDA graph timing measures repeated device execution without allocating the
# default benchmark's 256 MiB cache-flushing buffer. Persist selected configs so
# a new process need not retune every matrix shape.
_kernel = triton.autotune(configs=_upstream_kernel.configs, key=_upstream_kernel.keys, do_bench=_benchmark, cache_results=True)(
    _upstream_kernel.fn
)


@torch.library.custom_op("mini_trainer::scaled_int8_mm", mutates_args=())
def scaled_int8_mm(left: torch.Tensor, right: torch.Tensor, row_scale: torch.Tensor, column_scale: torch.Tensor) -> torch.Tensor:
    """Compute a scaled INT8 matrix product using locally tuned CUDA kernels."""
    rows, contraction = left.shape
    columns = right.shape[1]
    output = torch.empty((rows, columns), dtype=row_scale.dtype, device=left.device)

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
