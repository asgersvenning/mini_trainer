"""Optional ONNX lowering of the native QT forward's scaled integer product."""

import onnx
from onnxscript import FLOAT, INT8
from onnxscript import opset18 as op


def scaled_int8_mm(left: INT8, right: INT8, row_scale: FLOAT, column_scale: FLOAT) -> FLOAT:
    # Preserve signed codes exactly while using the U8/S8 CPU MatMulInteger path.
    # Replacing the row quantizer with DynamicQuantizeLinear would change its
    # symmetric range, rounding and zero-row behavior.
    offset = op.Constant(value=onnx.helper.make_tensor("offset", onnx.TensorProto.INT32, [], [128]))
    zero = op.Constant(value=onnx.helper.make_tensor("zero", onnx.TensorProto.UINT8, [], [128]))
    unsigned = op.Cast(op.Add(op.Cast(left, to=onnx.TensorProto.INT32), offset), to=onnx.TensorProto.UINT8)
    # Static weight width is available when lowering a Linear. Accumulate long
    # contractions in INT64 between safe INT32 products, just as the CUDA kernel.
    contraction = right.shape[0] if right.shape is not None else None
    if not isinstance(contraction, int):
        raise ValueError("Native INT8 ONNX lowering requires a statically known contraction width.")
    # Also bound the raw U8/S8 product before zero-point compensation, so the
    # provider need not rely on cancellation of overflowing intermediates.
    limit = (2**31 - 1) // (255 * 128)
    if contraction <= limit:
        accumulator = op.MatMulInteger(unsigned, right, zero)
    else:
        accumulator = None
        for start in range(0, contraction, limit):
            a = op.Slice(unsigned, [start], [min(start + limit, contraction)], [1])
            b = op.Slice(right, [start], [min(start + limit, contraction)], [0])
            partial = op.Cast(op.MatMulInteger(a, b, zero), to=onnx.TensorProto.INT64)
            accumulator = partial if accumulator is None else op.Add(accumulator, partial)
    return op.Mul(op.Mul(op.Cast(accumulator, to=onnx.TensorProto.FLOAT), op.Unsqueeze(row_scale, [1])), op.Unsqueeze(column_scale, [0]))
