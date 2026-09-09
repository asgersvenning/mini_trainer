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
    accumulator = op.MatMulInteger(unsigned, right, zero)
    return op.Mul(op.Mul(op.Cast(accumulator, to=onnx.TensorProto.FLOAT), op.Unsqueeze(row_scale, [1])), op.Unsqueeze(column_scale, [0]))
