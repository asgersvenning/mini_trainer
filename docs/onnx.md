# ONNX export

Export your own checkpoint with `mt_export` or the Python API below. For the
ready-made MAMBO model, use the [deployment package](../deployment/README.md).

Install the optional `export` extra alongside the backend needed to reconstruct your
model. For example, a disposable CPU development environment can use:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/mini-trainer-export uv sync --locked --extra cpu --extra export --extra timm
/tmp/mini-trainer-export/bin/mt_export --weights weights.pt --output exported-model
```

Keep the backend extras appropriate to your existing environment when installing
there. Export dependencies are lazy; core imports and CLI help work without them.

## Python API

```python
from mini_trainer.modeling.onnx import export_onnx

bundle = export_onnx(
    model,
    preprocessed_images,
    "exported-model",
    verification_inputs=[another_preprocessed_batch],
    preprocessing={"description": "Your exact decoding, resize, crop and normalization recipe"},
)
```

The exporter runs the model's actual evaluation forward. It has no architecture
allowlist or replacement classifier implementation. It accepts a single batched
floating-point tensor and flattens tensor outputs nested in lists, tuples and
string-keyed dictionaries. The manifest records how to reconstruct that structure;
`None` leaves are preserved. Custom models with additional inputs can expose a
single-input inference wrapper before export.

Flat, normalized, prior-adjusted, masked, hierarchical, conditional, independent,
and both autoregressive classifier families use this same path. No softmax or
other score conversion is added: output semantics are those of `model.eval()`.
Classifier metadata includes the effective class mappings after masking. In
particular, hierarchical outputs remain separate tensors in their original order.

A private copy uses the example tensor's dtype and `reference_device` (CPU by default). The source model's weights,
training modes, device and classifier caches remain untouched. Root DataParallel,
DDP and compiled wrappers are unwrapped. Use float32 examples for portable CPU
verification, including when the source was trained in bfloat16. Float16/float64
examples are accepted subject to the model and runtime's operator support.

Batch size is dynamic by default; other dimensions are fixed. Export traces batch
two and verifies batches one, two and four, the supplied example, and any extra
verification batches. Use `dynamic_batch=False` or CLI `--static-batch` when a
model requires a fixed batch size. There is no silent static fallback.

## Artifact and preprocessing contract

The destination must be new. The directory is published only after ONNX validation
and CPU ONNX Runtime parity succeed (default `rtol=1e-4`, `atol=1e-5`). Python
callers can choose tolerances and output names. Shape, dtype and finite values are
also checked. Exporter/operator failures propagate instead of substituting a model.

Copy the **whole directory**, including any external tensor data:

- `model.onnx` and optional external weight files;
- `manifest.json`: schema version, file hashes, model class, serialized export-state
  hash, checkpoint hash when available, input/output shapes and dtypes, output tree,
  classifier metadata, dependency versions, preprocessing recipe and parity results.

Preprocessing is outside the graph. Inputs must already have the exact channel
order, resizing, cropping, scaling and normalization used with the model. The CLI
uses the existing checkpoint loader and preprocessor to infer the default input
shape; `--input-shape C H W` overrides it. It disables pretrained weight downloads
where supported, but some backend constructors still require cached or downloadable
configuration files. `--model-args` accepts JSON constructor arguments.

Supply a deployment recipe using `--preprocessing recipe.json` or the Python keyword.
This is caller-provided documentation, not executable preprocessing or a verified
recipe. Without it, the manifest explicitly sets `requires_configuration: true`.
An ONNX graph alone therefore does not establish end-to-end image prediction parity.

## Standalone inference

A deployment environment needs NumPy and ONNX Runtime, plus your preprocessing code:

```python
import json
import onnxruntime as ort

with open("exported-model/manifest.json") as handle:
    manifest = json.load(handle)
session = ort.InferenceSession("exported-model/model.onnx", providers=["CPUExecutionProvider"])
outputs = session.run(
    [output["name"] for output in manifest["outputs"]],
    {manifest["input"]["name"]: preprocessed_numpy_images},
)
```

## Choosing an export path

| Checkpoint / purpose | Export path | Verification reference |
| --- | --- | --- |
| Floating checkpoint | Default export; calibration is a separate optional step | Model's floating evaluation forward |
| Native `cuda-int8-linear` checkpoint | `--reference-device cuda:0`; retains dynamic activation quantization | Native forward with autocast and TF32 disabled |
| Native checkpoint for static calibration | `--materialize-int8-training`; produces floating weights | Materialized floating forward, not the native quantized forward |

Integer weights or nodes do not establish integer execution on a chosen device.
Inspect the runtime profile and evaluate quality on that provider before selecting
a quantized recipe. Recorded results and their limits are linked below.

## Native INT8 training checkpoints

Native `cuda-int8-linear` checkpoints can export their captured integer forward
with an explicit CUDA reference. This requires the existing quantization and
export extras in a compatible CUDA environment; deployment itself needs only
ONNX Runtime, NumPy and external preprocessing.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/mt_export --weights native-int8-weights.pt \
    --output native-int8-onnx --reference-device cuda:0
```

The Python API requires float32 inputs and `reference_device="cuda:0"`.
The CLI retains restricted checkpoint loading. Export freezes only its private
copy; the caller's weights, gradient flags and precision settings are preserved.

The graph retains native dynamic row quantization and scaled integer Linear
products (`MatMulInteger`), including overflow-safe accumulation. Convolutions
remain floating. No calibration data or floating-weight substitution is involved.
The CUDA reference temporarily disables autocast and TF32; the manifest records
this choice. This reference need not match AMP/TF32 scores, since rounding can
change subsequent integer activation codes.

Supply representative `verification_inputs` and evaluate confidence thresholds
separately: the recorded full-validation results below show why passing a small
export sample does not establish parity for every input.

## Explicit materialization for deployment calibration

`mt_export --materialize-int8-training` exports a **separate floating model** from
a native INT8 training checkpoint. This is an opt-in route to subsequent static
calibration for runtimes that cannot execute the native dynamic quantizer. The
ordinary export path above remains unchanged.

```bash
mt_export --weights native-int8-last.pt --output materialized-onnx \
    --input-shape 3 128 128 --materialize-int8-training
```

Conversion validates the native recipe, materializes effective weights and retains
class metadata, masks and buffers. Invalid states and incompatible parameter ties
fail explicitly. The source checkpoint is unchanged; optimizer/training state is
excluded from the deployment artifact.

**Dynamic activation quantization is removed.** ONNX verification compares against
the materialized floating model, not against the native training forward. Its
manifest records the original checkpoint hash, the floating state hash and an
explicit `source.quantized_training_materialization` recipe, including
`dynamic_activation_quantization_preserved=false` and
`training_resume_supported=false`. The original checkpoint remains the source
for native training/resume. Materialization holds floating weights in memory;
it is not a training-memory optimization.

Use the maintained [input preparation](../dev/benchmarks/inference.md#maintained-image-input-preparation)
and [calibration](../dev/benchmarks/inference.md#maintained-onnx-calibration-command)
commands. Calibrate on training data and compare native, materialized and calibrated
predictions on the same held-out samples with `mini_metrics` (macro F1, recall,
precision, coverage and Theil's U). For efficiency, include a practical FP16
baseline. The calibration guide describes the tested TensorRT recipe; build and
inspect its engine on the destination device.

## Qualification evidence and limits

Export tests cover representative offline torchvision, timm, Transformers and
OpenCLIP backbones, all classifier head families, active masks and dynamic batches.
They include EfficientNetV2-S flat/hierarchical heads and native INT8 numerical edge
cases. This is not certification of every catalogue model: custom operators and
control flow remain subject to the
[PyTorch exporter](https://docs.pytorch.org/docs/stable/onnx_export.html).

The linked Blair experiments concern specific checkpoints and local x86 CPU/laptop
CUDA environments; they do not establish ARM support, large-vocabulary capacity or
performance on other GPUs. MAMBO has separate qualification in its deployment README.

- [Floating export](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#efficientnetv2-onnx-cpu-export-and-inference-quantization):
  trained checkpoints passed real-image CPU parity. Signed MinMax quantization
  lost substantial quality and left many convolutions floating.
- [CPU calibration](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#onnx-activation-calibration-execution-coverage-and-macro-metrics):
  unsigned Percentile quantization executed all convolutions as `QLinearConv` and
  roughly halved warm inference latency, with measured quality losses. This was
  exploratory, without a production acceptance gate.
- [Native full-validation comparison](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#native-onnx-full-validation-quality-and-numerical-limits):
  top-1 predictions and the five reported metrics matched the full-FP32 CUDA
  reference, but some scores exceeded export tolerance. Sample parity does not
  establish confidence-threshold equivalence.
- [CUDA placement](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#onnx-cuda-provider-placement):
  native integer heads fell back to CPU; calibrated convolutions ran in floating
  point. The calibrated recipe also changed scores and one top-1 prediction versus
  its CPU execution. CPU quality/speed results therefore cannot qualify that CUDA
  deployment.

Model cards, evaluation attachments and Hub upload commands remain separate
[roadmap work](roadmap.md).
