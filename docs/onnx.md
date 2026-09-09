# ONNX export

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

The test matrix covers small offline instances from torchvision (ResNet,
EfficientNet, ViT), timm, Hugging Face Transformers and the OpenCLIP encoder wrapper
used by BioCLIP, plus every classifier head family. This is representative backend
coverage, not certification of every model in the catalog. Arbitrary custom
operators and data-dependent Python control flow remain subject to the
[PyTorch ONNX exporter's support](https://docs.pytorch.org/docs/stable/onnx_export.html).
The actual EfficientNetV2-S backbone with symmetric normalized flat/hierarchical
heads is also covered by offline dynamic-batch export tests. Trained Blair
checkpoints passed ONNX Runtime CPU parity on real images; see the
[deployment experiment](benchmarks.md#efficientnetv2-onnx-cpu-export-and-inference-quantization).
Local [CUDA placement checks](benchmarks.md#onnx-cuda-provider-placement) expose
CPU fallback for native integer heads and floating execution for calibrated
convolutions. Target GPU hardware, ARM execution and arbitrary spatial dimensions
remain unvalidated.
An initial signed MinMax INT8 recipe lost substantial accuracy and retained
floating convolutions. A follow-up unsigned Percentile recipe executed all
convolutions as QLinearConv and roughly halved warm local CPU inference latency,
with remaining quality losses measured through `mini_metrics`; see the
[calibration and metric results](benchmarks.md#onnx-activation-calibration-execution-coverage-and-macro-metrics).
It remains exploratory, with no agreed production quality gate or target-device
verification. Native CUDA QT checkpoint export is a separate path described below;
these floating-checkpoint PTQ experiments do not validate it.

These local bundles are a foundation for Hugging Face hosting. Model cards,
evaluation attachments and Hub upload commands remain separate roadmap work.


## Native INT8 training checkpoints

Native `cuda-int8-linear` checkpoints can export their captured integer forward
with an explicit CUDA reference. This requires the existing quantization and
export extras in a compatible CUDA environment; deployment itself needs only
ONNX Runtime, NumPy and external preprocessing.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/mt_export --weights native-int8-weights.pt \
    --output native-int8-onnx --reference-device cuda:0
```

The Python API uses `export_onnx(model, float32_images, destination,
reference_device="cuda:0")`. Float32 inputs and a CUDA reference are required for
this backend. The CLI uses the scoped native-weight loader, retaining restricted
checkpoint loading. The export copy is frozen to let the exporter unpack integer
parameter storage; the caller's weights and gradient flags are preserved.

The graph retains the training backend's dynamic symmetric row quantization,
including clipping, ties-to-even rounding and zero-row behavior. Its scaled INT8
products lower to `MatMulInteger` with bounded INT32 partial accumulation and
floating row/column scales. Long contractions combine partial sums in INT64
before converting and applying scales, preventing accumulator saturation. Activation codes are represented as unsigned codes with zero point 128;
this preserves the signed values exactly. Weights stay signed INT8. No calibration
set, floating-weight substitution or replacement classifier is used. Convolutions
remain floating, as they do in native QT training. This is distinct from the
static Percentile recipe that also quantizes convolutions.

CUDA reference execution temporarily disables CUDA autocast and TF32 in cuDNN
and floating matrix products, restoring the caller settings afterward. Real trained images exposed a parity
failure with TF32 enabled; full-FP32 reference execution passed the original
rtol=1e-4/atol=1e-5 checks. The manifest records the reference device and TF32 choice.
This does not promise matching scores against AMP or TF32 evaluation of the same
checkpoint, whose rounding can change the subsequent integer activation codes.

Tests cover normalized symmetric flat/hierarchical heads, EfficientNetV2-S,
active-class filtering, dynamic batches, checkpoint CLI loading and numerical
edge cases. Trained Blair checkpoints also passed checks on eight real validation
images at batches 1, 2, 4 and 8. An exported graph still needs runtime profiling
and quality evaluation on the intended provider. Local CUDA-provider execution
retains CPU MatMulInteger operations; it does not establish integer GPU execution.
Target GPU/ARM deployment, million-class export capacity and production performance
of this native path remain unverified. On the full Blair validation split,
top-1 predictions and the
requested macro metrics matched the full-FP32 CUDA reference, but some image
scores exceeded the strict export tolerance; see the
[full-validation results](benchmarks.md#native-onnx-full-validation-quality-and-numerical-limits).
Supply representative `verification_inputs` and evaluate deployment thresholds
separately; passing the default sample checks does not establish universal score
parity or confidence-threshold equivalence. The generic exporter does not
impose a model allowlist; configurations outside this tested coverage must pass
the same export and parity checks before a bundle is published.

## Explicit materialization for deployment calibration

`mt_export --materialize-int8-training` exports a **separate floating model** from
a native INT8 training checkpoint. This is an opt-in route to subsequent static
calibration for runtimes that cannot execute the native dynamic quantizer. The
ordinary export path above remains unchanged.

```bash
mt_export --weights native-int8-last.pt --output materialized-onnx \
    --input-shape 3 128 128 --materialize-int8-training
```

The conversion checks the recorded native recipes, copies the state, materializes
INT8 weight representations, and removes the recipe that would restore native
training tensor types. It preserves class metadata, active-class masks and other
buffers. Normalized directions use signed integer codes, with magnitudes set to
zero for zero-scale rows. This follows the native effective-weight formula
and handles zero scales without introducing an ordinary weight-normalization
divide-by-zero. Compatible parameter ties are retained by normal model loading;
incompatible tied roles/views fail instead of silently loading different values
into one shared parameter. Undefined zero-code directions and invalid/nonfinite states fail.
The source checkpoint is never rewritten, and neither optimizers nor training
state are carried into this deployment artifact.

**Dynamic activation quantization is removed.** ONNX verification compares against
the materialized floating model, not against the native training forward. Its
manifest records the original checkpoint hash, the floating state hash and an
explicit `source.quantized_training_materialization` recipe, including
`dynamic_activation_quantization_preserved=false` and
`training_resume_supported=false`. The original checkpoint remains the source
for native training/resume. Materialization holds floating weights in memory;
it is not a training-memory optimization.

Use the maintained [input preparation](../dev/benchmarks/README.md#maintained-image-input-preparation)
and [calibration](../dev/benchmarks/README.md#maintained-onnx-calibration-command)
commands on this artifact. For the tested TensorRT recipe, choose signed symmetric
activations, signed per-channel weights and floating biases, with training-only
calibration data. Build and inspect a new engine for its destination device.
Compare native predictions, materialized predictions and the calibrated deployment
candidate on the same held-out samples using all five requested mini_metrics
metrics. Use a practical FP16 baseline for efficiency comparisons; successful
materialization/export alone does not establish acceptable quality, integer GPU
execution or a worthwhile cost reduction.
