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

A private CPU copy uses the example tensor's dtype. The source model's weights,
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
GPU providers, ARM execution and arbitrary spatial dimensions remain unvalidated.
An experimental ONNX static INT8 recipe executed on the local CPU but lost
substantial accuracy and retained floating convolution execution. It is not a
supported production recipe. Native CUDA QT checkpoints remain a separate backend
and are not made ONNX-exportable by these floating-checkpoint experiments.

These local bundles are a foundation for Hugging Face hosting. Model cards,
evaluation attachments and Hub upload commands remain separate roadmap work.
