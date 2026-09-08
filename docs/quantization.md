# INT8 quantization: initial x86 backend

This is an opt-in Python API for **static 8-bit weights and 8-bit activations**,
using TorchAO PT2E. Actual quantized training with reduced memory and training
time now has an initial [CUDA model integration](quantized-training.md); see the [QT/loader probes](../dev/benchmarks/README.md#quantized-training-and-loader-performance). It supports post-training calibration (PTQ) and
quantization-aware training (QAT). QAT uses fake quantization with float32 master
parameters/gradients; it does not promise integer backward computation or reduced
training memory. Converted inference executes native oneDNN integer Conv/Linear
kernels. This is separate from float16/bfloat16 AMP.

Install the optional dependency in an explicitly selected backend environment:

```bash
uv sync --extra cpu --extra quantization
# Existing CUDA environments: do not run a CPU sync; select their CUDA extra.
```

The first verified backend is x86 CPU with PyTorch 2.12 and TorchAO 0.17.
TorchAO is imported lazily. Ordinary training, prediction, checkpoint formats and
ONNX export are unchanged. The new API is in `mini_trainer.modeling.quantization`.

## Calibration and inference

```python
import torch
from mini_trainer.modeling.quantization import prepare_int8, load_int8

# model is a loaded floating-point mini_trainer model. All inputs below are
# float32 CPU batches AFTER the same preprocessing used for ordinary inference.
prepared = prepare_int8(model, example_batch)
with torch.no_grad():
    for images in training_calibration_batches:
        prepared(images)
converted = prepared.convert()
converted.save(
    "int8-model",
    example_batch,
    preprocessing={"recipe": "record the actual resize, scale and normalization"},
    calibration={"split": "train", "manifest_sha256": "record the actual manifest hash"},
)
inference, coverage = load_int8("int8-model").lower(example_batch)
with torch.no_grad():
    scores = inference(example_batch)
```

Calibration must use training data, never held-out validation/test examples.
The caller supplies provenance; the API cannot infer the provenance of tensors.
The prepared PTQ graph is a calibration object, including when its mode is eval.
Convert it before evaluating held-out data. Conversion refuses unobserved or
nonfinite ranges, and does not modify the prepared model.

Weights use symmetric per-channel int8; activations use affine per-tensor uint8.
Bias, normalization, score transforms and other non-linear operations may remain
floating point. The report includes the actual remaining operator inventory.
All captured Conv1d/Conv2d/Linear operations must receive weight and activation
annotations. Lowering fails if it cannot produce integer kernels or leaves
floating Conv/Linear kernels. It never labels a plain Q/DQ reference execution
as native integer inference.

The bundle contains a reference `model.pt2` graph, checksum, input shape, class
metadata, structured output mapping, bit widths, dependency versions,
preprocessing/calibration provenance, and verified lowering coverage. Real calibration tensors are excluded from the saved program. Packing
is performed again on the deployment CPU. A reference graph alone is not an
accelerated runtime. Existing output directories are never overwritten.

## Quantization-aware training

```python
prepared = prepare_int8(model, example_batch, qat=True)
optimizer = torch.optim.AdamW(prepared.parameters(), lr=1e-4)
prepared.train()
for images, targets in training_batches:
    optimizer.zero_grad()
    loss = criterion(prepared(images), targets)
    loss.backward()
    optimizer.step()

prepared.freeze_observers()  # Optional: hold learned ranges fixed for later steps.
prepared.eval()              # Evaluation does not update QAT ranges or BatchNorm.
with torch.no_grad():
    scores = prepared(validation_batch)
converted = prepared.convert()
```

Construct the optimizer **after** preparation. Save `prepared.state_dict()` and
the optimizer/scheduler/scaler states. Restore into an identically prepared model,
then restore optimizer state. Observer ranges, fake-quant flags and the explicit
freeze flag are part of the state. The recipe is checked on restoration. This is
not an ordinary `Classifier.build(weights=...)` checkpoint: automatic reconstruction
through `mt_train`/`mt_predict` remains a subsequent integration step.

QAT runs can use `train_one_epoch` with an appropriate criterion, disabled EMA,
and no embedding-dependent regularizer. Captured graphs neither consume ambient
supervision nor populate `EmbeddingContext`. Train/eval switching covers dropout
and BatchNorm; arbitrary Python training branches are specialized by capture.
Autoregressive teacher-forcing/sampling requires a separate training integration
and is not currently a supported QAT claim. Capture failures propagate explicitly.

## Coverage and next increments

Inputs currently have a fixed captured batch and image shape. All batches must
match it; pad and slice final inference batches, or prepare a separate shape.
The original model's parameters, modes and caches are preserved by preparation.
Functional linears, weight parametrization, hierarchical aggregation and class
masks are included in capture; this does not rely on a backbone allowlist.

Focused tests exercise flat, hierarchical, conditional and independent heads,
real gradients, exact controlled QAT/AdamW continuation, held-out-data isolation,
integer operator execution, checksums and artifact reload. A synthetic oracle
exercises QAT through the actual training loop and checks integer predictions.

Still required: user-facing checkpoint/CLI integration, dynamic batch support,
MNIST/Blair reports, broader backbone/operator coverage, GPU quantization,
ONNX/runtime conversion, and lower-bit profiles. Model quality, artifact size,
memory and latency need measured comparisons; no speedup or quality benefit is
claimed by passing compatibility tests. EMA remains unsupported.

Run focused checks without changing the installed environment:

```bash
OMP_NUM_THREADS=1 bash dev/check.sh test tests/test_quantization.py
```

Backend references: [PT2E x86 quantization](https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_x86_inductor.html),
[QAT workflow](https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html).
Strict graph capture is intentional: the installed backend otherwise misses
functional-linears' source metadata. Explicit `lower_pt2e_quantized_to_x86`
provides native kernels without relying on a compiler silently optimizing Q/DQ.
