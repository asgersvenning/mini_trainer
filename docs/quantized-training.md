# CUDA INT8 training integration

The opt-in training path stores eligible Linear weights and saved linear inputs
in INT8 and uses integer matrix products for forward, input gradients and weight
gradients. It retains no floating-point master copy of those weights. Gradients,
optimizer state, biases, activation normalization, convolutions and auxiliary regularization
remain floating point. This is separate from fake-quantized QAT and x86 PTQ.

Install the optional `quantization` extra while explicitly retaining the intended
PyTorch CUDA backend, as described in the README. The current implementation uses
TorchAO's experimental Triton kernels. It has been exercised on an RTX 3080 Ti;
CPU preparation and checkpoint inspection do not establish CPU execution support.
See the [validation audit](quantized-training-validation.md) for current tests,
measured training/loading benefits and the limits of those results.

```bash
mt_train -i /path/to/data --device cuda --quantized-training --dtype float16 --cache cpu --cache-workers 0
```

`--quantized-training` prepares weights before building the optimizer and logs
coverage. `--cache-workers 0` makes cache construction synchronous; its selection
is separate from DataLoader workers. Ordinary training defaults are unchanged.

The Python API also supports explicit module selection:

```python
import torch
from mini_trainer.modeling.quantized_training import prepare_quantized_training

# Load floating weights and move the model to its intended device first.
coverage = prepare_quantized_training(model)  # in place, before optimizer creation
# Or: prepare_quantized_training(model, module_names=["encoder.projection"])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

The returned recipe lists quantized modules, skipped operations, remaining
floating-point parameters and physical versus reference weight storage. Automatic
selection covers ordinary `nn.Linear` modules, including their functional use by
Classifier heads. It preserves shared weights when every owner is selected.
Initial conversion of large weights processes row chunks of at most 4,194,304
elements (or one row when wider), limiting the quantizer's floating temporaries.
This preserves deterministic INT8 codes and row scales; it does not reduce the
storage needed for source weights, validation, gradients or optimizer states.
Row-wise PyTorch weight normalization (`dim=0`) quantizes the direction parameter
while retaining its scalar magnitude per output row in floating point. Effective
normalized weights reuse the integer codes with new row scales; normalization
backward applies its Jacobian to the approximate Linear gradient. No floating
weight matrix is retained for this operation. Other parametrizations, normalization
dimensions and weights shared with an unselected operation stay floating point
and are reported. Explicit unsupported
selections and models with no eligible weights fail before changing weights.
Quantizing the hidden linear layer of a convolutional classifier does not make
its convolutions integer operations.

CUDA execution supports batched inputs, bias, masked classifier rows,
single-sample inference and float16/bfloat16 autocast with float32 parameters.
Float32 optimizer parameters avoid the FP16 AdamW epsilon underflow discussed in
the developer probe. Eager SGD, AdamW and the repository's MuonAuxAdamW update
paths are covered; fused optimizer variants are not established. Quantized
regularization uses a differentiable floating view of the represented weights.

Eager CUDA SGD/AdamW-style `add_` and `addcdiv_` updates fuse dequantization,
the weight update and stochastic requantization over the underlying storage
tensors. This kernel compiles on first use even when the outer optimizer is
eager. It retains INT8 codes and row scales, advances tensor version counters,
and preserves explicit intermediate precision casts. Scalar tensor weight decay
rescales rows without another stochastic rounding pass. CPU inspection/update
tests use the ordinary floating calculation and copy path. The fused row kernel
covers matching FP32/FP16/BF16 update tensors and rows up to 16,384 elements;
broadcasting, mixed dtypes and wider rows retain the ordinary update path.
No floating master weight is retained by either path. The new kernel uses CUDA
RNG seeds with Triton stochastic rounding, so exact trajectories differ from the
earlier floating update even when starting from the same seed.

Matrix products reuse TorchAO's INT8 kernel with a separate local tuner. CUDA
graph timing avoids the default tuner's 256 MiB cache-flushing allocation, and
selected configurations are cached on disk. TorchAO's global operators and tuner
are unchanged. The explicit update and matrix operators provide fake execution
implementations for model compilation.

`--compile-optimizer` supports tensor-learning-rate FMA updates and functional
stochastic requantization. Floating-to-INT8 copies use a fused row kernel for
matching CUDA FP32/FP16/BF16 tensors with at most 16,384 columns, returning fresh
codes/scales before the final storage mutation. Eager and compiled rounding can
follow different random trajectories even with a matching seed. Transient floating
updates remain; compilation compatibility does not imply a speedup.

`--compile-optimizer --optimizer-cudagraphs` opts into optimizer graph replay.
During AOT fake-tensor tracing, updates expose floating arithmetic followed by
functional requantization and storage copies. Eager native updates are retained.
Expected failed kernel-tuning candidates release their exception tracebacks
promptly so temporary tensors do not outlive graph pool tracking during first
use; unexpected kernel errors still propagate.
Learning rates stay on CUDA during replay; checkpoints retain numeric values and
explicit non-default rate precision. This leaves the AMP gate, scheduler and
MuonAuxAdamW outer counter in their existing roles. The training loop explicitly
marks each graph iteration before the model runs, keeping backward gradient
buffers alive until the optimizer consumes them. Custom training loops must call
[`torch.compiler.cudagraph_mark_step_begin()`](https://docs.pytorch.org/docs/2.12/generated/torch.compiler.cudagraph_mark_step_begin.html)
before each training iteration when
combining compiled models with optimizer graph replay. See the
[optimizer graph requirements](../dev/README.md#optimizer-cuda-graphs), including
native fused float32-rate restrictions. The native fused optimizer tests use
floating parameters; they do not establish native fused updates of INT8 weights.

## Checkpoints and inference

Prepared models include their recipe in `state_dict`. Ordinary `mt_train`
checkpoints retain INT8 parameter storage, and `Classifier.build(weights=...)`
restores the parameter types before loading weights. Known tensor classes are
allowed only within a scoped `weights_only=True` load. To resume through
`mt_train`, enable `--quantized-training` again so optimizer construction sees
the correct parameters. Existing stochastic-resume limitations still apply:
the trainer does not generally persist sampler or RNG state.

For a custom architecture outside `Classifier.build`:

```python
from mini_trainer.modeling.quantized_training import load_training_weights, restore_quantized_training

state = load_training_weights("weights.pt", map_location="cpu")
restore_quantized_training(model, state)
model.load_state_dict(state)
```

Use the same architecture and intended dtype. This restores model state; create
and restore optimizer/scheduler/scaler state in their normal order separately.
The same model supports CUDA inference with `eval()` and `inference_mode()`.
An opt-in [ONNX export path](onnx.md#native-int8-training-checkpoints) captures
the integer forward using a full-FP32 CUDA reference and verifies ONNX Runtime CPU
parity. Target-provider performance remains unverified. Checkpoint averaging,
DDP/FSDP, quantized activation normalization and integer convolution training
are not established for this path. Distributed training and
EMA are rejected by the training entry point.

## Model compilation modes

Model compilation accepts `--compile --compile-mode reduce-overhead` (or
`compile=True, compile_mode="reduce-overhead"` in Python). This is opt-in and
independent of optimizer compilation. The benchmark runner records the selected
mode. See [compilation guidance](../dev/README.md#model-compilation) for the other
modes and measurement requirements; selecting a mode does not establish a speedup.

Embedding publication preserves context side effects across compilation, with
normalized weights beside their Linear consumer. Context remains shared, not
thread/task-local. AMP fusion can change rounding and INT8 activation bins;
eager and compiled trajectories are not promised to be bitwise identical.
Compiler cache keys include backend source and tensor metadata so hidden backward
changes invalidate cached graphs.

## Evidence and remaining work

See [current findings](benchmarks.md), [validation contracts](quantized-training-validation.md)
and [the roadmap](quantization-roadmap.md). These own measured benefits, negative
results and remaining hardware/quality work. Use [the training guide](../dev/benchmarks/training.md)
for reproducible commands. Kernel speedups do not establish real-model convergence
or end-to-end efficiency.

Row-wise normalization tests cover represented-value forward/backward, signed
scales, zero magnitudes, saved storage, masked inference and checkpoint restoration.
Initial zero direction rows are rejected before mutation. Whole-run memory includes
initialization, gradients and optimizer storage as well as compressed parameters.

## Large-class accumulator bounds

Input-gradient products contract over the output class count. For contractions
above 131,071, the backend now combines bounded INT32 dot products in INT64 before
converting and scaling the result. This prevents finite but saturated gradients
for large vocabularies. Shorter contractions keep the existing tuned kernel.
ONNX export also bounds integer partial products and combines them in INT64.
See the [arithmetic regression and limits](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#long-contraction-int8-accumulator-correctness).
A million-output Linear gradient test does not establish that a full million-class
EfficientNetV2 training configuration fits the available GPU; parameter,
initialization, gradient and optimizer storage still require separate measurement.

## Implementation layout

Public APIs remain in `modeling.quantized_training` and `modeling.quantization`
(the separate PTQ/QAT backend). Private native INT8 code lives in
`modeling/_quantized_training/`: tensor dispatch in `__init__.py`, plus `matmul.py`,
`normalization.py`, `update.py` and ONNX translation in `onnx.py`. Keeping the backend
package at its original module path preserves serialized `TrainingWeight` identities.
