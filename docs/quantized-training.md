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

CUDA SGD/AdamW-style `add_` and `addcdiv_` updates fuse dequantization,
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

`--compile-optimizer` now supports the FMA primitive that Dynamo uses for tensor
learning-rate updates. The formerly failing twelve-group SGD regression passes,
as does twelve-group AdamW, with hard failure enabled on compiler-cache fallback.
This establishes compilation compatibility, not a performance recommendation:
the batch-128 dense MNIST comparison is slower and less accurate with QT than float.
The [larger-batch profile](benchmarks.md#larger-batches-and-direct-collation) shows
lower memory and faster training in individual runs after compiler caches are
populated. The [three-seed requantization comparison](benchmarks.md#functional-fused-requantization)
shows lower memory and slightly higher accuracy than float, but later-phase
speed remains mixed and broader workload validation is still required.
Compiled stochastic requantization can follow a different random trajectory from
the eager row kernel, so a matching seed does not establish identical training.
Floating-to-INT8 copies now use a fused row-quantization kernel for matching CUDA
FP32/FP16/BF16 tensors with at most 16,384 columns. It returns fresh codes and
scales; ordinary tensor copies perform the final storage mutation so compiled
optimizer calculations that need the old weight remain correctly ordered. This
keeps no floating master weight, though transient floating updates still exist.
The operator is marked as seeded randomness, and compiled regressions check
independent rounding, generator replay, sub-code updates and optimizer state.
The random trajectory can differ from earlier compiled requantization.
The earlier storage prototype's fake-tensor and dtype-cache failures are resolved
by an explicit Triton kernel and custom-operator boundary; the storage kernel
does not depend on Dynamo's per-frame variant cache. Model `--compile` remains a
separate option. See the [measured results](benchmarks.md#optimizer-fma-dispatch).

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
ONNX export, checkpoint averaging, DDP/FSDP, quantized activation normalization
and integer convolution training are not established for this path. Distributed training and
EMA are rejected by the training entry point.

## Model compilation modes

Model compilation accepts `--compile --compile-mode reduce-overhead` (or
`compile=True, compile_mode="reduce-overhead"` in Python). This is opt-in and
independent of optimizer compilation. The benchmark runner records the selected
mode. See [compilation guidance](../dev/README.md#model-compilation) for the other
modes and measurement requirements; selecting a mode does not establish a speedup.

Normalized heads resolve their parametrized weights after publishing embeddings,
keeping weight normalization and the integer Linear operation in the same graph.
Previously the embedding publication could split them and cause AOTAutograd to
expect an INT8 tensor subclass gradient where the backward supplies a float tensor.
This affected both ordinary and CUDA graph compilation. Regression coverage now
includes normalized and ordinary heads, eager/default/reduce-overhead execution,
AMP training, compiled/eager optimizers, checkpoint loading and eager resume.
A separate FP32 test compares input and parameter gradients with an embedding
auxiliary loss. AMP compilation can change rounding and INT8 activation bins;
these checks do not promise bitwise-identical eager and compiled trajectories.

## Evidence and remaining work

The [developer probes](../dev/benchmarks/README.md#quantized-training-and-loader-performance)
record both positive and negative workload-dependent results. Compiler cache keys
include backend source and tensor metadata to avoid reusing obsolete backward
graphs. Numerical checks include small gradients, compiled/eager agreement,
weight storage, optimizer updates, regularization and model-state restoration.

Kernel-probe results do not establish a real-model speedup or convergence. The
integrated path now has paired synthetic-oracle, MNIST and hierarchical Blair
smoke runs, recorded in [the dataset benchmark results](benchmarks.md#integrated-int8-training).
Complete optimizer/resume coverage, demonstrated real-workload speedups and broader
quantized operation coverage remain outstanding. These are requirements
for the overall QT goal, not conclusions implied by this initial integration.

The integrated backend was rerun on the four-layer, 4096-wide, batch-2048 compiled
SGD probe: 15.61 ms/step and 319,063,552 peak allocated bytes for INT8, versus
30.05 ms and 386,139,648 bytes for FP16. These single-run numbers are about
1.93x faster and 17% lower peak memory for that workload, with nonzero gradients
checked before timing. They remain kernel-probe evidence, not a claim about
MNIST, Blair or typical convolutional models.


Row-wise normalization is checked against PyTorch's represented-value forward
and backward results, including signed scales and zero magnitudes. Tests inspect
saved tensors to exclude a retained floating direction matrix, and exercise
checkpoint restoration, eager/compiled CUDA training and masked inference.
Initial zero direction rows are rejected before preparation mutates any weights;
normalization is undefined for these rows. The mathematical contract follows
[PyTorch weight normalization](https://docs.pytorch.org/docs/2.12/generated/torch.nn.utils.parametrizations.weight_norm.html).

The [dense MNIST comparison](benchmarks.md#dense-mnist-profile-with-corrected-peak-measurements)
now exercises a quantized backbone through the compiled trainer. Its accuracy is
similar for one seed, but QT is slower and uses a higher whole-training peak than
the floating path despite substantially smaller parameter storage. Corrected
benchmark logging preserves CUDA peaks across all resets; older dataset-run CUDA
readings do not establish whole-run memory reductions. Compiled training now saves
unwrapped model keys so ordinary checkpoint restoration and resume remain valid.
