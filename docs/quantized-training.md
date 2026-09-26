# CUDA INT8 training integration

This opt-in path stores eligible Linear weights and saved Linear inputs in INT8,
uses integer forward/backward matrix products and retains no floating master
weights. Gradients, optimizer state, biases, convolutions, activation normalization
and auxiliary regularization remain floating point. It is separate from
[x86 PTQ and fake-quantized QAT](quantization.md).

Install the `quantization` extra with the intended PyTorch CUDA backend using the
[root installation guide](../README.md#installation). Execution uses TorchAO's
experimental Triton kernels; CPU preparation/inspection does not support CPU
inference. Local RTX 3080 Ti results are recorded in [benchmark findings](benchmarks.md).

```bash
mt_train -i /path/to/data --device cuda --quantized-training --dtype float16 --cache cpu --cache-workers 0
```

`--quantized-training` prepares weights before optimizer creation and logs coverage.
`--dtype float16` selects autocast; model parameters remain float32. The example
builds its CPU image cache synchronously; cache and DataLoader workers are separate
settings. Ordinary training defaults are unchanged.

For explicit module selection:

```python
import torch
from mini_trainer.modeling.quantized_training import prepare_quantized_training

# Load floating weights and move the model to its intended device first.
coverage = prepare_quantized_training(model)  # in place, before optimizer creation
# Or: prepare_quantized_training(model, module_names=["encoder.projection"])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

The returned recipe reports selected/skipped modules, remaining floating parameters
and physical/reference weight storage. Automatic selection covers `nn.Linear`,
including functional use by classifier heads. Shared weights are converted only
when all owners are selected. Explicit unsupported selections and models with no
eligible weights fail before mutation.

| Feature | Contract |
| --- | --- |
| Row-wise weight normalization (`dim=0`) | INT8 direction, floating magnitude per output row; no retained floating weight matrix. Other parametrizations/dimensions stay floating and are reported. Initial zero directions are rejected. |
| Inputs and inference | Batched inputs, bias, masked classifier rows and single-sample CUDA inference are supported. |
| Precision | Float16/bfloat16 autocast with float32 parameters; retain float32 optimizer parameters to avoid FP16 AdamW epsilon underflow. |
| Optimizers | Eager SGD, AdamW and MuonAuxAdamW update paths have coverage. Native fused optimizer support for INT8 weights is not established. |
| Capacity | Preparation uses bounded row chunks, but source weights, validation, gradients and optimizer state still contribute to peak memory. Quantizing Linear layers does not quantize convolutions. |
| Unsupported | Native QT DDP/FSDP, checkpoint averaging, integer convolution training and quantized activation normalization are not established. `mt_train` rejects distributed native QT and EMA. |

Updates stochastically requantize represented weights. CUDA row kernels fuse
eligible updates; unsupported shapes and dtype combinations use the ordinary path.
Neither retains a floating master weight. First use includes kernel compilation/tuning, so measure startup separately from steady-state training.
Eager, fused and compiled execution can follow different rounding trajectories
with the same seed.

## Checkpoints and inference

Prepared `state_dict`s include the quantization recipe and INT8 parameter storage.
`Classifier.build(weights=...)` restores parameter types before loading weights;
`load_training_weights` uses a scoped known-class allowlist with `weights_only=True`.
For `mt_train` checkpoint resume, enable `--quantized-training` again so optimizer
construction sees the correct parameters. General RNG/sampler state is not fully
checkpointed, so arbitrary stochastic continuation is not guaranteed identical.

For a custom architecture outside `Classifier.build`:

```python
from mini_trainer.modeling.quantized_training import load_training_weights, restore_quantized_training

state = load_training_weights("weights.pt", map_location="cpu")
restore_quantized_training(model, state)
model.load_state_dict(state)
```

Use the same architecture and intended dtype. This example restores model weights;
create and restore optimizer/scheduler/scaler state separately in their normal order.
The restored model supports CUDA inference with `eval()` and `inference_mode()`.

The [ONNX guide](onnx.md#native-int8-training-checkpoints) distinguishes native
integer-forward export from materialization to floating weights for static
calibration. These have different numerical contracts; export success alone does
not establish integer execution or useful performance on the destination provider.

## Model compilation modes

Model and optimizer compilation are independent, opt-in controls:

| Setting | Role |
| --- | --- |
| `--compile --compile-mode reduce-overhead` | Compile the model; Python equivalent: `compile=True, compile_mode="reduce-overhead"`. Other modes are in the [compilation guide](../dev/README.md#model-compilation). |
| `--compile-optimizer` | Compile updates with tensor learning rates and functional stochastic requantization. Transient floating updates remain. |
| `--compile-optimizer --optimizer-cudagraphs` | Request optimizer graph replay. Scheduler updates, AMP skip decisions and Muon's outer counter remain outside capture. |

For custom loops, compile the optimizer after scheduler construction and checkpoint
restoration. When combining a compiled model with optimizer graph replay, call
`torch.compiler.cudagraph_mark_step_begin()` before each training iteration so
backward gradient buffers survive until the optimizer consumes them. `mt_train`
already marks this boundary. See [optimizer graph requirements](../dev/README.md#optimizer-cuda-graphs)
for device, learning-rate precision and optimizer restrictions; floating-parameter
fused-optimizer tests do not qualify INT8 weights.

Embedding publication preserves context side effects across compilation, but
context remains shared rather than thread/task-local. AMP fusion can change INT8
activation bins; compiled/eager trajectories need not be bitwise identical.
Compilation compatibility does not itself establish a speedup: compare startup,
steady-state throughput, memory and held-out quality for the intended workload.

## Evidence and remaining work

[Benchmark findings](benchmarks.md) own measured benefits and negative results;
the [validation contract](quantized-training-validation.md) owns required invariants;
the [quantization roadmap](quantization-roadmap.md) owns remaining qualification.
Use the [training benchmark guide](../dev/benchmarks/training.md) for commands.
Local kernel or synthetic-capacity gains do not establish real-model convergence
or end-to-end efficiency on HPC, desktop/Spark or ARM hardware.

## Large-class accumulator bounds

Contractions above 131,071 combine bounded INT32 dot products in INT64 before
scaling, preventing saturated input gradients at large output-class counts. ONNX
export also bounds integer partial products. See the
[arithmetic regression](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#long-contraction-int8-accumulator-correctness).
A million-output Linear test does not establish that full million-class training
fits GPU memory; measure initialization, gradients and optimizer storage too.

## Implementation layout

Public integration is in [modeling/quantized_training.py](../mini_trainer/modeling/quantized_training.py);
`modeling.quantization` is the separate x86 PTQ/QAT backend. Private native code is
in [modeling/_quantized_training/](../mini_trainer/modeling/_quantized_training/):
tensor dispatch in `__init__.py`, with matrix, normalization, update and ONNX
operators in separate modules. Preserve this package path: checkpoints serialize
its `TrainingWeight` identity. Kernel eligibility, tuning and compiler-cache
invalidation details belong beside those implementations.
