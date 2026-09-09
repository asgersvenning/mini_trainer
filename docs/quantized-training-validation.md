# Quantized training and loading validation

This audit records an initial Linear-only milestone, not completion of the
representative-model or deployment objective. It covers genuine quantized training
and shared loading for float and quantized training/inference. The capability is
opt-in; speedups are workload-dependent. Local validation was recorded on 2026-09-09 using
Python 3.13.7, PyTorch 2.12.0/CUDA 13.0, TorchAO 0.17.0 and an RTX 3080 Ti Laptop GPU.

The [2026-09-09 goal status report](quantization-status.md) separates verified
capabilities from remaining work and defines the evidence required for completion.

## Primary model and deployment targets

The primary model is EfficientNetV2 with a symmetric hidden layer (`hidden=True`)
and a normalized `Classifier` or `HierarchicalClassifier`. Start with the
`efficientnet_v2_s` configuration used in `examples/blair.ipynb`, with both heads
on the same reviewed Blair splits. The dense MNIST model is a kernel diagnostic;
TinyConv on Blair is an integration check. Neither establishes performance or
quality for the primary model.

Class count is an independent scaling dimension: production cases may have
10,000–1,000,000 classes. At embedding width 1280, output weights alone occupy
51.2 MB, 512 MB or 5.12 GB in FP32 at 10k, 100k or 1M classes. The 25-class Blair
head is not representative of those parameter, gradient, optimizer-state or score
storage costs. Include synthetic capacity probes alongside dataset quality runs;
never infer large-vocabulary accuracy from randomly assigned synthetic labels.

The [large-class probes](benchmarks.md#large-class-head-capacity-and-initialization)
now exercise full EfficientNetV2-S training steps at 10k and 100k classes for both
normalized heads. They exposed and motivated a quadratic-memory initialization
fix. At 100k classes the current INT8 path saves parameter bytes but increases
peak training allocation, making transient normalization/gradient storage a
priority for investigation. The subsequent
[normalization backward kernel](benchmarks.md#bounded-int8-normalization-backward-storage)
reduces measured 100k-class training-step peak allocation by 17.4% versus float
for both heads. Timings and single-seed quality changes remain mixed. A
million-class training run remains unverified. Local full-model capacity checks
use up to 100k classes; million-class full-model validation is reserved for a
larger machine. Small arithmetic regressions can still test million-element
contractions without constructing a million-class EfficientNetV2 model.

| Deployment target | Execution path to validate | Required measurements |
| --- | --- | --- |
| HPC: A40, A100, B300-class GPU systems with AMD EPYC hosts | PyTorch GPU training | End-to-end training time, steady-state throughput, allocated/reserved GPU peaks, host memory, loading/transfer costs, convergence and checkpoint/resume; record actual GPU, allocation, precision and kernels separately for each system. |
| Local batch processing: NVIDIA Spark or the intended RTX desktop with Ryzen 7 9800X3D | PyTorch training/fine-tuning; ONNX GPU inference | Full and frozen-backbone training separately; export parity, actual execution-provider placement, batch throughput, latency and memory including preprocessing/transfers. Exact installed device and runtime support must be verified. |
| Edge integration: Raspberry Pi or similar | ONNX CPU inference | Export/score parity, accuracy, model size, process memory, batch-one latency and sustained throughput on the actual ARM device, with explicit thread settings. |

These are acceptance targets, not claims of hardware or backend support. Laptop
measurements remain useful for debugging but cannot establish gains on these
systems. Do not assume a single quantized artifact or kernel recipe works across
CUDA PyTorch, ONNX GPU and ONNX ARM CPU. Native QT now has an opt-in
[integer-forward ONNX export](onnx.md#native-int8-training-checkpoints), verified
against a full-FP32 CUDA reference on the local CPU provider. Subsequent
[local CUDA-provider checks](benchmarks.md#onnx-cuda-provider-placement) retain
CPU execution for native integer heads; full Blair validation preserves the five
requested metrics, while strict score parity still fails. The calibrated QDQ
recipe uses floating Conv/Gemm on CUDA and fails the small CPU/CUDA parity probe.
The subsequent [TensorRT candidate](benchmarks.md#tensorrt-int8-calibration-candidate)
executes calibrated INT8 convolutions and head GEMMs for both representative heads.
It has a separate numerical contract and mixed metric changes. Degradation of a
few percentage points is acceptable to the user when paired with a significant
measured inference speed/cost or memory benefit; quality alone does not establish
acceptance. Native-export integer GPU execution, target GPU performance and ARM
execution remain open.
ONNX Runtime training/fine-tuning
would be a separate integration; an inference export does not provide it.

The shared dataset harness now selects backbone and head independently while
preserving existing defaults. A CPU integration test trains and reloads both
EfficientNetV2-S heads on synthetic image files with a reviewed test taxonomy,
checking identical split manifests and leaf-class ordering. Reports record
initialization choice, symmetric width, normalization and image size; commands
and the target-machine handoff are in the [benchmark guide](../dev/benchmarks/README.md).
Next compare
float and quantized paths with identical splits and paired seeds, reporting
quantized versus floating operators and physical storage before making speed
claims. Profile the real backbone before selecting convolution or other storage
reductions. Keep kernel probes separate from full training and deployment results.

Accuracy superiority has not been demonstrated. The three paired dense MNIST
accuracy differences were -0.24, +0.62 and -0.34 percentage points (INT8 minus
float); Blair below is one short seed. Agree quality tolerances before evaluating
candidate recipes, retain negative results, and report uncertainty separately
from speed and memory measurements. Target-machine runs and representative-model
comparisons remain outstanding.

The first [representative EfficientNetV2-S comparison](benchmarks.md#efficientnetv2-s-on-blair-initial-representative-model-comparison)
now completes training/reload/inference for both heads and precisions on real
Blair data with random initialization. It shows only 3.1–3.4% lower peak allocation,
slower training and lower accuracy under current head-only QT, including a large
flat-head regression. This is evidence against recommending the present recipe,
not completion of the speed/quality target. It motivated the pretrained comparison
below; target-machine measurements and ONNX deployment remain open.

The subsequent [pretrained comparison and fixed-batch diagnostic](benchmarks.md#pretrained-initialization-and-fixed-batch-numerical-check)
did not reproduce the large flat accuracy drop in the same seed. INT8 still had
3–4% slower later training phases and only about 3.5% lower peak allocation.
Initial gradient discrepancies do not by themselves explain the convergence
difference. Repeated-seed quality checks, optimizer/stochastic-layer investigation
and backbone cost profiling remain necessary; no production speedup is established.

### EfficientNetV2-S structural coverage probe

On 2026-09-09, CPU-only preparation through `Classifier.build` and
`HierarchicalClassifier.build`, using `model_type="efficientnet_v2_s"`,
`num_classes=25`, `hidden=True`, `normalized=True` and
`model_args={"pretrained": False}`, produced the same storage counts:

| Quantity | Bytes / count |
| --- | ---: |
| All floating parameters before preparation (FP32) | 87,407,112 bytes |
| Selected head weights before preparation | 6,681,600 bytes |
| Selected head INT8 weights including row scales | 1,675,620 bytes |
| Reduction relative to all parameter bytes | 5.73% |
| Convolutions left floating | 170 |

`prepare_quantized_training(model)` selected only `classifier.hidden` and
`classifier.linear`. This constructed random models without downloading weights;
the hierarchical construction did not supply taxonomy masks or execute a forward
pass. It establishes operator/parameter coverage only, not hierarchical runtime
correctness, activation storage, peak training memory, convergence or throughput.

## Requirements and evidence

| Requirement | Delivered behavior and evidence |
| --- | --- |
| Actual QT rather than AMP or fake quantization | Eligible Linear weights and saved linear inputs use INT8; forward, input-gradient and weight-gradient products use integer kernels. There is no retained floating master weight. The recipe reports actual coverage and physical parameter bytes. Numerical/storage tests are in `tests/test_quantized_training.py`; integration details are in [CUDA INT8 training](quantized-training.md). |
| Lower training memory and faster training | The [three-seed batch-512 MNIST comparison](benchmarks.md#larger-batch-model-and-optimizer-graph-results), with model and optimizer graph replay, used 22.5% less peak memory and had 3.6–9.3% faster later training phases than float under the same settings. Whole-call times were lower in all three pairs. Against the faster observed float setting per seed, later-phase advantages were 2.7–4.9%. |
| Quality and realistic model coverage | CPU/float-CUDA/INT8-CUDA synthetic runs each reached the 100% oracle. MNIST uses held-out images and three paired seeds. Fresh two-level Blair runs exercised normalized hierarchical heads, hidden Linear layers, MuonAuxAdamW, AMP, compilation, optimizer graph replay, checkpoint reload and inference. Details follow below. |
| Training-state compatibility | Tests cover SGD, AdamW and MuonAuxAdamW, AMP overflow gating, scheduler advancement, the composite optimizer counter, changing learning rates, checkpoint restoration and compiled graph replay. CUDA kernel and optimizer/model matrices passed before the final loader-only change; fresh real runs also passed afterward. |
| Faster training and inference loading | [Repeated loading probes](benchmarks.md#loading-audit-on-the-delivered-implementation) verified identical batches: cached gathering was 2.59× faster without workers and 1.09× with one worker; uncached real-image inference loading was 1.14–1.41× faster. The reader/batching paths are shared independently of model weight precision. |
| Controlled loading resources | Cache construction/read-ahead are bounded; worker selection honors affinity, process limits, visible cgroup quotas and Slurm task CPU allocations. Existing caps/reserve and explicit overrides remain. Tests cover zero workers, spawn, cache modes, sample order, batch ownership, distributed sampling and quota fallbacks. |
| Usable checkpoints and inference | The native QT recipe restores parameter types before loading weights. Synthetic, MNIST and Blair runs reloaded checkpoints and produced finite held-out scores with matching class/split contracts. Separate tests cover masked rows, normalized heads and single-sample CUDA inference. |
| Repeatable continuous validation and visible results | Shared commands run CPU, synthetic QT, real-data and multi-seed graph profiles. The optional GPU workflow includes `qt-optimizer-cudagraphs`, publishes summaries and retains reports/failures for 90 days. Configuration, source/lock hashes, dataset manifests, score semantics and coverage are recorded. Current measured results are committed in [benchmarks](benchmarks.md). |
| Installed-package/default-path safety | `bash dev/check.sh all`: 347 passed, 134 skipped, one existing EMA expected failure. Static checks passed. `bash dev/check-wheel.sh` passed minimal imports, CLI/resources, CPU training, reload and prediction in a disposable installed-wheel environment; the working CUDA environment was not synchronized. |

## Fresh synthetic and hierarchical progression

The final progression used `a0a27c0`. All synthetic profiles passed their explicit
100% oracle gate. Blair used the existing reviewed 25/15-class specification,
3,704 training images, 912 validation images and 1,161 held-out images. Both
precisions used seed 42, TinyConv with a 64-feature hidden head, five epochs,
batch 32, FP16 AMP, CPU cache, zero workers, model `reduce-overhead`, optimizer
compilation and optimizer graph replay.

| Profile | Held-out fine-level accuracy | Parent accuracy |
| --- | ---: | ---: |
| Synthetic CPU | 100% | — |
| Synthetic float CUDA | 100% | — |
| Synthetic INT8 CUDA | 100% | — |
| Blair float | 64.86% | 78.12% |
| Blair INT8 | 66.24% | 81.40% |

Blair INT8 coverage is `fc.hidden` and `fc.linear`, including the normalized
head's direction parameter. Convolutions, normalization magnitudes, biases and
optimizer state remain floating. Paired manifests match and all saved scores are
finite. These short Blair runs establish functional coverage, not statistical
quality superiority or a hierarchical training speedup.

Local reports are retained in `tmp-optimizer-cudagraphs/audit/`, including the
loading repetitions, synthetic/Blair outputs, full-suite log and installed-wheel
log. Two focused loader attempts stalled in restricted-sandbox worker IPC and
were interrupted; the complete suite subsequently passed with multiprocessing
access. Reproduction commands live in [the benchmark guide](../dev/benchmarks/README.md)
and [development guide](../dev/README.md).

## Limits of the delivered capability

- Native QT execution is CUDA Linear-based. Other operations are reported as
  floating or rejected for explicit unsupported selections. CPU preparation is
  not CPU integer execution. CPU integer inference uses the separate [PTQ/QAT
  backend](quantization.md).
- Small or convolution-dominated workloads can be slower under QT. Compiler
  caches were not cleared for the paired timing study; it does not establish a
  cold-start speed advantage. The reproduced first-use tuning failure was fixed
  and validated with forced retuning, as documented in the benchmark history.
- Native QT ONNX export, checkpoint averaging, DDP/FSDP, quantized activation
  normalization and EMA are unsupported. Native fused optimizers are not claimed
  for INT8 weights. Gradients and optimizer state remain floating point.
- Controlled checkpoint continuation is covered; arbitrary stochastic resume
  does not promise identical trajectories because RNG/sampler state is not
  generally checkpointed. Real-data completion has no automatic quality gate.
- Loading ratios isolate the stated loading paths; they are not end-to-end
  inference speedups. CPU resource ceilings do not reveal contention from other
  jobs or partition a shared allocation automatically.
- Optional GPU CI requires an appropriate runner and datasets. The shared runs
  were executed locally; this audit does not claim that a remote GPU workflow
  has been dispatched or completed. Birds and iNaturalist remain stretch work.

These limits qualify the supported capability; broader hardware/operator
coverage, optimizer-state quantization and further model-quality studies remain
future increments. The requested native QT and shared-loader objective has
verified implementations and measured benefits within the supported regime.
