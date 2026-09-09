# Quantized training and loading validation

This audit covers the delivered objective: genuine quantized training that lowers
memory and increases training speed on supported workloads, plus faster shared
loading for float and quantized training/inference. The capability is opt-in;
speedups are workload-dependent. Validation was completed on 2026-09-09 using
Python 3.13.7, PyTorch 2.12.0/CUDA 13.0, TorchAO 0.17.0 and an RTX 3080 Ti Laptop GPU.

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
