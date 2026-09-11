# Benchmark findings

These are the current conclusions of the quantization branch, based on local
measurements. They do not certify HPC, Spark/desktop, or ARM performance.
The [branch roadmap](quantization-roadmap.md) defines the remaining work and
completion criteria. Exact configurations, trials and superseded findings remain
in the [historical experiment record](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md).

## Representative workload

EfficientNetV2-S, symmetric hidden layer and normalized flat or hierarchical
classifier; reviewed Blair splits and paired seeds. Full training and
parameter-frozen fine-tuning are different workloads. Synthetic 10k/100k-class
heads measure capacity, not classification quality. Full million-class runs are
reserved for larger machines.

## Evidence that drives the next decisions

| Regime | Measured finding | Consequence |
| --- | --- | --- |
| Native INT8, full pretrained Blair training | About 3.4% lower allocated peak memory; timing tied or worse in short three-seed studies. Five-epoch hierarchical parent F1 fell in all three seeds, up to 5.10 points; precision up to 5.38 points. | The short full-training recipe is not a demonstrated useful trade-off. Profile backbone costs and qualify time to comparable quality. |
| Native INT8, parameter-frozen Blair fine-tuning | About 17.8% lower allocated peak; mixed timings. Five-epoch absolute quality was below full training. | Qualify fine-tuning separately; memory savings do not establish equal-quality training efficiency. |
| Longer hierarchical training | One 20-epoch pair: leaf/parent F1 differences +0.331/−0.082 points, but INT8 training took 9.2% longer locally. | Longer budgets can change the quality result. Repeat seeds and measure time to useful quality on target hardware. |
| BatchNorm refresh | Training-only refresh improved some early checkpoints; twelve-checkpoint qualification had mixed held-out effects and remaining parent-level regressions. | No automatic refresh is recommended. Running-statistic sensitivity is a diagnostic finding, not a demonstrated universal fix. |
| 100k-class native training capacity | Bounded normalization backward reduced full-training peak allocation by 17.4%; corrected frozen comparison saved 6.6%. Compiler/graph variants had mixed speed and reserved-memory results. | Large heads can benefit, but distinguish allocated, reserved, setup and steady-state memory. These are synthetic capacity results. |
| ONNX CPU inference on x86 | Unsigned activation/per-channel INT8 recipe executed 170 integer convolutions and two head GEMMs; three trials per head showed 44–54% lower warm batch-one latency and 45–48% lower RSS. Largest observed metric loss was 2.956 points in hierarchical leaf precision. | Promising edge candidate; verify kernels, quality and sustained resources on ARM before recommending it there. |
| TensorRT inference with 100k random classes | INT8 engine about 49% smaller. Batch 1/8 was slower; batch 64 was effectively tied. Batch-64 device snapshots were 140 MiB lower: 37% of the post-initialization increment, 9% of total warm usage. | A local memory-saving candidate, not a demonstrated speedup or trained large-vocabulary quality result. |
| Native QT export | Integer-forward ONNX is supported in a limited tested runtime path, but CUDA MatMulInteger falls back to CPU and TensorRT rejects the native representation. Explicit floating materialization followed by static calibration provides a separate deployable route. | Do not claim the calibrated export preserves the native dynamic quantizer. Validate its quality and placement independently. |
| Large-head float export parity | Some near-zero scores required an explicit 1e-4 absolute tolerance; FP64 diagnostics were consistent with accumulated rounding. | Keep the public default parity gate. Qualify any profile-specific tolerance with numerical evidence. |

## Measurement rules

- Compare practical floating baselines: BF16/FP16 training where supported,
  FP16 TensorRT and floating ONNX CPU. Use matched data, preprocessing and budgets.
- Evaluate Macro-F1, Macro-Recall, Macro-Precision, Coverage and Theil's U with
  `mini_metrics`, at leaf and parent levels. Retain undefined values and negative
  results. Differences multiplied by 100 include Theil's U; label that convention.
- Accept a few points of quality loss only alongside a substantial measured
  speed/cost or memory benefit in the intended workload.
- Separate cold setup, first execution, warm latency, end-to-end throughput,
  loading/transfers, allocated/reserved GPU memory and process/device snapshots.
  Device-wide snapshots are not transient peaks or per-process allocation.
- Runs overlapping CPU checks do not support timing/resource claims. Random
  large-class heads and synthetic ARM metadata do not establish real quality or
  ARM execution.
- Historical epoch CSVs generated while all statistic loggers were disabled are
  invalid convergence evidence; their independently evaluated `last.pt` results
  remain usable. `best.pt` selection from those runs is not trustworthy.

## Reproduction and visibility

Use the [benchmark command index](../dev/benchmarks/README.md):
[training](../dev/benchmarks/training.md),
[inference](../dev/benchmarks/inference.md), and
[reporting](../dev/benchmarks/reporting.md).
CPU and TensorRT composed reports bind quality to retained runtime evidence;
compact history distinguishes CPU medians/RSS from GPU paired timing/snapshots.
New CPU reports also retain and verify the requested trial budget.

An opt-in default-branch publisher connects compact TensorRT artifacts to monthly
draft-release storage and GitHub Pages. Live activation, authenticated storage,
public rendering and target-runner execution remain unverified. CPU producer and
training-history handoffs are follow-up integration work, not established services.

Raw temporary paths cited by older notes are now governed by the
[artifact-retention record](quantization-artifacts.md). Downloaded datasets,
original research files and the working environment are separate from disposable
benchmark outputs.
