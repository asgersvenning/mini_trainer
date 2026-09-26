# Quantization roadmap

The implementation is merged. **Useful target-machine trade-offs remain open.**
Do not repeat completed branch cleanup or broad laptop sweeps as a prerequisite.
[Measured findings](benchmarks.md), [commands](../dev/benchmarks/README.md) and
[artifact retention](quantization-artifacts.md) are the maintained evidence and
execution references; detailed historical experiments remain in
[the recorded snapshot](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md).

## Supported scope and decision criteria

Implemented: opt-in native CUDA INT8 Linear training with normalized symmetric
heads, checkpoint safeguards, floating materialization/calibration, generic ONNX
export and CPU/TensorRT placement, quality and resource reporting. Defaults remain
floating point. Native QT DDP/FSDP, integer convolution training, exact native
quantizer GPU ONNX execution and ARM production performance are not established.
EMA remains unsupported.

Evaluate EfficientNetV2-S flat and hierarchical heads, full and frozen training,
with reviewed data/splits and paired seeds. Synthetic/oracle and 10k/100k-class
capacity checks establish correctness/capacity, not image-level quality.
Use practical floating baselines: supported AMP training, FP16 TensorRT and floating
ONNX CPU. Through `mini_metrics`, retain macro F1/recall/precision, coverage and
Theil's U at leaf and parent ranks. Set workload-specific quality/resource gates
before a run; a small quality loss may be useful only with a measured resource gain.

| Target | Question and required evidence |
| --- | --- |
| HPC A40/A100/B300-class + EPYC | Does full/frozen training reach comparable quality sooner or with useful memory savings? Include startup, loading/transfers, allocated/reserved memory and resume correctness. |
| Intended RTX desktop or Spark | Same training question; qualify target-built ONNX/TensorRT inference with real operator placement, batches, end-to-end cost and memory. |
| Raspberry Pi/comparable ARM | Does installed ONNX CPU inference fit memory and sustain useful batch-one performance, including preprocessing, threads and thermal/power conditions? |

Record the actual device/allocation, OS, driver/runtime and installed packages.
Available hardware determines the next bounded comparison; the table is not a
requirement to obtain every listed architecture before making progress.

## Next target run

1. Prepare an explicit backend environment and reviewed inputs. Preserve dataset,
   split, taxonomy, preprocessing and model hashes; use training-only calibration.
2. Run a small construction/train-step/checkpoint/inference and placement pilot.
   Reject unsupported execution before a large comparison.
3. Use `bash dev/check-benchmarks.sh qt-efficientnet FRESH_RESULTS` for the paired
   training matrix, with [documented variables](../dev/benchmarks/training.md).
   Its default three-seed, five-epoch comparison is not settled convergence evidence.
4. Use `bash dev/check-tensorrt-deployment.sh FRESH_RESULTS` for target-built engines,
   or `python -m dev.benchmarks.inference.cpu_deployment` for CPU. Follow the
   [inference guide](../dev/benchmarks/inference.md).
5. Retain success/failure reports, inputs and identities. Separate cold setup,
   warm compute and end-to-end costs; use repeated uncontended runs where variation
   affects the decision. Return evidence through the existing reporting format.

## Choose changes from the limiting cost

| Existing evidence or boundary | Next useful action |
| --- | --- |
| Head-only QT gave about 3.4% peak savings in full EfficientNet training, without reliable local speed gains. | Profile the full target workload. Expand integer coverage only if the unquantized cost dominates; report physical storage and end-to-end benefit. |
| Large normalized heads have transient normalization/gradient and optimizer costs. | Preserve bounded preparation/backward behavior; qualify realistic head sizes and optimizer memory instead of optimizing an isolated GEMM. |
| Compilation/graphs trade startup and reserved memory for steady-state speed. | Include setup, first step and total useful work. Recorded DDP stride, recompilation and hierarchy graph-break warnings are leads, not proven bottlenecks. |
| Short hierarchical QT runs lose parent metrics; BatchNorm changes had mixed results. | Compare longer paired budgets/seeds and time to quality. Do not adopt automatic BN refresh without evidence including its cost. |
| Native dynamic INT8 ONNX falls back to CPU for MatMulInteger; TensorRT rejects that representation. | Qualify materialize-then-calibrate first. Exact native integer GPU lowering is a separate feature, with actual provider placement as acceptance. |
| Smaller INT8 TensorRT engines were not reliably faster than FP16. | Measure target tactics, batches, transfers and runtime memory; serialized size is not a speed result. |
| ARM runtime and sustained behavior are unverified. | Test the real device and unsigned-activation CPU recipe before claiming edge support. |
| Shared-storage/CPU contention can dominate. | Tune bounded reads, workers and transfer overlap within the allocation; preserve order and measure whole training. |

Retain export score/decision checks. A profile-specific large-head tolerance
exception does not justify weakening the generic exporter gate. Distributed native
QT needs separate parameter/communication/sharding/optimizer/resume qualification;
floating DDP tests do not establish it.

## Integration and deferred work

Reuse existing export manifests for portable bundles: preprocessing, mappings,
score semantics, calibration provenance, hashes and a minimal installed example.
Keep optional runtimes lazy and validate installed wheels when packaging changes.
MAMBO's floating bundle offers a packaging example, not quantized qualification.

[Continuous reporting](../dev/benchmarks/reporting.md) has local/simulated evidence;
remote activation still needs configured runners/Pages, reviewed
`ENABLE_BENCHMARK_HISTORY`, stored-asset readback and live success/failure/retry
checks. Preserve training/inference and CPU/GPU scopes and visible missing results.
Define automated quality/resource regression gates from the paired target evidence.

EMA repair, unrelated feature/dataset expansion and dashboard cosmetics are separate
work. New precision formats or distributed backends belong here only when required
by the target workload. Negative results should narrow supported claims, not trigger
another indiscriminate experiment matrix.
