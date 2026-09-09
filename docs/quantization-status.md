# Quantization goal status — 2026-09-09

The goal remains **incomplete**. There are working native INT8 training and
calibrated INT8 inference paths, but the required benefit on the intended
deployment regimes has not been established. The latest completed comparison is
recorded in [the benchmark report](benchmarks.md#tensorrt-int8-versus-fp16-initial-paired-trade-off).

The acceptance principle is a joint trade-off: the user tolerates metric losses
of a few percentage points when accompanied by a substantial inference speed/cost
or memory benefit. This does not make a quality-only result, smaller engine file,
or integer operator count sufficient evidence of production readiness.

## Current evidence

| Workstream | Verified locally | What remains unproven |
| --- | --- | --- |
| Native PyTorch INT8 training | INT8 Linear weights and saved inputs, normalized symmetric flat/hierarchical heads, optimizer/AMP/checkpoint regressions, full EfficientNetV2-S updates at 100k classes; bounded preparation and initialization | Benefit on A40/A100/B300-class hardware; representative end-to-end training gains; integer convolution training; distributed QT |
| Native QT ONNX export | Generic integer-forward export; full Blair predictions and five metrics preserved on the tested hybrid CUDA/CPU path, with strict score differences | Integer head execution on GPU: CUDA falls back to CPU for MatMulInteger; TensorRT rejects the native representation |
| Calibrated TensorRT inference | Both representative heads execute 170 INT8 convolutions and two INT8 head GEMMs; full 912-image Blair evaluation; maintained calibration, build/inspection/smoke, paired timing, full-dataset collection and paired quality commands | Significant speed/runtime-memory benefit against FP16; target desktop/Spark results; automated source-dataset preparation and orchestration |
| CPU/edge inference | Calibrated ONNX CPU execution and quality measurements on x86; portable provider/timing runner | Raspberry Pi/ARM numerical behavior, sustained latency, throughput, process memory and deployment packaging |
| Continuous validation/reporting | CPU safeguards, optional GPU training workflow, visible job summaries and 90-day artifacts | Latest TensorRT/paired-engine experiments in maintained commands and continuous profiles; durable cross-device result history and acceptance gates |

Native training currently leaves convolutions, gradients and optimizer states
floating. Calibrated TensorRT inference is a separate recipe derived from floating
checkpoints; it does not prove that the native dynamic training quantizer exports
unchanged into integer GPU execution. DDP/FSDP remain unsupported for native QT.
EMA repair is explicitly deferred and is not a prerequisite for this goal.

The latest TensorRT comparison uses matched builder settings and three fresh
paired processes per head. At batches 1 and 8, INT8 did not beat FP16 in local
host latency. Engines are approximately 43% smaller, while reported execution
context memory is only 1.46% lower; total runtime memory has not been measured
reliably. CUDA-event durations conflicted with enclosing host timing and are
excluded from device-only performance conclusions.

Against FP16, INT8 Macro-F1 changes are approximately −0.40 percentage points for
the flat classifier, +0.09 for hierarchical leaves and −1.23 for parents. Recall
and Theil's U also require consideration; a small leaf F1 improvement is not a
general quality improvement. All five requested metrics are recorded in the
benchmark report. These are single-checkpoint validation results, not a multi-seed
production acceptance study.

## Remaining work, in practical order

1. **Make the latest experiments reproducible from a clean checkout.** Engine
   build, inspection and optional smoke-test parity now have a
   [maintained command](../dev/benchmarks/README.md#maintained-tensorrt-build-and-inspection-command).
   [Calibration](../dev/benchmarks/README.md#maintained-onnx-calibration-command)
   also has a maintained command, verified to reproduce both candidates' ranges,
   initializer arrays and graph nodes from the retained 128 training samples.
   [Paired inference timing](../dev/benchmarks/README.md#maintained-paired-tensorrt-timing-command)
   now retains alternating host-time pairs in a maintained command, with explicit
   pageable/pinned IO regimes. A maintained
   [quality comparison](../dev/benchmarks/README.md#maintained-paired-quality-comparison)
   validates paired prediction CSVs against an explicit held-out manifest and
   evaluates all five requested mini_metrics metrics. Maintained
   [full-dataset collection](../dev/benchmarks/README.md#maintained-full-dataset-prediction-collection)
   now feeds that evaluator for ONNX Runtime and TensorRT, with complete Blair
   replays for both heads. Connect source-dataset preparation to the calibration
   and held-out input contracts and automate the composed pipeline. Input batches
   for the real-data replay were still prepared with a local script.
   Preserve calibration records, class/preprocessing contracts, hashes, failures
   and raw timing samples. Resolve or exclude inconsistent timing sources. The
   current detailed probes and engines are retained locally under ignored `tmp-*`
   directories; documentation alone does not make them a continuous pipeline.

2. **Find and verify the beneficial workload regimes.** Pair INT8 with practical
   FP16/BF16 baselines while varying batch size, resolution and head size in a
   controlled way. Include 10k/100k-class synthetic heads alongside real-data
   quality checks; keep full million-class models off this laptop. Measure cold
   setup, steady-state throughput/latency, transfers, loading and runtime memory
   separately. Optimize the dominant measured costs rather than assuming integer
   arithmetic is faster. Include full and frozen-backbone training/fine-tuning.

3. **Settle the trained-checkpoint-to-deployment contract.** Either provide a GPU
   implementation preserving native dynamic quantization, or explicitly convert
   and calibrate a native QT checkpoint into a distinct deployment artifact and
   validate the resulting quality change. The existing float-checkpoint TensorRT
   candidate does not close that loop. Keep model configurations generic and
   verify normalized/masked/hierarchical output semantics and class ordering.

4. **Close the training and target-hardware evidence gaps.** Run the paired
   training profiles on the intended A40/A100/B300-class GPU and AMD EPYC systems,
   including actual shared-storage/loading conditions and CPU allocation limits.
   Validate convergence, time to useful quality, memory and resume behavior.
   Broaden quantized operator coverage where required to achieve the real-model
   efficiency objective. If the target workload requires distributed training,
   implement and validate distributed QT explicitly; single-GPU results cannot
   establish it. Repeat training/fine-tuning and ONNX inference on the intended
   Spark/RTX desktop separately.

5. **Validate the edge deployment on ARM.** Run the same versioned ONNX candidates
   on Raspberry Pi or the intended comparable device. Measure batch-one latency,
   sustained throughput and process memory with explicit threads and preprocessing,
   and reevaluate all five metrics. Verify the actual runtime kernels and installed
   dependency set; neither x86 CPU results nor CUDA results establish ARM support.

6. **Turn the accepted trade-offs into continuous release evidence.** Select
   concrete quality and benefit thresholds for each supported deployment profile
   once paired measurements exist. Automate synthetic correctness checks on CPU,
   representative quality/performance checks on appropriate runners, and retained
   visible reports with immutable inputs and baselines. Document supported recipes,
   limitations, installation, calibration, export and hardware-specific engine
   rebuilding. Finish compatibility review without weakening the ordinary float
   path or implying support for untested configurations.

## Completion standard

I would consider the goal finished only when the intended HPC training, local
training/fine-tuning and GPU inference, and ARM inference profiles each have a
reproducible supported path, measured worthwhile benefit against a practical
baseline, acceptable paired quality using Macro-F1/Recall/Precision, Coverage and
Theil's U, and repeatable deployment/regression reporting. Each claim must be
backed by the corresponding hardware and workload, including preprocessing,
class mappings and checkpoint behavior where applicable.

There is useful local work remaining before target machines become available;
their absence does not prevent the next implementation steps. It does prevent
final performance certification of those targets. Birds/iNaturalist, unrelated
training-feature comparisons, additional dataset formats and EMA repair are not
new prerequisites for completing this quantization goal.
