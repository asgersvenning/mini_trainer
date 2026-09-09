# Quantization goal status — 2026-09-09

The goal remains **incomplete**. There are working native INT8 training and
calibrated INT8 inference paths, but the required benefit on the intended
deployment regimes has not been established. The latest completed comparison is
recorded in [the hierarchical BatchNorm-state diagnostic](benchmarks.md#hierarchical-validation-replay-and-batchnorm-state-diagnostic).

The acceptance principle is a joint trade-off: the user tolerates metric losses
of a few percentage points when accompanied by a substantial inference speed/cost
or memory benefit. This does not make a quality-only result, smaller engine file,
or integer operator count sufficient evidence of production readiness.

## Current evidence

| Workstream | Verified locally | What remains unproven |
| --- | --- | --- |
| Native PyTorch INT8 training | INT8 Linear weights and saved inputs, normalized symmetric flat/hierarchical heads, optimizer/AMP/checkpoint regressions; maintained 100k-class BF16 capacity comparison: full-model peak 17.4% lower; corrected frozen probe peak 6.6% lower after releasing an obsolete floating parameter; mixed timing; bounded preparation and initialization | Benefit on A40/A100/B300-class hardware; representative end-to-end training gains; integer convolution training; distributed QT |
| Native QT ONNX export | Generic integer-forward export; full Blair predictions and five metrics preserved on the tested hybrid CUDA/CPU path, with strict score differences | Integer head execution on GPU: CUDA falls back to CPU for MatMulInteger; TensorRT rejects the native representation |
| QT checkpoint to calibrated GPU deployment | Explicit materialization of the matched trained INT8 checkpoints, training-only calibration, TensorRT INT8 convolution/head execution, and full Blair comparison against native and FP16 baselines | Broader configuration/large-head qualification; target-machine quality and cost/runtime-memory benefit; exact native dynamic quantization is not preserved |
| Calibrated TensorRT inference | Both representative heads execute 170 INT8 convolutions and two INT8 head GEMMs; full 912-image Blair evaluation; maintained input preparation, calibration, build/inspection/smoke, paired timing, full-dataset collection and paired quality commands | Significant speed/runtime-memory benefit against FP16; target desktop/Spark results; composed continuous orchestration |
| CPU/edge inference | Full Blair metrics and isolated process-memory/timing on x86; unsigned CPU recipe executes 170 integer convolutions and two head GEMMs, with 44–54% lower warm batch-one latency and 45–48% lower resident memory in three trials per head | Raspberry Pi/ARM numerical behavior, sustained latency, throughput, process memory and deployment packaging; larger-class qualification |
| Continuous validation/reporting | CPU safeguards, optional GPU training workflow, visible job summaries and 90-day artifacts | Latest TensorRT/paired-engine experiments in maintained commands and continuous profiles; durable cross-device result history and acceptance gates |

Native training currently leaves convolutions, gradients and optimizer states
floating. Calibrated TensorRT inference now also has an explicit route from trained
native INT8 checkpoints through floating materialization and static recalibration.
The initial float-checkpoint study and this new study are distinct. Neither proves
that the native dynamic training quantizer exports unchanged into integer GPU
execution. DDP/FSDP remain unsupported for native QT.
EMA repair is explicitly deferred and is not a prerequisite for this goal.

The dataset runner now exposes parameter-frozen fine-tuning through the existing
builder. A [five-epoch pretrained Blair comparison](benchmarks.md#pretrained-parameter-frozen-blair-fine-tuning)
completed both normalized symmetric heads with BF16 and native INT8 and evaluated
all five requested metrics on 1,161 held-out images. Macro-F1 differences were
small, with mixed changes in other metrics; this single seed is not evidence of
a general quality improvement. Local allocated peaks were 17.8% lower, but CPU
checks overlapped the runs, so no timing benefit is claimed. This regime retains
training-mode BatchNorm/dropout, unlike the synthetic evaluation-mode backbone.

An [isolated three-seed flat-head study](benchmarks.md#isolated-three-seed-flat-head-training-comparison)
now compares full and parameter-frozen pretrained training in twelve fresh
processes. Full-training INT8 was tied with or slower than BF16 and saved 3.4%
allocated peak memory. Parameter-frozen INT8 saved 17.8%, with mixed timings and
a largest observed quality loss of 2.082 points in Macro-Precision. All five
metrics and checkpoint step counts were checked. The absolute frozen-model
quality remained well below full training at five epochs, and timing drift
prevents claiming a stable speedup. The hierarchical replication below is now
complete; target-hardware replication and time-to-useful-quality evidence remain
outstanding.

The [hierarchical three-seed study](benchmarks.md#isolated-three-seed-hierarchical-training-comparison)
completes twelve corresponding training/reload runs and six two-level metric
comparisons. Memory savings match the flat study, with mixed timings. Full-training
parent Macro-F1 and Macro-Precision fall in every seed, by up to 5.10 and 5.38
points respectively; leaf changes are mixed. This negative result argues against
recommending the five-epoch recipe for its 3.4% full-training memory saving.
Longer matched-quality budgets and investigation of hierarchical optimization
sensitivity remain necessary. All final checkpoint step checks passed.

A subsequent harness audit found that all statistic loggers were disabled. The
historical epoch-summary CSVs therefore contain zeros and cannot reveal training
curves; historical `best.pt` selection also used a constant validation statistic.
The five-metric results above use independent predictions from `last.pt`, so this
does not explain their quality differences. The runner now retains its default
metric logger to support real convergence diagnostics. New timing comparisons
must include this overhead in both baselines; see the
[epoch-statistics contract](../dev/benchmarks/README.md#epoch-statistics-for-convergence-comparisons).
The correction passed static checks and the full CPU-default suite: 503 passed,
160 skipped and one known EMA expected failure. Focused tests verify all epoch
rows, positive finite losses, oracle accuracy and both hierarchical loss levels.
A one-epoch pretrained hierarchical BF16/native INT8 fine-tuning run also completed
on CUDA, reloaded its checkpoint and recorded actual statistics at both levels.
Its artifacts are in ignored `tmp-epoch-logger-int8/`; CPU checks overlapped, so
its timings are excluded from performance claims. This restores the prerequisite
for a longer convergence study; it does not supply the missing historical curves.


A [fresh twenty-epoch hierarchical pair](benchmarks.md#twenty-epoch-hierarchical-convergence-diagnostic)
at seed 42 now completes with actual epoch statistics. Held-out INT8 leaf/parent
Macro-F1 differences are +0.331/−0.082 points, with mixed changes in other metrics.
INT8 retains a 3.4% allocated-memory saving but takes 9.2% longer for the local
training call. Both checkpoints record 2,300 optimizer/scheduler updates. Strong
INT8 validation-loss spikes in the first half of training settle later; the
checkpoint replay and state ablation below investigate that behavior. One longer-budget seed neither resolves the multi-seed
quality question nor establishes target-hardware benefit.

The [checkpoint replay and BatchNorm ablation](benchmarks.md#hierarchical-validation-replay-and-batchnorm-state-diagnostic)
reproduce all six logged validation results after fresh loads. Refreshing only
backbone BatchNorm running statistics on training images reduces early INT8 loss
from 5.865 to 1.283 and raises parent accuracy from 48.81% to 85.45%, with all
parameters and non-BatchNorm buffers unchanged in the verified repeat. Float also
improves; final-checkpoint effects are much smaller and not uniformly positive.
This identifies running-statistic sensitivity as a substantial contributor in
this checkpoint, but not its origin or a universally beneficial production recipe.
Next inspect the training-state updates and qualify any proposed refresh on
additional seeds/heads with five held-out metrics and its extra execution cost.

The [100k-class optimizer study](benchmarks.md#isolated-100k-class-optimizer-compilation-comparison)
now separates eager, compiled and graph execution in eighteen isolated processes.
Ordinary optimizer compilation improves both precisions' local update times and
reduces INT8's steady allocated peak from 2.616 to 2.259 GiB. Setup peak remains
2.616 GiB; graphs reduce steady allocation further but increase reserved memory
and leave INT8 slower than graph float in all three trials. This is a synthetic
capacity/execution result, not dataset quality or target-hardware acceptance.

The earlier floating-checkpoint TensorRT timing comparison uses matched builder settings and three fresh
paired processes per head. At batches 1 and 8, INT8 did not beat FP16 in local
host latency. Engines are approximately 43% smaller, while reported execution
context memory is only 1.46% lower; total runtime memory has not been measured
reliably. CUDA-event durations conflicted with enclosing host timing and are
excluded from device-only performance conclusions.

In that earlier study, against FP16, INT8 Macro-F1 changes are approximately −0.40 percentage points for
the flat classifier, +0.09 for hierarchical leaves and −1.23 for parents. Recall
and Theil's U also require consideration; a small leaf F1 improvement is not a
general quality improvement. All five requested metrics are recorded in the
benchmark report. These are single-checkpoint validation results, not a multi-seed
production acceptance study.

The new native-checkpoint conversion study evaluates all 912 Blair validation
images at four stages: native integer ONNX, materialized floating ONNX, TensorRT
FP16 and TensorRT INT8. Against matched FP16, INT8 Macro-F1 changes are −0.303,
+0.409 and −0.106 percentage points for flat, hierarchical leaf and parent
outputs; hierarchical leaf precision drops 2.241 points. No new latency result
was measured for these checkpoints. Regression validation passed all static
checks and **470 tests**, with **152 skips** and **one known EMA expected failure**.

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
   replays for both heads. Maintained
   [image preparation](../dev/benchmarks/README.md#maintained-image-input-preparation)
   now reproduces every retained calibration and validation NPZ hash for both
   heads from source images and export metadata. A
   [paired quality pipeline](../dev/benchmarks/README.md#paired-inference-quality-pipeline)
   now runs both collectors and the evaluator in fresh processes, retaining logs,
   failure reports and a Markdown summary, with separately selectable runtime and
   metric interpreters. A [composed CPU deployment check](../dev/benchmarks/README.md#composed-cpu-deployment-comparison)
   now connects full quality, candidate operation requirements and repeated
   fresh-process memory/latency trials, checks artifact hashes across phases and
   produces a combined Markdown summary. Preparation, calibration, GPU builds and
   GPU resource checks still need composition; package the explicit optional environments.
   The default preparation factory
   uses current architecture-loader transforms; custom preprocessing still needs
   an explicit reviewed factory and input verification.
   Preserve calibration records, class/preprocessing contracts, hashes, failures
   and raw timing samples. Resolve or exclude inconsistent timing sources. The
   current detailed probes and engines are retained locally under ignored `tmp-*`
   directories; documentation alone does not make them a continuous pipeline.
   The [training-prediction adapter](../dev/benchmarks/README.md#training-predictions-to-paired-quality-evaluation)
   now also feeds saved flat/hierarchical training predictions into the shared
   five-metric evaluator without checkpoint or image transfers. It reproduced the
   pretrained fine-tuning metrics exactly; continuous orchestration and durable
   result publication remain to be completed.
   The [representative training profile](../dev/benchmarks/README.md#representative-paired-efficientnetv2-training-profile)
   now composes paired EfficientNetV2 training, prediction preparation, all five
   metrics and visible summaries in one shared command, with selectable heads,
   full/frozen modes, seeds and epoch budgets. Scheduled runner wiring, durable
   hosting and deployment orchestration remain unfinished.
   Validation passed static checks and 508 CPU-default tests, with 160 skips and
   the known EMA expected failure. A bounded one-epoch hierarchical frozen
   float/INT8 Blair pair completed through the shared command, including both
   levels of mini_metrics and the combined summary. Reports remain in ignored
   `tmp-representative-profile-smoke/`. CPU checks overlapped this smoke run, so
   it supplies integration evidence only, not new performance or acceptance claims.

2. **Find and verify the beneficial workload regimes.** Pair INT8 with practical
   FP16/BF16 baselines while varying batch size, resolution and head size in a
   controlled way. Include 10k/100k-class synthetic heads alongside real-data
   quality checks; keep full million-class models off this laptop. Measure cold
   setup, steady-state throughput/latency, transfers, loading and runtime memory
   separately. Optimize the dominant measured costs rather than assuming integer
   arithmetic is faster. Include full and frozen-backbone training/fine-tuning.
   The [100k-class BF16 capacity study](benchmarks.md#maintained-100k-class-bf16-training-comparison)
   covers both modes and heads across three seeds. Its apparent frozen-mode
   regression was a retained floating parameter in the probe; the
   [corrected allocation comparison](benchmarks.md#correcting-the-frozen-head-allocation-comparison)
   shows a 6.6% local peak reduction for both heads. Repeat the corrected probe
   across seeds and with realistic pretrained features and longer integrated training.
   Both flat and hierarchical pretrained Blair profiles now have isolated
   three-seed comparisons against full-backbone baselines. Extend them to longer
   matched-quality budgets; the current five-epoch frozen models do
   not reach the full-training quality level. Investigate measured execution costs
   and larger-head regimes rather than inferring a general speedup from these runs.

3. **Qualify the trained-checkpoint-to-deployment contract.** Explicit conversion
   and calibration now connect the matched native QT checkpoints to distinct
   TensorRT INT8 artifacts. Both real heads have full-data comparisons against
   the native and materialized FP16 baselines; see the
   [conversion study](benchmarks.md#native-int8-checkpoint-to-calibrated-tensorrt-deployment).
   Extend qualification to the large-head/target profiles and verify worthwhile
   efficiency. Keep normalized/masked/hierarchical semantics and class ordering
   under regression coverage. Unsupported tied parameter roles/views must fail
   explicitly. An exact native dynamic-quantizer GPU implementation remains a
   different, unimplemented route, not a claim made by materialization.

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
   The [isolated Linux CPU study](benchmarks.md#isolated-linux-cpu-memory-and-inference-study)
   now supplies a maintained RSS/PSS/peak and load/first/warm inference probe.
   The initial signed candidate saved memory but retained 107 floating convolutions
   and was slower. A [CPU-specific unsigned recipe](benchmarks.md#cpu-specific-activation-and-bias-calibration)
   now executes all 170 convolutions as integer operations locally, with substantial
   latency/memory savings and a largest observed quality loss of 2.956 percentage
   points in hierarchical leaf precision. Repeat on real ARM hardware with preprocessing and sustained-load
   conditions, rather than extrapolating the local memory percentage.

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
