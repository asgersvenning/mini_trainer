# Quantization feature-branch roadmap

This is the current plan for `quant`. The local implementation and evidence are
substantial, but the goal remains incomplete until useful trade-offs are verified
on the intended hardware. After repository consolidation, pause further broad
laptop sweeps and reporting expansion. Resume with a specific bottleneck,
target run or integration requirement from the sequence below.

## Scope and acceptance

Use EfficientNetV2-S with symmetric hidden layers and normalized flat and
hierarchical classifiers. Evaluate full training and parameter-frozen fine-tuning
separately. Use reproducible synthetic/oracle checks, MNIST integration and reviewed
Blair splits; use 10k/100k-class synthetic heads for capacity. Larger class counts
belong on appropriately provisioned hardware. Synthetic capacity does not prove
large-vocabulary classification quality.

| Target | Required execution | Evidence needed to finish |
| --- | --- | --- |
| HPC: A40/A100/B300-class systems with AMD EPYC | PyTorch training | Paired convergence/time to comparable quality, throughput, cold/setup costs, host and allocated/reserved device memory, actual loading/transfer costs, checkpoint/resume correctness. Record each tested GPU/runtime/allocation. |
| Local: NVIDIA Spark or intended RTX desktop/Ryzen system | PyTorch training/fine-tuning and ONNX GPU inference | Full/frozen training evidence plus target-built inference engines, actual integer placement, paired quality, realistic batches, end-to-end inference cost and memory. Verify exact installed device and architecture first. |
| Edge: Raspberry Pi or comparable ARM device | ONNX CPU inference | Installed runtime/operator compatibility, five-metric quality, batch-one latency, sustained throughput, process memory, preprocessing and deployment packaging under realistic thread/power/thermal conditions. |

The quality contract is Macro-F1, Macro-Recall, Macro-Precision, Coverage and
Theil's U through `mini_metrics`, including parent levels. A few points of loss
can be acceptable with a substantial measured speed/cost or memory benefit.
Choose and record per-profile quality/resource gates before qualification; do not
hide a failed metric behind accuracy or engine size. Keep baselines practical:
BF16/FP16 training where supported, FP16 TensorRT, and floating ONNX CPU.

## 1. Finish local preparation and handoff

The implementation, current findings, command guides, grouped tests and retained
evidence should form a reviewable branch milestone. This stage does not require
new target hardware or another performance sweep.

- Run static checks and the complete grouped suite; preserve test collection,
  checkpoint fixture imports and spawn behavior. Keep optional/GPU skips explicit.
- Retain essential trained final models, deployable ONNX sources, predictions,
  reports, input identities and scripts; remove disposable duplicate/intermediate
  models and laptop-specific engines. See [artifact retention](quantization-artifacts.md).
- Use existing [training](../dev/benchmarks/training.md),
  [inference](../dev/benchmarks/inference.md) and
  [reporting](../dev/benchmarks/reporting.md) commands as the handoff. Avoid a
  second implementation of model construction, evaluation or configuration.
- Keep the branch unmerged until reviewed. Repository settings and public
  publication require a deliberate activation step after review.

Exit: clean committed branch, unchanged collected test cases, passing checks,
concise documentation, and an inventory of retained/restorable evidence.

## 2. Obtain target runs and establish practical baselines

**Dependency:** access to the target machines, or an operator who can run the
commands and return the full artifacts. No local measurement can remove this
dependency. Record GPU/CPU identity, architecture, memory, OS, driver/runtime,
PyTorch/ONNX versions, thread/worker limits and allocation conditions.

1. Prepare explicitly selected backend environments; do not synchronize away a
   working CUDA environment. Verify TorchAO/kernel support on the actual GPU and
   host architecture. Keep `mini_metrics` in an explicit compatible interpreter.
2. Stage the reviewed dataset/split/taxonomy and training-only calibration inputs.
   Keep preprocessing and input/model hashes identical across each pair.
3. Run a bounded correctness pilot: model construction, a training step, checkpoint
   reload, inference and operator placement. Reject unsupported execution before
   launching a long benchmark.
4. Run `bash dev/check-benchmarks.sh qt-efficientnet FRESH_RESULTS` with the documented
   data/interpreter variables. Its default covers both heads, full/frozen training,
   BF16/INT8 and three seeds. Select longer budgets explicitly where needed;
   five-epoch quality is not a settled convergence comparison.
5. Rebuild TensorRT engines on the target with
   `bash dev/check-tensorrt-deployment.sh FRESH_RESULTS`. On ARM, use
   `python -m dev.benchmarks.inference.cpu_deployment` with the reviewed baseline/candidate,
   manifest and representative inputs. Begin with explicit conservative threads.
6. Collect fresh-process repeated runs without competing benchmark jobs. Include
   cold startup, warm execution and end-to-end input handling as separate scopes.
   Return raw reports and artifacts even when a stage fails.

Exit: reproducible floating and quantized baseline evidence for the actual target;
an identified dominant cost or a demonstrated useful trade-off. Scope claims to
tested configurations rather than extrapolating across hardware families.

## 3. Unlock measured performance bottlenecks

The [findings](benchmarks.md) summarize the evidence behind these priorities.
Choose the next row using target profiles; do not implement all possible precision
formats or distributed backends merely because they are available.

| Observed bottleneck | Action that can unlock it | Verification gate |
| --- | --- | --- |
| Full EfficientNet training gets only about 3.4% peak savings from head-only QT, with no reliable local speed gain | Profile the entire model, backward pass, optimizer, loading and transfers on the target. If convolution/activation traffic dominates, select and implement a supported deeper quantization path for that cost; if the head dominates, focus on its kernels/storage. | Report actual quantized/floating coverage and physical storage. Demonstrate end-to-end benefit at comparable quality; AMP or fake quantization alone does not complete this work. |
| Large normalized heads incur transient normalization/gradient storage and optimizer overhead | Preserve bounded preparation/backward fixes. Profile realistic head sizes, batching and fused update paths; consider optimizer-state reduction only if its footprint is limiting. | Fresh-process setup/steady-state measurements; allocated and reserved peaks; numerical, accumulation, optimizer-step and checkpoint regressions. Do not optimize only an isolated GEMM. |
| Compilation/CUDA graphs can improve steady state while increasing setup or reserved memory | Measure eager, compiled and graph modes separately. Qualify first-use tuning and cold setup on the target; reduce capture/tuning overhead only where it matters. | Include initialization and first step, replay correctness, total useful-work time and reserved memory. A warmed kernel speedup is insufficient. |
| Short hierarchical QT runs lose parent metrics; early BatchNorm sensitivity is substantial | Run longer paired budgets/seeds with working epoch statistics. Investigate precision/optimization sensitivity at fixed data and model state. Qualify any state-refresh or recipe change with its extra cost. | Time to comparable leaf/parent quality across seeds. Do not adopt automatic BatchNorm refresh from the mixed existing results. |
| Native dynamic INT8 ONNX uses MatMulInteger CPU fallback on CUDA; TensorRT rejects that representation | Qualify the explicit materialize-then-calibrate route first. If exact native quantizer semantics are required, implement a supported integer GPU lowering/kernel as a separate feature. | Actual provider/operator placement and end-to-end quality against the correct reference. Never silently label floating/fallback execution as integer GPU inference. |
| Large-head float export needs a profile-specific tolerance override | Continue numerical attribution with higher-precision references on the failing target/profile. Distinguish accumulated rounding from an export/operator defect. | Explicit score/parity and decision/metric evidence; retain the public default gate until a justified general change exists. |
| INT8 TensorRT engines are smaller but can be slower; local batch-64 results are near tied | Profile target batch sizes, transfers, tactics and head/backbone balance against FP16. Measure actual runtime memory and sustained throughput, not just serialized size. | Significant useful speed/cost or memory benefit with acceptable paired quality on the intended workload. |
| ARM kernels/runtime, thermal behavior and packaging are unverified | Prepare the actual ARM environment and inspect unsigned-activation CPU recipe placement; benchmark sustained batch-one inference including preprocessing. Adjust calibration/operator choices only from measured failures. | Repeated ARM measurements, all five metrics, memory within device limits, and an installed deployment example. x86 savings do not qualify ARM. |
| Shared storage/loading or CPU oversubscription may dominate training | Measure loader wait and host/device transfer on the real allocation. Tune explicit threads/workers, caching and prefetch/transfer options within resource limits. | Identical sample/order semantics, bounded host/shared memory, and improved end-to-end training rather than loader-only throughput. |

If production size requires multiple GPUs, native QT DDP/FSDP is a separate
unsupported boundary to implement and qualify. Check parameter representation,
communication/reduction, sharding, optimizer state and distributed resume. Existing
floating DDP tests do not establish distributed QT support.

## 4. Close development and integration bottlenecks

| Dependency or gap | Required action | Done when |
| --- | --- | --- |
| Target-only optional runtimes and backend versions | Record supported combinations; preserve lazy imports and actionable errors; run installed-wheel validation after dependency/packaging changes. | Clean installation and intended kernels work on each claimed target without changing floating defaults. |
| Target inputs/models currently require explicit staging | Prepare a versioned deployment bundle with preprocessing, mappings, score semantics, calibration provenance, model hashes and a minimal inference example. Reuse current export manifests. | Another operator can reproduce the target run without private notebook state or this laptop's paths. |
| Continuous CPU/training handoff is incomplete | Connect the existing CPU command and representative training reports to configured runners and compact history. Add only the adapters needed for real collected evidence. | Failed and successful target runs retain attributable reports; history preserves CPU/GPU and training/inference measurement scopes. |
| Release storage/Pages has only local and simulated validation | After branch review/merge, configure Pages and target runner variables, then enable `ENABLE_BENCHMARK_HISTORY`. Run one live success, failure and publishing retry. | Stored asset readback, immutable run identity, correct public page and recoverable failure verified remotely. |
| Quality/resource acceptance is not automated | Define per-profile gates from practical paired evidence and the agreed few-points trade-off. Preserve all five metrics and uncertainty/repetition context. | A regression is visible and fails the appropriate check; unsupported/untested profiles cannot look production-qualified. |

## 5. Completion and deferred work

Finish with supported opt-in recipes for the three target regimes, reproducible
quality/efficiency evidence, functioning checkpoint/export/deployment contracts,
visible continuous results, and a final compatibility/packaging review. A negative
result should constrain the supported recipe or motivate a measured implementation
change; it must not be reported as a general speedup.

EMA repair, unrelated optimizer/loss/augmentation comparisons, new dataset formats,
Birds/iNaturalist expansion and cosmetic dashboard work remain deferred. Broader
operator or distributed support becomes necessary when the target workload needs
it to achieve the stated goal, not as an unconditional expansion of this roadmap.
