# Repository strengthening roadmap

This is the implementation order. Each increment should leave the existing default
training and prediction interfaces working and include its own validation evidence.
Items below are planned unless explicitly marked delivered.

## 1. Agent guidance and development safeguards

Delivered in the initial increment:

- Root `AGENTS.md` with repository context, compatibility boundaries, environment
  handling, and validation requirements.
- Shared `dev/check.sh` commands for local checks and CI, including import contracts.
- Explicit no-sync validation after CI selects and installs CPU dependencies.
- Contributor documentation explaining CPU, GPU, slow-backbone, and sandbox limits.

Delivered in the validation increment:

- Consolidated `.agents/rules/` references and a focused repository maintenance skill.
- Minimal installed-wheel checks covering imports, CLI help, packaged resources, and
  image-folder training/reload/prediction outside the source tree.
- Tracked `uv.lock`, locked regular CI, and a separate scheduled/manual workflow for
  latest-compatible dependency resolution. Both cover Python 3.12, 3.13, and 3.14.

Acceptance: a fresh checkout has one documented path to validation; architecture
checks do not initialize CUDA; local and CI commands match; missing optional
integrations do not prevent core imports or CLI help.

## 2. Behavior-preserving simplification

Preparation delivered: checkpoint contract tests for live/reloaded prediction equality,
state restoration, and deterministic CPU continuation with the original epoch budget,
fixed data order, and stochastic transforms disabled. EMA restoration is tested separately.
Active AMP, GPU execution, and arbitrary RNG/sampler continuation are not covered.

Known failure discovered by this coverage: full EMA continuation fails after evaluation
populates `Classifier._linear_weight` and `_linear_bias` caches. These nonpersistent
buffers can have different shapes on the EMA and training models, breaking the next
averaging update. A strict expected-failure test records the problem. EMA is temporarily nonfunctional
and unsupported, with a runtime warning when enabled. Repair is explicitly deferred;
leave it disabled and exclude it from current feature comparisons.
The validation increment changed no training behavior.

Loader increment delivered: shared resize validation and worker-selection helpers,
removal of an unreachable label-conversion branch and obsolete resampling comments,
and regression coverage for shapes, labels, sampling, subsampling, and worker settings.
Resize tuple ordering and existing error messages are preserved.

The accompanying worker-selection fix uses process CPU availability and affinity
instead of the whole node's CPU count when available. Existing training/inference
caps and headroom are retained, as are explicit counts and the CUDA-cache override.
RAM-cache preloading uses the same CPU detection and now always selects at least one
thread, including on one-CPU systems. This does not account for every CPU quota or
competing job; shared unrestricted allocations still need explicit worker settings.

Next, review the orchestration boundaries in `builders.py`, `train.py`, and `trainer.py`
only where an extraction has a concrete benefit. Review checkpoint and metadata
boundaries separately.

Before each extraction, cover observable behavior: class order, labels, shapes,
dtypes, device placement, sampling, errors, return values, and public imports.
Use synthetic training/resume and CPU DDP integration tests for cross-cutting changes.
Do not combine cleanup with changed defaults, new dependencies, or numerical fixes.

Acceptance: focused regression tests and existing integration tests pass; import
contracts remain intact; existing configurations and checkpoints remain usable.

## 3. ONNX export and Hugging Face integration

Delivered: a generic export API and `mt_export` CLI with optional dependencies.
The actual evaluation forward is exported without architecture/head allowlists,
including structured and hierarchical outputs, masks, priors and normalized heads.
Dynamic batch parity, caller-state preservation, artifact manifests and standalone
ONNX Runtime inference are checked. See [the export guide](onnx.md) for the input,
preprocessing and score contract, representative coverage and operator limitations.

A separate eager-inference fix uses the backbone embedding width before an explicit
hidden layer; a core regression covers vector and singleton-spatial embeddings.

Acceptance evidence covers CPU float32 on representative offline backbones and all
head families, not every catalog variant or GPU/quantized provider. Preprocessing
remains external and requires a caller-supplied deployment recipe.

Then add local Hugging Face bundle preparation: weights/export, manifest, model card,
evaluation summary, and an inference example. Treat Hub artifact hosting and a live
inference service as separate deliverables. Validate a local bundle before adding
explicit upload commands or a serving deployment.

## 4. Training efficiency and augmentation

The primary implementation target is **actual quantized training and faster data loading**.
QT must reduce retained training storage and demonstrate lower peak memory and faster
training on supported workloads. QAT with floating-point master weights is a separate
capability and does not complete this target. The initial CUDA integer forward/backward
kernel probe and cached-loader benchmark are documented in [the benchmark guide](../dev/benchmarks/README.md).
Optimizer support, checkpoint integration, real-data convergence and end-to-end
measurements remain required before claiming a supported QT training path.
Loader hardening, float16/bfloat16 AMP and benchmark infrastructure do not complete
that target. The implementation and comparison plan is in
[training feature validation](training-feature-validation.md).

Deliver quantization-aware training and post-training inference quantization as
separate opt-in capabilities, recording actual weight/activation bit widths,
calibration data, backend kernels, checkpoint/resume and export/runtime support.
FP8 or other reduced-precision compute is a separate hardware-dependent profile.
Keep unsupported model/backend combinations explicit; do not silently run an
unquantized model while reporting a quantized result. EMA support is not a gate
for this work while the feature is declared defunct.

Improve the default augmentation pipeline after paired task-aware experiments,
retaining a reproducible legacy recipe. Respect label semantics: digit tasks,
color-based synthetic tasks and biological imagery need different invariances.
Document uint8 input and augmentation-before-preprocessing behavior accurately.

Benchmark foundation delivered: the [continuous dataset benchmark suite](benchmarks.md)
uses a synthetic oracle, MNIST and hierarchical Blair, with CPU and GPU/AMP/cache
profiles, explicit split/provenance records, Actions summaries and retained artifacts.
This is a verifiable milestone, not a CPU-only scope boundary. Add durable hosted
history, repeated comparisons and coverage of the remaining training features next.

Next establish repeatable measurements for loader wait time, images/second, host/GPU
memory, and validation quality on fixed configurations. Existing code already uses
autocast, optional compilation, persistent workers, cache modes, and DDP spawn handling.

Add opt-in loader controls only where measurements justify them (for example,
prefetching and transfer overlap), with coverage for zero workers, cache modes,
deterministic seeding, and distributed sampling. Extend augmentation through the
existing builder interface, documenting label and dtype requirements.

Evaluate lower-precision training, quantization-aware training, and inference
quantization as distinct paths. Keep current defaults and checkpoint compatibility;
validate optimizer/resume behavior (EMA is deferred), numerical stability, hardware support, export
compatibility, memory, throughput, and quality before recommending a configuration.

Acceptance: reproducible baseline and comparison results, explicit supported hardware
and backends, opt-in configuration, and no regression in default training behavior.

## 5. mini_metrics and continuous model evaluation

Document measured effects of MuonAuxAdamW versus AdamW/SGD, `normalized`,
EMLACrossEntropy, class-weight distribution regularization, automatic label smoothing,
and hierarchical versus flat classifiers. Use the same datasets/splits and paired
seeds; retain negative and null results. See the [comparison protocol](training-feature-validation.md).
Optimizer step tracking for AdamW/SGD is now fixed, including native fused AMP skips.
CPU/CUDA regressions preserve scheduler/EMA gating and batch-based EMA indices;
checkpoint continuation is covered for MuonAuxAdamW, AdamW and SGD. Comparative
quality experiments remain planned; EMA itself remains unsupported.


Use `publication/experiments/statistics/boot_metrics.py` and prediction CSV output as
the current integration boundary. The adjacent `../mini_metrics` checkout currently
declares Python >=3.13 while mini_trainer supports >=3.12. Resolve this with a separate
evaluation environment or a verified compatible release before adding an extra;
never rely on a sibling checkout being present in installations.

Define and test the prediction/evaluation contract: sample identity, ground truth,
class order, score meaning, confidence thresholds, missing classes, and hierarchical
labels. Verify installed mini_metrics APIs before writing an adapter. Keep evaluation
optional and preserve existing prediction formats and research scripts.

Build the model zoo around versioned manifests linking immutable weights, configuration,
preprocessing, dataset/split identity, code/dependency versions, and evaluation results.
Use fixed held-out data and seeds; distinguish threshold selection from test evaluation.
Add tiny fixture-based contract checks to pull requests, then explicit or scheduled
evaluation of registered artifacts with retained reports and declared regression limits.

Acceptance: a model artifact can be evaluated reproducibly in a clean environment,
results trace back to exact inputs, and metric changes cannot silently rewrite baselines.

## 6. Additional dataset formats (low priority)

Add formats through existing metadata and reader boundaries after the higher-priority
interfaces stabilize. Require format-independent class ordering, split handling,
multilabel behavior, lazy loading, and useful errors. Keep format dependencies optional.

Acceptance: tiny fixtures exercise each new format through training and prediction
without changing existing format detection or defaults.
