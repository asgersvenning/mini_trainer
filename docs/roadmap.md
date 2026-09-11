# Repository strengthening roadmap

The active quantization feature branch has a separate
[execution roadmap](quantization-roadmap.md) covering measured bottlenecks,
target-machine dependencies, integration gates and deferred work.

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

Delivered: opt-in native CUDA INT8 Linear training, x86 PTQ/QAT inference,
checkpoint/export integration, bounded preparation/normalization storage, and
allocation-aware loader hardening. Defaults remain floating point. EMA is deferred;
native QT DDP/FSDP is unsupported.

The [quantization roadmap](quantization-roadmap.md) owns remaining performance,
quality and deployment work. [Measured findings](benchmarks.md) distinguish local
memory savings from mixed speed/convergence results; HPC, desktop/Spark and ARM
qualification remain open. AMP, fake quantization and CPU smoke tests do not
complete the deeper quantization objective.

After quantization, evaluate a task-aware augmentation recipe against the legacy
pipeline. Preserve label semantics, input dtype, optimizer/accumulation/resume
behavior and existing builder extension points. Add loading controls only when
whole-training measurements demonstrate a benefit.

Acceptance: opt-in supported recipes, reproducible paired quality/resource evidence,
and unchanged default behavior. Use [training feature validation](training-feature-validation.md)
for the deferred optimizer, loss and augmentation comparisons.

### Optional dataset preparation for scalable loading

Target: remove repeated small-file access bottlenecks on shared storage through an
optional preprocessing command that prepares an indexed, sharded dataset for the
existing training and prediction loaders. This is an efficiency representation of
already-supported inputs, not a requirement to migrate source datasets or adopt a
new training API. Keep the public design independent of a particular provider.

- Reuse the current source/metadata adapters. Cover every currently supported
  input, flat and hierarchical labels, multilabel targets, supplied splits,
  class ordering and sample identity. Retain lazy loading, inference and supported
  single-process/DDP behavior. Define a compatibility matrix before implementation;
  do not silently drop cases that are inconvenient for a shard backend.
- Preserve original encoded image bytes by default and apply the existing decoder,
  resize, transforms and hooks at loading time. Do not bake stochastic augmentation
  into prepared data. Any later materialized-preprocessing option must be explicit,
  versioned and checked against the requested runtime preprocessing.
- Store a versioned manifest with source provenance, sample-to-shard index, labels,
  splits, class mappings and integrity information. Support bounded, resumable
  preparation with atomic publication of completed artifacts and explicit errors
  for missing, changed or corrupt samples; never silently skip them.
- Evaluate indexed uncompressed TAR shards before inventing a container format.
  Preserve current sampler order and epoch coverage in the first implementation.
  Treat locality-aware or streaming shuffles as separate opt-in behavior changes,
  with explicit DDP partitioning, equal-step and checkpoint/resume contracts.
- Allow direct reads of prepared shards and optional staging to verified node-local
  storage. Share a byte-bounded cache across ranks/workers on each node, coordinate
  downloads, protect in-use shards and publish verified cache entries atomically.
  Measure cache churn under the actual sampler; packing files alone does not
  guarantee efficient random access or eliminate cold-read latency.
- Keep implementation within the existing metadata/reader boundaries, using a
  focused preparation/storage module where needed. Expose it through the normal
  CLI and Python configuration paths, keep new dependencies optional, and preserve
  the original uncached path and defaults.

Deliver in bounded increments: compatibility fixtures and a preparation/loader
round trip; indexed shards; optional shared local staging/cache. A separate small
loader improvement may add bounded concurrent encoded-byte reads within each
batch, preserving sample order and hook execution without multiplying process-local
metadata. Benchmark that independently of changing storage representation.

Acceptance: byte-identical decoded inputs for the default representation, identical
labels/splits/class ordering and sampler coverage, supported training/prediction and
DDP restoration checks, corruption/interrupted-preparation recovery, and bounded
memory/disk use. Compare first-pass and repeated-pass end-to-end throughput on a
representative working set, including conversion/staging time, validation and
figures. Report the number of epochs needed to amortize preparation; a warmed tiny
subset is insufficient evidence. Implementation and GPU qualification remain open.

## 5. mini_metrics and continuous model evaluation

Planned target: economical continuous benchmarks with a GitHub audit dashboard.
Connect hosted CPU checks and small, on-demand GPU runs to durable report history
through an independent publisher. Use fixed representative subsets of existing
datasets, preserve supplied splits, and compare baseline/candidate under matching
hardware and workload conditions. Large distributed qualification and production
training remain separately triggered activities, outside the continuous schedule.

Keep allocation credentials in a trusted controller isolated from candidate code,
pull requests and report publishing. Restrict profiles, revisions, dataset access,
concurrency and allocation duration; enforce a persistent spending budget, reconcile
ambiguous submissions before retrying, and disable automatic time extension.
Provider/account configuration and infrastructure review details remain local.

Acceptance: dry-run and mocked lifecycle tests cover interruption, duplicate
submission, budget exhaustion and credential isolation before a capped live trial.
Then demonstrate cancellation/cleanup and auditable success, failure and publishing
retry records before enabling a schedule. Missing runs must remain visible; partial
GPU allocations must not imply full-device or distributed performance qualification.
Build on the existing [reporting integration](../dev/benchmarks/reporting.md).

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

Deferred: consolidate flat and hierarchical inference behind one public CLI.
`mt_hpredict` already shares the generic prediction CLI and has an explicit flat
head route; use that existing functionality while training qualification proceeds.
Before refactoring, assess whether stored head/taxonomy metadata can reliably
select the builder and result collector, or whether weights need additional
versioned configuration. Preserve explicit CLI overrides, legacy checkpoints,
existing entry points, class ordering and prediction/mini_metrics output contracts.
Acceptance: installed CLI tests cover flat and hierarchical weights, including
older artifacts with missing metadata and actionable handling of ambiguous cases.
This is not a prerequisite for DDP qualification or the production training run.

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
