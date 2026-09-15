# Production training workflow: post-mortem and next-run plan

Status: proposed development plan; no implementation is authorized by this document.
Campaign: 10–11 September 2026. The quant branch has since been merged into master.

## Executive assessment

The campaign produced a useful model, completed four-GPU training with figures and
W&B, evaluated in-domain and independent expert data, exported floating and PTQ
ONNX artifacts, and packaged the evidence with verified checksums. The original
objective of faster, memory-efficient fully quantized training was not established.
These are separate outcomes: merging useful improvements does not qualify every
experimental backend or prove absence of regressions against a full master run.

The largest opportunity for the next run is operational reliability and data access,
not another optimization matrix. Small warmed trials selected a viable compute
configuration but substantially underestimated first-pass storage costs. Repeated
manual handoffs, opaque status and late preparation of evaluation/export consumed
scarce allocation time. Preserve the flexible CLI/API design and human decisions;
make the routine steps repeatable, inspectable and recoverable.

This assessment uses the operator-provided logs, metric tables and command results
from the campaign, checked against the current repository interfaces. It is not an
independent audit of the downloaded archive or a replay of the experiments. The
archive checksum was reported as passing. Its final public URL and immutable model
hashes must be attached when available; do not invent them from version names.
Older notes that say production was incomplete, or retain four loader workers as
the production setting, are superseded by this campaign evidence.

## Outcomes and strength of evidence

| Area | Observed result | What it establishes / does not establish |
| --- | --- | --- |
| Production | EfficientNetV2-S, normalized hierarchical head, size 384; 4 full B200 GPUs; batch 256/rank; FP16 AMP; model compilation; 32 loader workers/rank in the successful run | A functional recipe on this allocation, not an optimum across precisions, machines or models |
| Optional features | Optimizer compilation, explicit CUDA prefetch, INT8 training and EMA disabled | Deliberate scope reduction; EMA and native INT8 DDP remain unsupported |
| Training | 30 epochs, 17:34:23 total; training 16:27:22, evaluation timer 43:33; best model reported at epoch 30 | Finished within allocation; phase timers do not cover every logging/teardown cost |
| Final metrics | Train species micro accuracy 98.1645%; validation species/genus/family 94.2353% / 97.6693% / 99.5047%; finite recorded losses | Strong completed-run result; not a controlled comparison with the old master model |
| In-domain test | 632,913 images; baseline micro accuracy 94.21% / 97.64% / 99.50%; macro accuracy 92.94% / 97.00% / 98.48% | Test agrees closely with validation; no dataset-leakage audit was completed here |
| Expert test | 58,640 images, 522 species; baseline species micro accuracy 57.59% all labels, 66.74% known only | Substantial domain/vocabulary shift remains despite strong in-domain results |
| Figures/W&B | Full-head diagnostics and distributed logging completed throughout production | Required product functionality worked; cost and summary semantics still need improvement |
| Resume probe | All 4 ranks reported identical checkpoint hash, start epoch 3 and restoration of model/optimizer/scheduler/scaler | Controlled restoration worked; arbitrary interrupted stochastic continuation is not proven bit-exact |
| FP32 ONNX | Dynamic batches 1, 2, 4 passed at rtol=1e-4, atol=1e-4; largest observed absolute error 3.84e-5 | Synthetic-input numerical parity, not real-image end-to-end parity or target-GPU performance |
| ONNX PTQ | 128 training images; Percentile 99.9; Q/DQ graph validation and finite runtime smoke passed | Quantized ONNX artifact exists; retained quality and native integer execution remain unqualified; no quantized .pt was created |
| Retention | 23.71 GiB before ZIP; ZIP integrity check and subsequent archive checksum passed | Evidence was preserved; size, discoverability and clean-environment reuse need refinement |

### Allocation timeline and where time was exposed

The four-GPU allocation began around 17:24. The successful production training
started around 19:01 (inferred from the final duration), about 97 minutes later.
Training finished around 12:35 the following day; test inference was reported
complete at 16:11, evaluation output around 16:44, and archive verification at
17:13, shortly before the 17:24 expiry. These are operational timestamps, not a
profile attributing every minute to a specific component. Qualification had value;
not all of this interval was avoidable waste. Nevertheless, hours of post-training
storage/evaluation and last-minute packaging were clearly on the critical path.

The reported source split contained 5,063,857 training images, 633,224 validation
images and 632,913 test images. Thirty training epochs divided by the reported
training timer imply about 2,564 images/s aggregate, or 641/rank, consistent with
the observed roughly 650 training images/s/GPU. Use the training split rather than
all six-million-plus source rows for runtime planning, and budget validation and
diagnostics separately. This cross-check also illustrates why units belong in the
report rather than being reconstructed from ETA afterward.

### Compute selection: useful, but deliberately bounded

Reported aggregate warmed training throughput from four-GPU qualification:

| Batch per GPU | Images/s | Peak allocated bytes per reported maximum | Peak reserved fraction |
| --- | ---: | ---: | ---: |
| 32 | 1,059.648 | 8,332,350,464 | 0.059 |
| 64 | 1,741.601 | 15,640,267,776 | 0.098 |
| 128 | 2,272.305 | 30,251,580,416 | 0.178 |
| 256 | 2,517.928 | 59,508,456,960 | 0.324 |

Increasing batch 128 to 256 gained about 10.8% throughput with nearly twice the
allocated memory. Larger batches were not exhaustively tested. Spare memory is an
opportunity, not evidence that a further increase pays off. Use equal-memory-budget
and time-to-quality comparisons where appropriate, not just equal batch size.
Changing global batch also changes optimizer updates per epoch and schedule behavior.
The qualitative impression of improved prototypes/generalization at larger batches
is useful feedback, not a controlled convergence result.

Earlier single-GPU trials repeatedly found model compilation reduced later-epoch
training time by roughly one third and allocated memory by roughly one third.
Optimizer compilation alone or combined with model compilation did not improve
those steady-state timings and added substantial cold startup. Explicit prefetch
showed no convincing additional gain. INT8 combined runs timed out and were not
qualified. Keep these negative results so that the next run does not repeat the
same matrix without a new hypothesis, implementation change or target requirement.
FP16 is the tested baseline; BF16 and other compiler modes were not eliminated by
a comprehensive comparison.

## Failure modes and lessons

### 1. Storage behavior invalidated small-subset extrapolation

Full-data training and test inference initially stalled while data workers had low
CPU use and waited in `D` state / `folio_wait_bit_common`. A 32-step storage window
spent about 146–150 of 211 seconds waiting for the loader across ranks. Test
inference reached only 69/2,473 batches after 43 minutes. GPU utilization snapshots
sometimes looked high despite poor progress; they did not identify the bottleneck.

The mounted filesystem was WEKA. Repeated reads of the same expert subset went
from hundreds of seconds to below one second. The operator observed improved cold
staging with 512 readers, and a later disjoint-sample read sweep favored 512 readers
(~117 images/s confirmation median) over lower concurrency. Another workload with
larger files provisionally favored 64. These are workload/cache-state results, not
universal worker defaults. Strong evidence supports cache-state-dependent storage
latency; the exact client/server cache or network mechanism was not instrumented.

The successful production run used 32 loader workers/rank and eventually stabilized
near 650 training images/s/GPU and 2,000–2,500 evaluation images/s/GPU. The long
warmup within the run supports a cache-related explanation; it does not quantify a
separate causal percentage for each storage/concurrency effect. Practical mitigation
need not wait for a filesystem-internal diagnosis.

Lessons:

- Separate process workers for decode/augmentation from concurrent outstanding
  encoded-byte reads. More processes are not the only way to hide blocking I/O.
- Measure first-pass and repeated-pass behavior. Disjoint paths avoid direct sample
  reuse but cannot guarantee cold shared caches. Never drop global caches on a
  shared allocation to manufacture a benchmark.
- Start with bounded but genuinely high concurrency candidates when latency is
  evidenced; do not spend the allocation creeping from 4 to 8 to 16 threads.
- Separate reader startup, first result, steady work and drain/cancellation time.
  Time-limited trials can be informative without completing their target file count.
- Report images/s and bytes/s, file-size distribution, errors, actual concurrency
  and sample coverage. A sampled optimum is a provisional operating point.
- Prefer optional staging/encoded-byte preparation. Decoded caching of millions of
  images is not a safe default even on a high-RAM node.

### 2. Orchestration was too easy to interrupt and too hard to resume

Examples included relative commands launched from the wrong directory, missing
inference YAML files, torchrun interpreting `--run` as its own abbreviated option,
preparation tied to historical commits, and cancelled stages refusing an existing
directory. A trial had final weights and finite metrics but was marked timed out
when the shared wall budget expired. This is neither evidence of failed training
nor permission to relabel the whole process successful: export, logging, teardown
or restore checks may still be incomplete.

The package pin and checkout revision sometimes differed intentionally, while
`PYTHONPATH` overrides added a third possible source identity. Hand-built commands
made that hard to see. The late archive command also lost its terminal connection;
file descriptors could not recover previously lost stdout. Durable files and a
completion checksum, rather than terminal appearance, ultimately established success.

The necessary response is a small extension to existing harness state and commands,
not a general workflow engine. Record phase completion separately from process exit,
retain partial results, and make retry/resume/archive-incomplete actions explicit.
A checkpoint's existence is not a replacement for a loss audit or clean shutdown.

### 3. Qualification tested the right features, but not early enough in combination

Figures were initially disabled to avoid expensive rendering, although they were a
required training diagnostic. At larger taxonomy size the dendrogram caused long
validation pauses and recursion warnings. Confusion/dendrogram improvements allowed
production diagnostics to remain on: at 12,632 species, confusion generation was
about 8–13 seconds and warmed dendrogram rendering about 12–13 seconds. First label
resolution added roughly a minute in one run; subsequent resolution was much faster.

Whole-matrix confusion patterns and prototype organization matter to the operator.
Do not replace them with a few selected classes or disable them as the standard
performance fix. Preserve visual interpretation while bounding file/display costs.
Which hierarchy levels render must be an explicit option recorded in the resolved
configuration; the unexpected level-0-only output showed that defaults were opaque.

The production smoke should exercise the actual CLI, full head dimensions, figures,
W&B, validation and save/reload on the intended topology. A small subset tests those
contracts; a separate bounded storage probe tests uncached access. Neither replaces
the other. Four GPUs passing does not establish eight-GPU behavior.

### 4. Numerical and compiler warnings need a triage budget

Early validation loss NaNs appeared in both branch baselines and optional-feature
runs; later epochs could be finite and the final production loss audit passed.
This weakens attribution to prefetch or quantization but does not prove the NaNs
were harmless. Keep a cheap first-occurrence record: phase/batch/sample identities,
input/output/loss finiteness, dtype and enabled features. Capture a bounded replay
only when triggered; avoid a permanent expensive debug path.

Observed compiler issues were stochastic-depth specialization/recompile limits,
hierarchy initialization using `Tensor.item()`, and DDP gradient/bucket stride
mismatches. They are performance targets, not demonstrated model-quality failures.
Prioritize a change only after a representative trace shows repeated fallback,
recompilation or copy cost. Do not globally raise limits, suppress warnings or change
numerics merely to make the log clean. Keep EMA repair outside the next-run critical
path unless the operator explicitly needs EMA.

### 5. Inference ingestion and evaluation preparation lagged behind training

Folder inference unnecessarily depended on training-oriented index construction
and model-vocabulary membership. Valid expert species could be absent from the
model. A taxonomy rank-count/index mismatch was initially plausibly attributed to
network resolution. The bounded discovery fix was useful; broader ingestion should
keep discovery, taxonomy resolution, split policy and model indexing separate.

A model-vocabulary filter restricts candidate predictions, not the validity of
ground-truth labels. Future source formats must reuse this separation for both
training and inference. External taxonomy access needs cache/provenance and clear
transport-versus-local-mapping errors; a hardcoded socket timeout is not a batch
or job deadline.

Inference should normally require input, output and weights. Derive shape,
preprocessing and head/collector behavior from saved metadata, with explicit
operator overrides and actionable errors for ambiguous legacy metadata. Generate
resolved config for inspection; do not fill YAML with guessed defaults.

Staging eventually made expert inference finish in 2:49 and full test inference in
30:02, after slow source reads had dominated. Prepare the evaluation plan before
training and overlap independent preparation where resource budgets allow. Do not
launch competing cold scans blindly. Pin mini_metrics in its own compatible Python
environment and preserve its pretty tables plus machine-readable outputs.

### 6. Evaluation success and deployment readiness are different gates

The expert set's 16 unseen species accounted for 8,042 images (13.71%) and errors;
five of those species contributed 84.3% of unseen-species errors. Error concentration
also existed among known species. This supports the operator's interpretation of
uneven domain/vocabulary effects. Geographic candidate lists are promising opt-in
priors, but their provenance/scope and excluded true labels must remain visible.
No regional-filter accuracy gain was established in the supplied results.

Retain all-label/known-only and unfiltered/threshold-optimized reports. Clarify
micro versus macro, rank, aggregation period and abstention denominator. In-domain
optimized species accuracy was 95.73% at 97.46% coverage; expert known-only optimized
species accuracy was 92.25% at 53.66% coverage. These are not equivalent operating
points. Threshold search/split semantics and the distinction between an applied
threshold and a subsequently reported optimum must be documented before rollout.
Choose deployment thresholds using suitable calibration data and evaluate them
frozen; current optimized test reports remain exploratory. Empty-abstention summary
NaNs are distinct from nonfinite model outputs or loss.

FP16 AMP training did not imply an FP16 ONNX export. FP32 export required an explicit
absolute-tolerance adjustment after small near-zero discrepancies; tolerances and
errors were retained. Do not silently weaken defaults. Add real-image parity and
rank/top-k changes alongside absolute errors, with predeclared acceptable limits.
PTQ happened directly on ONNX, so no quantized PyTorch artifact exists. Q/DQ node
counts and runtime loadability are not proof of integer kernel placement, target
speed or retained accuracy. These must be separate candidate acceptance checks.

### 7. Packaging worked, but happened too late and bundled too much together

The verified archive preserved a large set of results, logs and figures. Packaging
was improvised near expiry, and the chosen public bundle also contained a resume
checkpoint and extensive history. Use one relocatable directory but distinguish a
small deployment subset, reproducibility evidence and optional training archive.
Never omit ONNX external tensors. Retain omissions in the manifest; calibration
manifests referring to omitted tensors must not imply self-contained replay.

A publication command should inventory intended files, sizes and public/private
scope before copying; omit credentials, caches and raw training images by default.
Record package revision, harness revision, weights hash, splits, preprocessing,
class mapping, evaluation versions, export tolerances and target qualification.
Write logs to disk from process start and publish completion atomically. Checksums
establish integrity, not accuracy, data provenance or deployment correctness.

## Prioritized development plan

Effort below is a planning estimate for implementation plus focused validation,
not a promise: S = roughly 1–2 developer days; M = several days; L = one or more
weeks with integration work. Re-estimate after inspecting the existing boundary.
Prefer the smallest vertical slice that removes an observed failure.

| Priority | Increment | Value / effort | Done when |
| --- | --- | --- | --- |
| P0 | Durable stage state and resolved execution plan | High reliability, S–M | A fresh job can inspect, run, stop, retry and resume each existing stage; logs survive disconnect; completed stages are reused only after input/hash validation |
| P0 | Prepare evaluation/export/publication before allocation | High saved allocation time, S–M | One tiny installed-CLI fixture runs train → predict → metrics → export → package; commands, required paths and dependencies are checked before expensive work |
| P0 | Separate compute qualification from storage qualification | High decision value, S | A compact report distinguishes startup, warm training, first-pass I/O, validation/figures and teardown; it states scope and units and supplies a bounded next action |
| P1 | Reusable concurrent read/staging calibration | High on affected storage, M | Startup and work budgets are separate; incomplete trials remain informative; bounded cancellation/recovery, disjoint sampling, file sizes and actual concurrency are reported; operator override survives |
| P1 | Optional prepared encoded-byte dataset round trip | Potentially high recurring gain, L | Existing input compatibility matrix and sampler/DDP contracts pass; end-to-end savings exceed preparation/maintenance cost on representative data |
| P1 | Real-image export and candidate evaluation | High rollout confidence, M | Same preprocessed held-out inputs compare PyTorch, floating ONNX and optional PTQ; score/rank changes, quality, coverage, placement and target resource evidence are retained |
| P2 | Figure/metadata caching and explicit figure policy | Medium operational gain, S–M | Whole-matrix diagnostics and requested hierarchy levels remain inspectable; first/warm rendering and file sizes are recorded without repeated taxonomy resolution |
| P2 | Targeted compiler/numerical fixes | Conditional gain, M per demonstrated issue | A bounded reproducer demonstrates the cost/failure and a paired check verifies the fix without changing checkpoint or prediction contracts |
| Deferred | Native INT8 DDP, EMA, expanded precision/compiler matrices, automated allocation, universal inference CLI | Uncertain value, M–L | A concrete requirement or measured bottleneck justifies a separately reviewed experiment; none blocks the baseline run |

### A. Next-run minimum: complete the P0 vertical slice first

Build on `dev/ucloud/{setup.sh,compare.py,scaling.py,production.py}`, the evaluation
scripts and existing ONNX tools. Do not replace the public training/inference APIs.
A small shared run manifest should reference existing stage outputs rather than
copy their schemas into another database. Record stage input identities, resolved
command/config, start/end, output/log paths, exit cause and completion checks.
Differentiate preparation, compilation, training, validation/figures, checkpoint,
logger synchronization and teardown. Start with phases that are already observable.

Use absolute paths in generated commands, explicit interpreter selection, package
and harness identities, and inspected CLI-over-YAML precedence. Keep W&B's existing
interactive authentication in the foreground. No secrets in manifests. Pin an
accepted master revision; do not embed a permanent dependency on the old quant
branch name or four-GPU topology in otherwise reusable planning helpers.

Define an allocation deadline separately from stage limits, compilation allowance
and cleanup reserve. Human pauses consume allocation time but must not masquerade
as model regressions. Estimate feasible epochs from measured full-data throughput,
actual training split count, validation cost and reserve; do not silently alter the
learning-rate schedule to fit. At a supported checkpoint boundary, give the operator
an explicit continuation/stop decision. Do not promise exact arbitrary-batch resume
without the necessary RNG/sampler contract.

Acceptance should include interruption after preparation, failed child exit, timeout
after checkpoint save, logger teardown failure and re-running a completed stage.
Use tiny fixtures and existing integration tests, not a second production matrix.
A single topology smoke on a changed target remains necessary.

### B. Storage work: validate the cheap mitigation before building shards

First improve the existing calibration/staging helper and, if needed, add bounded
encoded-byte read-ahead inside the current reader boundary. Bound requests and
bytes, preserve sample order, propagate failures with identities, and avoid repeating
large Python metadata per reader. Do not conflate read concurrency, DataLoader
processes and CUDA prefetch. Keep direct filesystem loading available.

Only then implement the optional prepared-data backend already specified in the
[main roadmap](roadmap.md#optional-dataset-preparation-for-scalable-loading).
Start with byte-preserving indexed shards and explicit manifests. Defer distributed
cache eviction services and alternative shuffle policies until measured need.
Prepared formats must support all existing cases or explicitly remain an opt-in
partial prototype; they must never silently redefine splits, labels or sample order.

Decision rule: report total preparation plus expected training/evaluation time and
break-even reuse count. Prefer a modest throughput gain delivered simply over a
complex backend that saves less than its preparation cost for the intended run.
Higher-concurrency staging may be enough for an immediate run; repeated campaigns
on the same corpus strengthen the case for persistent preparation.

### C. Keep human decisions first-class

The proposed workflow exposes `plan`, per-stage execution, compact `status`, and
explicit retry/resume actions; exact command spelling is an implementation choice.
Each stage can still be run through the normal CLI. The human can inspect figures,
change a proposed batch/worker count, select a candidate, pause or reject promotion.
Changes produce a new resolved configuration and identity; completed unrelated
stages are retained rather than overwritten or automatically rerun.

Operator decisions are required at scientific boundaries: dataset/split choice,
global batch and schedule, geographic priors, thresholds, acceptable PTQ quality
loss and production promotion. Routine bounded retries and report generation do
not require repeated permission prompts. Do not turn every warning into a blocker.
Expose reason, scope and recovery command when intervention really is required.

The next-run sequence is: prepare the plan and tiny end-to-end fixture before
allocation; inspect the actual node/storage; perform one production-feature smoke
and bounded storage calibration; choose a batch using existing evidence plus at
most a few useful candidates; verify save/reload; launch production in the same
allocation; evaluate and export at completion; finalize durable artifacts. Preserve
manual allocation and permit an explicit decision to skip optional experiments.

## Experiment and validation budget

- Every experiment states the decision it changes, baseline, measurement scope,
  maximum elapsed/allocated GPU time, and stop criterion before launch.
- Reuse the qualified floating recipe unless hardware, code or workload changes
  invalidate it. Run a full old-master training comparison only as a separately
  funded scientific question, not a prerequisite for ordinary deployment.
- For the first batch/worker search, set a small candidate budget. Stop when gains
  are within observed noise or too small to affect run completion. Expand only with
  evidence and an operator decision; do not exhaust a Cartesian product.
- A single null measurement is not proof of equivalence. Repeat only close decisions
  or unexpected failures that matter; multiple seeds are for quality claims, not
  every operational smoke. Keep sample-cache caveats in the report.
- Preserve figures and W&B in qualification. Report end-to-end time as well as
  compute-only timing so optimizations cannot hide costs in validation or shutdown.
- Default PR checks should exercise contracts with tiny fixtures. Report test
  durations, consolidate duplicated behavior checks, and reserve real-backbone,
  GPU, full-scale storage and performance experiments for the relevant changes.
  Do not weaken numerical assertions simply to get green CI.
- Continuous benchmarks should use small representative jobs, not large nodes.
  Allocation credentials remain separate from candidate code and publishing;
  automatic cloud provisioning is not part of this next-run plan.

## Success criteria and deliberately open questions

The next campaign should require no hand-edited browser YAML, no reconstruction of
missing commands from chat, and no ambiguity about which stage finished. It should
produce a usable selected-model bundle even if optional PTQ or an upload fails.
Measure operator interventions, allocation time before first productive training,
first-pass/warm throughput, evaluation turnaround, failed/repeated stages and final
artifact size. Set concrete budgets from the next dataset/node plan; the current
record does not support a universal target percentage or preparation-time guarantee.

Still open: unbiased convergence comparisons across batch sizes/precisions; exact
resume semantics under stochastic training; real-image exported-model parity;
PTQ quality and target performance; deployment threshold policy; regional prior
provenance; whether storage preparation amortizes for the next corpus; and the
published immutable artifact URL. None should be reported as solved by this plan.

The existing [quantization roadmap](quantization-roadmap.md) remains the specialist
backlog for quantized performance and deployment work. This document supplies the
operational order for the next production campaign, not a competing feature matrix.
