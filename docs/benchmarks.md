# Continuous dataset benchmarks

[Browse benchmark runs and their summaries](https://github.com/asgersvenning/mini_trainer/actions/workflows/benchmarks.yml).
The workflow is defined in this repository; results start appearing there once the
change is merged and runs execute. Initial local evidence is recorded below.

The benchmark foundation exercises the actual training, checkpoint reload and
inference paths. CPU is a fast verification milestone, not the scope boundary.
The progression is synthetic oracle → MNIST → hierarchical Blair, with Birds and
iNaturalist 2021 reserved for larger, more expensive follow-ups.

Each run produces a visible Actions summary and a downloadable artifact containing
JSON reports, predictions, dataset/split manifests, configurations, checkpoints and
logs. Failures are retained alongside successful results. The artifact excludes
synthetic image files, which can be regenerated from the recorded seed; source
images for real datasets are never uploaded by the workflow.

Reports identify the source revision and source-content hash, lockfile hash,
installed versions, dataset/checkpoint hashes, seeds, class mappings, device,
precision, cache settings and explicitly exercised capabilities. Results marked
`completed` only establish that the real-data pipeline ran: they do not claim a
quality improvement. Synthetic profiles have an exact 100% oracle quality gate.

Artifacts are configured for 90-day retention. This is an initial reporting surface,
not a permanent model zoo. A durable results index/dashboard should archive these
versioned records before expiry, retain failures, and compare matching configurations
across revisions. It must separate quality from speed and avoid combining CPU/GPU,
precision, dataset, or dependency changes into a misleading trend.

## Initial local evidence

These are baseline observations, not proposed model-quality or speed improvements.
They use seed 42, one CPU thread, zero loader workers, no augmentation/EMA, and small
offline models. The synthetic runs use 12 epochs; real-data runs use 5 epochs.

| Profile | Held-out accuracy | Scope |
| --- | --- | --- |
| Synthetic CPU float32 | 100% | Exact repeatability verified within the local environment |
| Synthetic CUDA float32 | 100% | Strict deterministic algorithms; CUDA cache |
| Synthetic CUDA float16 | 100% | AMP and CUDA cache |
| Synthetic CUDA bfloat16 | 100% | AMP and CUDA cache |
| MNIST CUDA float16 | 97.12% | AMP, CUDA cache, explicit nondeterministic profile |
| MNIST CPU float32 | 97.3% | 4,000 train / 1,000 validation / 5,000 test |
| Blair CUDA float16 | 64.3% species / 80.4% parent | 3,704 train / 912 validation / 1,161 test |

GPU observations use an NVIDIA GeForce RTX 3080 Ti Laptop GPU. The environment
uses Python 3.13.7 and PyTorch 2.12.0. Single-run timing is diagnostic; it is not a
portable performance gate. CPU synthetic training-call wall time was about 2.2 s,
MNIST CPU about 11 s, and Blair GPU about 13.4 s. These scopes include setup,
training, validation, logging and checkpoints, but exclude dataset inventory and
final held-out inference. Runs were not a controlled throughput comparison.

Strict deterministic mode rejected CUDA adaptive-pooling backward in the real-data
backbone. The explicit `--allow-nondeterministic` profile permits those operations
and records that choice. Seeds and preserved inputs make the experiment repeatable;
they do not promise bitwise-identical GPU results. Neither AMP nor CUDA caching is
inference quantization or quantization-aware training.

## Coverage still needed

The baseline reports mark EMA, distributed training, augmentation, quantization
and ONNX integration as unexercised. The explicit QT profiles below establish
limited quantization coverage; the other capabilities remain unexercised by this suite. Existing focused tests
provide other evidence, but do not make those boxes true for these dataset runs.
Add explicit profiles and comparison criteria before claiming coverage or improvement.
The known EMA continuation failure remains recorded in the roadmap.

The plain AdamW/SGD step-tracking blocker discovered by this baseline is now fixed.
The trainer gates scheduler/EMA advancement on completed optimizer steps and the
native fused-optimizer AMP overflow signal. Real CPU and CUDA overflow regressions
cover MuonAuxAdamW, AdamW, SGD and fused AdamW/SGD. This establishes compatibility;
it does not yet measure their relative model quality. See the [step contract](../dev/README.md#optimizer-step-contract).

See the [benchmark guide](../dev/benchmarks/README.md) for local commands, GPU runner
configuration, real-data inputs and reproduction details.

## Integrated INT8 training

The shared runner now offers paired `qt` (synthetic) and `qt-real` (MNIST/Blair)
profiles. Each pair uses the same architecture, dataset manifest, seed, optimizer,
AMP setting and CPU cache with synchronous construction. QT coverage is checked
after checkpoint reload and reported by operation, alongside parameter storage,
peak CUDA allocation and training-call wall time. Runs are headless; the wrapper
also records process failures that cannot produce a Python exception report.

Local observations on the RTX 3080 Ti Laptop GPU, Python 3.13.7, PyTorch
2.12.0+cu130 and TorchAO 0.17.0 use seed 42, batch size 32, one CPU thread,
zero loader/cache workers and FP16 AMP with float32 optimizer parameters.
Synthetic uses 12 epochs; MNIST and Blair use 5. No augmentation or EMA is used.

| Dataset / path | Held-out accuracy | Parameter bytes | Legacy CUDA reading MiB | Training wall seconds |
| --- | --- | ---: | ---: | ---: |
| Synthetic float | 100% | 88 | 64.04 | 4.22 |
| Synthetic INT8, repeat | 100% | 68 | 32.03 | 12.95 |
| MNIST float | 97.12% | 44,968 | 64.18 | 6.06 |
| MNIST INT8 | 97.00% | 29,648 | 32.17 | 43.08 |
| Blair float | 66.41% species / 80.28% parent | 158,792 | 64.50 | 10.64 |
| Blair INT8 | 64.86% species / 79.59% parent | 60,744 | 64.40 | 26.51 |

Synthetic INT8 repeated with bitwise-identical held-out scores. Its first run
required 80.43 seconds, showing how compiler/autotuning cache state affects these
short runs even without whole-model compilation. Wall time includes setup,
training, validation, logging, checkpoints and first-use compilation; it excludes
final held-out inference. These are single-seed smoke comparisons, not isolated
causal estimates or steady-state throughput measurements. Real-data runs permit
nondeterministic CUDA pooling. QT stochastic rounding can also change the RNG
sequence used by dropout.

MNIST quantizes only its final Linear. Blair uses a 64-unit hidden layer in both
paths and quantizes that layer; convolutions and normalized hierarchical heads
remain floating point. Parameter bytes exclude buffers, gradients, optimizer
state and activations. The legacy CUDA readings in these tables were captured after the logger reset
its peak counters. They do **not** establish whole-run peak allocation or its
reduction, and should not be used for that comparison. The corrected benchmark
now preserves the maximum across every phase reset and the final training call,
and marks the timing/memory scope in each report. Older reports are rendered as
`unverified` in summaries. Parameter storage and accuracy measurements are unaffected. None of these dataset runs demonstrates a training speedup.

The initial Blair QT attempt aborted in Tkinter cleanup before reporting; the
headless fix allowed the successful rerun above. Reports now preserve skipped
operation reasons across reload. These local artifacts remain outside the checkout;
the shared workflow retains future reports, predictions and logs in Actions.
See [the reproduction commands](../dev/benchmarks/README.md#integrated-qt-dataset-profiles).


With row-wise weight normalization supported, a further matched Blair pair uses
no hidden layer and quantizes the normalized classifier direction directly:

| Blair path, hidden size 0 | Held-out accuracy | Parameter bytes | Legacy CUDA reading MiB | Training wall seconds |
| --- | --- | ---: | ---: | ---: |
| Float | 64.25% species / 80.45% parent | 75,848 | 64.34 | 11.65 |
| INT8 normalized direction | 62.62% species / 76.14% parent | 37,548 | 32.25 | 25.32 |

The environment, seed, five-epoch budget, batch size and CPU cache settings match
the preceding comparisons. Dataset manifest hashes agree between the two runs.
Convolutions still remain floating point. This establishes real hierarchical
training and restored-checkpoint inference with normalized integer weights, with
roughly half the parameter storage. Accuracy is lower in this single run and QT
is slower; neither convergence parity nor a throughput improvement is established.

## Dense MNIST profile with corrected peak measurements

The dense spatial MLP profile quantizes all four Linear weights, including the
backbone. Both paths use 15 epochs, batch size 128, seed 42, SGD momentum 0.9,
head/backbone LR 0.3/0.1, zero weight decay, FP16 AMP, model compilation and a CPU
cache. The 4,000/1,000/5,000 train/validation/test split is unchanged. Optimizer
updates are eager. This is a compute-heavy classification profile, not a proposed
MNIST model-quality baseline or a claim about CNN performance.

| Path | Test accuracy | Parameter bytes | Whole-training peak CUDA MiB | Total training-call seconds | Median train-loop seconds, epochs 2–15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float | 93.16% | 52,944,936 | 236.30 | 12.49 | 0.191 |
| INT8 | 93.36% | 13,291,600 | 373.30 | 99.98 | 0.355 |

A second INT8 run reproduced the accuracy and peak, with total wall time reduced
to 58.77 seconds after compiler caches were populated. Startup costs and cache
conditions prevent interpreting the total wall-time ratio as a steady-state
speed ratio. Batch-loop times include loading, preprocessing, compute and batch
logging; they exclude figures and checkpoint writes. CUDA measurements now
preserve maxima across **every batch/phase/finish reset**, rather than reading
only the counter remaining after training. Early experimental per-phase memory
fields without `phase_peak_memory_scope` are likewise unverified; the corrected
logger tracks each phase across its batch resets.

This is a negative performance result: physical parameter storage falls by about
75%, but total peak memory increases and the timed training loop is slower. The
small accuracy difference is one seed, not evidence of a quality improvement.
The standalone fast kernel probe compiled optimizer updates as well as the
model; this trainer profile compiles only the model. Fusing optimizer updates
while preserving overflow detection, scheduler advancement and resume semantics
is therefore a concrete next investigation, together with temporary-allocation
profiling. Kernel-probe speedups do not establish completion of the QT goal.

Use [`qt-dense`](../dev/benchmarks/README.md#dense-real-data-qt-comparison) to run
this pair and retain the full reports, checkpoints and predictions in the shared
pipeline. The optional GPU Actions job includes it when QT and real-data profiles
are enabled; no self-hosted job was dispatched from this session.

### Fused INT8 storage updates (2026-09-09)

The same dense MNIST profile was exercised with experimental fused CUDA storage updates.
This implementation has not passed the full CUDA regression set and is not a
delivered feature; the results below guide the next implementation attempt.
These are individual runs on the same RTX 3080 Ti, seed 42, data split, SGD
settings and 15-epoch budget used above. All compile the model. The floating
comparison also compiles the optimizer; INT8 is shown both ways to expose the
effect of compiling the outer update wrapper. Later-epoch figures below cover
epochs 3–15, after first-use compilation and optimizer initialization.

| Execution | Test accuracy | Whole-run peak MiB | Later training peak MiB | Median later training epoch s | Training wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float, compiled optimizer | 92.44% | 236.30 | 236.30 | 0.185 | 14.91 |
| INT8 fused storage, eager optimizer | 93.32% | 373.30 | 164.58 | 0.273 | 44.26 |
| INT8 fused storage, compiled optimizer | 93.12% | 423.79 | 164.35 | 0.271 | 47.01 |

Fusing the storage update lowers later-phase allocation and runtime relative to
the earlier INT8 update, but does **not** establish a whole-run memory or speed
win over float. Accuracy differences remain single-seed observations. Wall times
include compilation/autotuning and depend on cache state; they are not controlled
cold-cache comparisons. The corresponding report source hashes begin
`8fce9ea8fe3c`, `4f88ebf5632d`, and `a4e805ebe5c4a` respectively.

A separate two-epoch CUDA allocation trace identified a 268,435,456-byte buffer
allocated by Triton's `get_empty_cache_for_benchmark` during TorchAO INT8 matrix
kernel autotuning. This occurs during first-use training and evaluation, and
explains why steady-state storage savings do not translate to a lower whole-run
peak. Reducing this tuning overhead remains necessary; excluding it from the
reported peak would conceal a real allocation that users must accommodate.

The four-weight dense run emitted no recompilation-limit fallback, but a separate
twelve-group SGD stress test did: outer optimizer compilation specialized on
`TrainingWeight` object identities. Keep that option off for general QT workloads
until the wrapper path is fixed. Seven isolated storage tests passed rounding
bounds, CUDA RNG replay and saved-tensor invalidation checks. However, the combined
CUDA suite finished with 40 passing and three failing tests: fake-tensor execution
reached the real storage kernel during normalized Adam updates, and the storage
compiler exhausted its eight-variant cache across dtype/operation combinations,
also preventing the twelve-parameter reuse regression from running successfully.
Raising that global limit would conceal the underlying dispatch/cache design
problem. The next implementation needs a storage kernel that handles these
variants without depending on per-frame Dynamo specialization.

### Explicit CUDA kernels and local tuning

The replacement uses a row-wise Triton update behind a custom operator with fake
execution support. It avoids the failed prototype's per-frame compilation cache.
INT8 matrix multiplication retains TorchAO's kernel/configurations, but a separate
local tuner measures kernels with CUDA graphs and caches selected configurations
on disk. This avoids the 256 MiB cache-flushing buffer without changing global
TorchAO or Triton behavior.

The dense MNIST pair was rerun with the same dataset, seed 42, 15 epochs, batch
128, model compilation and eager optimizers. Both report source hash
`0175e5a8d8e8998e98e57288478b4adc5fd5645e93f0855ab3ef1b9bef55bd2f`
and the same dataset manifest. These are single runs, not statistical estimates.

| Execution | Test accuracy | Whole-run peak MiB | Median training epoch s, epochs 3–15 | Training wall s |
| --- | ---: | ---: | ---: | ---: |
| Float | 93.16% | 236.30 | 0.237 | 14.13 |
| INT8, explicit kernels | 92.42% | 154.00 | 0.242 | 24.74 |

This demonstrates approximately 35% lower **whole-run** peak allocation for the
INT8 profile, including first-use tuning. It does not establish a training speed
win: later-epoch times are similar and total wall time remains higher. First-use
compilation and cache state affect the wall comparison, and one seed cannot
establish equivalent model quality. The original QT speed objective remains open.

The combined CUDA kernel/model regressions now pass the previously failing
normalized checkpoint and dtype/operation cases. Coverage includes stochastic
rounding bounds, CUDA RNG replay, saved-tensor invalidation, and reuse across
twelve distinct weights with changing learning rates. A forced-cold matrix tuning
test compares against integer reference arithmetic and requires less than 16 MiB
temporary allocation for a tiny product, so a cached tuning result cannot mask a
return of the old 256 MiB allocation.

Outer optimizer compilation still falls back when sufficiently many quantized
groups specialize on wrapper identities. A separate strict expected-failure test
enables hard failure on that fallback; it must be removed when the outer path is
fixed. This limitation is distinct from the now-working compiled storage operator.

### Optimizer FMA dispatch

Tracing the outer optimizer failure identified missing `prims.fma` dispatch.
Dynamo rewrites tensor-learning-rate `add_`/`addcdiv_` updates as an out-of-place
fused multiply-add followed by `copy_`. The missing operation broke the optimizer
loop into per-weight frames, eventually exhausting the compilation cache.
The quantized weight now supplies its represented floating values for this
out-of-place primitive; the following copy performs stochastic requantization.
No floating master weight is retained.

The twelve-group SGD probe now stabilizes at two compiled graphs after warmup.
SGD and AdamW regressions pass with hard failure enabled for compiler-cache
fallback, replacing the prior strict expected failure. Additional checks compare
optimizer state and update error against floating arithmetic, verify that FMA
itself neither mutates weights nor consumes RNG, and confirm that quarter-code
updates retain their expected average while CUDA RNG and rounding masks advance.

Both dense MNIST paths were rerun with model **and optimizer** compilation, the
same seed/split and 15-epoch, batch-128 budget. Both report source hash
`b31568d70b4d48a60e3d03a0c1016a06e7f49d31bbedeb134f4b50489db6ddc7`.

| Execution | Test accuracy | Whole-run peak MiB | Median training epoch s, epochs 3–15 | Training wall s |
| --- | ---: | ---: | ---: | ---: |
| Float | 92.44% | 236.30 | 0.167 | 11.49 |
| INT8 | 87.38% | 182.97 | 0.227 | 25.11 |

Compilation compatibility is fixed, but this is still a negative speed/quality
result. Whole-run allocation remains lower than float, but rises relative to
the explicit eager-optimizer row kernel. The INT8 checkpoint records 464 completed
updates versus 465 for float, reflecting one AMP overflow skip; gating remains
active. The accuracy drop is not explained by the passed single-update checks.
Different stochastic rounding trajectories and accumulation over training require
further investigation. Neither this single seed nor compilation success establishes
model-quality parity or completion of the QT goal. Wall times still include
first-use compilation and depend on cache state.

### Larger batches and direct collation

The dense MNIST profile was also evaluated at batch size 512 for 60 epochs,
using the same data split, seed 42, model, SGD learning rate and model/optimizer
compilation. This is a separate workload, not a replacement for the unfavorable
batch-128 results. Before the loader change, the float/INT8 pair reached
93.06%/92.84% accuracy, with median later training epochs of 0.0809/0.0756 seconds
and whole-run peaks of 254.26/190.86 MiB. Initial wall times were 33.00/48.02 seconds,
including INT8 first-use compilation and tuning.

A warmed-up epoch trace showed the cached loader creating per-sample views of
already-stacked tensors before the repository collator returned those same batch
tensors. Direct collation removes this work. In the separate one-thread cache
probe (4,096 RGB uint8 28×28 images, batch 128, seven trials), cached iteration
increased from 520,193 to 2,219,771 samples/s; the scalar-fetch control measured
281,768 and 293,087 samples/s respectively. These are loader-only measurements.

The integrated pair with direct collation and an unchanged INT8 repeat produced:

| Execution | Test accuracy | Whole-run peak MiB | Median training epoch s, epochs 3–60 | Training wall s |
| --- | ---: | ---: | ---: | ---: |
| Float | 93.06% | 254.26 | 0.0802 | 30.61 |
| INT8 | 93.00% | 184.32 | 0.0711 | 27.13 |
| INT8, unchanged repeat | 92.54% | 184.32 | 0.0684 | 26.02 |

These runs share source hash
`60773b54432fcbe6a9f8f5ee6b9f40186d1a4d2b3d33760136a5187205ef6a13`
and the same dataset manifest. They provide evidence of lower whole-run memory
and faster training for this workload, including total wall time with previously
populated compiler/tuner caches. They do not establish a cold-start advantage or
a speedup for other batch sizes, architectures or hardware.

Float predictions matched the pre-loader-change run exactly. INT8 predictions
varied both across the loader change and between two runs of identical code and
seed; these CUDA profiles explicitly allow nondeterministic execution. All runs
retained the same held-out labels and paths. Independent shuffled-loader tests
require exact batches and identical CPU RNG consumption over multiple epochs.
The accuracy variation means these single-seed observations do not establish
quality parity; multi-seed convergence and broader workload validation remain open.

### Graph-visible matmul experiment (not retained)

An experimental replacement of the opaque INT8 matrix operator with
`torch.library.triton_op` made its launch visible inside compiled graphs. A
profiler regression confirmed that Python custom-operator dispatch disappeared,
and 53 CUDA numerical/model regressions passed. Graph capture initially lost the
upstream kernel's default `GROUP_M` argument; passing it explicitly fixed that
compilation error. These results were insufficient to establish a usable change.

The batch-128, 15-epoch dense MNIST control ran from an isolated copy of commit
`2bcc32f`, preserving the direct loader and all training settings. It reached
92.16% accuracy with a 0.221-second median later training epoch and 12.33-second
training call. The graph-visible experiment initially measured 0.211 seconds per
later epoch, but required 78.44 seconds overall. With the final experiment source
hash `335c9dd82b854dd199f7d838b5595a312d46a6b57580e86dff2835b378e7474b`,
two runs produced 76.18% and 62.82% accuracy; their training-call times were
75.39 and 11.32 seconds, and later-epoch medians 0.209 and 0.166 seconds. All
retained the same 182.97 MiB whole-run allocation peak.

The cached speed result cannot justify the quality degradation, and the numerical
tests did not identify its cause. The graph-visible path was therefore removed;
the opaque operator remains in use. This is an unresolved experimental result,
not evidence that the operator API itself is incorrect.

One necessary safeguard is retained: the training tensor's compiler fingerprint
now includes the matrix and update kernel source files as well as the backend
file. The previous fingerprint could allow an old opaque graph to hide a changed
operator implementation during validation. First-use compilation must be measured
again after any of these source files change.

### Continuous multi-seed large-batch profile

The shared [`qt-large-batch` profile](../dev/benchmarks/README.md#multi-seed-large-batch-comparison)
adds seeds 42, 43 and 44 to the dense MNIST batch-512, 60-epoch comparison, with
both model and optimizer compilation. Run order alternates float/INT8 between
seeds. The optional QT plus real-data Actions job retains all six reports and
shows later-epoch timing alongside accuracy, peak memory and whole training-call
time. The original small-batch comparison remains in the pipeline.

Each seed controls both initialization and the training/validation split; paired
float/INT8 runs use the same manifest. The test set is fixed and evaluates the
final checkpoint. These runs allow nondeterministic CUDA execution, and compiler
caches are not cleared between runs. Three seeds are a useful regression signal,
not a quality-equivalence test or a controlled cold-start benchmark.

The first local run on the RTX 3080 Ti Laptop GPU produced the following results.
All six reports share runtime source hash
`27184f27cd4158db5ad94cdd10776b6d94b2c2dd8ec0248a4712120374298deb`
and lock hash `43ad5c7df81212b3bcd536220201f8888666723507c317e8c86dd538fb595745`.
Each pair's manifest, held-out labels and paths were verified equal. These are
sequential runs with one compiler worker and no concurrent GPU tests.

| Seed | Execution | Test accuracy | Whole-run peak MiB | Median training epoch s, epochs 3–60 | Training wall s |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Float | 93.06% | 254.26 | 0.074 | 28.65 |
| 42 | INT8 | 92.52% | 184.32 | 0.071 | 34.72 |
| 43 | Float | 92.76% | 249.13 | 0.076 | 35.44 |
| 43 | INT8 | 92.42% | 184.32 | 0.071 | 28.49 |
| 44 | Float | 92.74% | 249.13 | 0.074 | 33.22 |
| 44 | INT8 | 92.64% | 184.32 | 0.078 | 28.98 |

INT8 reduces whole-run peak allocation by 26–28% in every pair, while held-out
accuracy is lower by 0.10–0.54 percentage points (mean difference −0.33 points).
Later training phases are faster in two pairs and slower in one; whole training
calls likewise show mixed results. The first INT8 run follows a kernel fingerprint
change, which can trigger recompilation. This supports the memory improvement,
but does not establish a reliable speed win or quality parity. Convergence and
startup profiling remain necessary before recommending QT for this workload.

Local reports, logs, checkpoints and predictions were retained under
`/tmp/mini-trainer-qt-large-batch-multiseed`; these temporary artifacts are not
committed. Future enabled Actions runs retain the corresponding artifacts for
90 days and publish the table in the job summary. No remote job was dispatched
for this local validation.

The synthetic CUDA float/INT8 oracle pair was rerun with the retained kernels;
both reached the required 100% accuracy after training and checkpoint reload.
This verifies the simple task, not convergence equivalence on MNIST or Blair.

### Functional fused requantization

The compiled optimizer's floating-to-INT8 copy now uses a fused row kernel that
returns fresh codes and scales, followed by ordinary tensor copies. A first
in-place custom-operator experiment was rejected: generated SGD code updated
weight storage before calculating momentum that still depended on the old
weights. The existing floating-reference momentum regression caught the error.
The functional version passes that regression, independent stochastic-rounding
and RNG-replay checks, and checkpoint tests. It retains no floating master weight.

The same three-seed profile was rerun, sequentially with no concurrent GPU tests.
Runtime source hash is
`ae6843b04efb8697fd2b830b3be729cbf61a061d3439abc8673c1d1d752d4542`;
the lock hash is unchanged. Paired manifests, held-out labels and paths match,
and all floating accuracies match the preceding run.

| Seed | Execution | Test accuracy | Whole-run peak MiB | Median training epoch s, epochs 3–60 | Training wall s |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Float | 93.06% | 254.26 | 0.076 | 30.77 |
| 42 | INT8 | 93.22% | 175.37 | 0.074 | 30.57 |
| 43 | Float | 92.76% | 249.13 | 0.074 | 31.54 |
| 43 | INT8 | 92.98% | 175.37 | 0.076 | 29.20 |
| 44 | Float | 92.74% | 249.13 | 0.071 | 32.40 |
| 44 | INT8 | 93.00% | 175.37 | 0.081 | 28.41 |

Peak allocation falls another 4.9% relative to the preceding INT8 run, and is now
30–31% below float. Accuracy is 0.16–0.26 percentage points above the paired
floating runs; the changed rounding stream means this is not proof of an
intrinsically better learning algorithm. Whole training calls are shorter in
all three pairs, but later training phases are slower in two. Startup, validation
and logging costs remain part of the whole-run measurement, and caches were not
cleared. These results support retaining the memory improvement; they do not
establish a consistent compute speedup. Repeated timing and broader workloads
remain necessary. The synthetic CUDA pair again reached 100% oracle accuracy.

Reports and predictions are retained locally under
`/tmp/mini-trainer-qt-large-batch-functional-requant`, with the oracle pair under
`/tmp/mini-trainer-qt-oracle-functional-requant`. The shared pipeline will exercise
the new backend without changing its recipes or acceptance thresholds.

A separate eight-trial alternating dispatch probe compared the cached autotuner
with direct launches of its identical selected kernel. Skipping tuner bookkeeping
saved about 4 microseconds for the batch-128 square contraction, but less than
2% for the three larger training shapes. A second launch cache was not added.

### Avoiding disabled-EMA work

The ordinary training loop previously preprocessed a second batch for the teacher
even with EMA disabled. The disabled teacher returned a CUDA zero scalar, adding
allocation and synchronization work despite contributing no loss. The loop now
skips that teacher call and retains a floating zero. Enabled teacher behavior,
loss checks, optimizer overflow handling and scheduler/update gating are unchanged.
EMA itself remains temporarily unsupported.

Custom preprocessing functions now run once per training batch when EMA is
disabled. The former unused call could consume random numbers or cause other
side effects; eliminating those effects is intentional. A regression checks
call counts and exact student updates against direct SGD training.

A warmed 31-batch INT8 training trace shows 62 CUDA stream synchronizations,
down from 124, and 1,519 runtime kernel launches, down from 1,705. Separate
unprofiled MNIST runs used the dense batch-128, 15-epoch, seed-42 recipe with
both model and optimizer compilation. Before/after order alternated across
two trials for float and INT8, with no concurrent GPU tests.

| Precision | Trial | Accuracy, both paths | Median train epoch s before | After | Training wall s before | After |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Float | 1 | 92.44% | 0.179 | 0.181 | 11.50 | 11.48 |
| Float | 2 | 92.44% | 0.196 | 0.187 | 12.79 | 12.14 |
| INT8 | 1 | 93.26% | 0.239 | 0.226 | 12.07 | 11.71 |
| INT8 | 2 | 93.26% | 0.275 | 0.255 | 13.26 | 12.99 |

All paired prediction arrays, including every score, label and path, match
exactly. INT8 later-phase time falls by 5–7% in these trials; float timing is
mixed. Whole training calls are shorter in all pairs, but the smallest difference
is within ordinary timing noise. This removes measured overhead from both paths;
INT8 remains slower than float on this small-batch workload.

The control ran from an isolated copy of `a7007ac`, with source hash
`ae6843b04efb8697fd2b830b3be729cbf61a061d3439abc8673c1d1d752d4542`.
The changed source hash is
`f6069758fdf01810bb343c1042b6b24fbae7422a71b1f1d065957f6949241635`.
Reports, logs and predictions are retained locally under
`/tmp/mini-trainer-disabled-ema-comparison`; traces are
`/tmp/mini-trainer-requant-epoch-trace.json` and
`/tmp/mini-trainer-disabled-ema-epoch-trace.json`.

### Shared storage for cached worker batches

CPU-cache gathers inside repository DataLoader workers now allocate their final
shared buffer before `index_select`. Previously, the gather produced an ordinary
tensor and the multiprocessing queue copied its storage again. This follows the
allocation approach used by PyTorch's worker collator. A spawned-worker regression
checks storage before it reaches the queue, along with exact values, partial
batches, retained-batch ownership and absence of CUDA initialization in workers.

The optimization applies to the repository sampler/collator pair. External
collators retain their ordinary input allocation so they do not receive an extra
shared buffer. Worker-count defaults, parent-side pinning, CUDA caching and
main-process gathers are unchanged.

Two alternating before/after trials used 512 synthetic uint8 RGB tensors at
224×224, batch size 32, one spawn worker per loader and one Torch thread. Each
trial reports the median of seven measured passes after warmup. The control used
an isolated copy of `0520129` with the same updated probe script.

| Trial | Cached batched images/s before | After | Observed gain |
| --- | ---: | ---: | ---: |
| 1 | 13,425 | 14,395 | 7.2% |
| 2 | 13,226 | 15,367 | 16.2% |

The separate scalar reference varied between 13,388 and 14,555 images/s across
these processes, so these are host-specific observations rather than a universal
speedup. The probe verifies identical batch tensors. Timing includes IPC but
excludes worker startup, cache construction, image decoding, H2D and model compute;
it does not establish an end-to-end training or inference gain.

An uncached shared-stack candidate was also tested. It changed throughput from
11,668 to 11,503 and from 12,129 to 11,298 images/s in the two trials. That path
was removed; uncached assembly retains its previous implementation. Fewer copies
did not establish a speed benefit, and changing where work occurs may affect
overlap between decoding/assembly and the queue's sharing work. Real image-decoding
and uncached throughput remain separate optimization targets. In particular,
`get_inference_dataloader` currently streams uncached data; this cached-worker
change does not establish a speed gain for that helper.

The shared [loader probe](../dev/benchmarks/README.md#worker-batch-assembly) now
accepts explicit `--workers` and `--cache` options with picklable readers.
Detailed trial results are retained under `/tmp/mini-trainer-shared-batch-final`.

### Uint8 nearest resize in the streaming reader

The default nearest-neighbor reader now gathers decoded uint8 image rows and
columns directly, avoiding torchvision's temporary float image and conversion
back to bytes. It preserves the legacy nearest coordinate mapping, unchanged-size
identity, output layout and dtype conversion. Other interpolation modes and
outputs larger than 4096 pixels on either axis retain the existing path. Coordinate
caching is limited to 32 one-dimensional arrays, at most about 1 MiB per process.

The replacement was checked against legacy float resizing across thousands of
source/target length combinations and randomized images, including non-square and
singleton dimensions. Layout checks caught and corrected a singleton-stride
difference before measurement. Direct uint8 `interpolate` was also measured but
was slower; that candidate was not adopted.

The [file-backed reader probe](../dev/benchmarks/README.md#streaming-image-reader-comparison)
uses the actual uncached inference loader. Each profile reads the first 128 sorted
JPEG/PNG paths under the dataset's test directory, resizes to 224×224, and batches
16 images. One Torch thread is used, with either zero workers or one spawn worker.
Every batch matches the former reader exactly. The table reports medians of seven
alternating measured passes after equivalence checks and warmup.

| Dataset | Workers | Former reader images/s | Gather reader images/s | Observed gain |
| --- | ---: | ---: | ---: | ---: |
| MNIST | 0 | 4,071 | 4,889 | 20.1% |
| MNIST | 1 | 3,123 | 3,685 | 18.0% |
| Blair | 0 | 2,890 | 3,197 | 10.6% |
| Blair | 1 | 2,049 | 2,653 | 29.5% |

These measurements include decoding, resizing, batch assembly and IPC with a warm
filesystem cache. They exclude worker startup, H2D and model compute, and do not
establish model-inference or training speedups of the same size. The MNIST profile
deliberately resizes to 224×224; it is not the 28×28 MNIST training recipe. Worker
defaults are unchanged, and adding a worker was slower on both datasets here.

Results, per-file content hashes, versions and timing samples are retained under
`/tmp/mini-trainer-reader-comparison`. The standalone CPU command can be reused in
continuous validation wherever the corresponding dataset is available.

### Explicit CUDA graph compilation

The opt-in `qt-cudagraphs` profile repeats the batch-512, 60-epoch, three-seed
comparison with `--compile-mode reduce-overhead` on **both** float and INT8.
The preceding profiles retain their ordinary compilation settings. The optional
QT plus real-data workflow runs this additional profile and retains its summaries
and artifacts. Reproduction is documented in the
[benchmark runner guide](../dev/benchmarks/README.md#cuda-graph-comparison).

A separate warmed, batch-128 MNIST trace recorded 186 `cudaGraphLaunch` calls
across its third training epoch, verifying actual replay with the retained opaque
integer matrix operation. This is distinct from the rejected graph-visible
matrix-operator experiment above. The trace is local at
`/tmp/mini-trainer-cudagraph-epoch-trace.json`; profiling timings are not used below.

The matched runs below used implementation commit `95ca909`, the same RTX 3080 Ti
Laptop GPU and dependency versions as the preceding comparisons, one CPU and
compiler thread, sequential GPU execution, and alternating float/INT8 order.
Runtime source hash:
`ee66fe6f88471cbbb798366f1ef056478ec569ca79140208d8928149cbbcaca2`.
Lock hash:
`43ad5c7df81212b3bcd536220201f8888666723507c317e8c86dd538fb595745`.
All six reports record the explicit mode; paired manifests, test labels and paths
match. Compiler caches were not cleared, and CUDA nondeterminism was permitted.

| Seed | Execution | Test accuracy | Whole-run peak MiB | Median training epoch s, epochs 3–60 | Training wall s |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Float | 93.06% | 199.15 | 0.069 | 31.44 |
| 42 | INT8 | 93.22% | 138.31 | 0.066 | 28.52 |
| 43 | Float | 92.76% | 199.15 | 0.071 | 33.90 |
| 43 | INT8 | 92.98% | 138.31 | 0.066 | 27.78 |
| 44 | Float | 92.74% | 199.15 | 0.064 | 33.74 |
| 44 | INT8 | 93.00% | 138.31 | 0.070 | 27.21 |

INT8 uses 30.5% less peak allocation than the equally configured float model.
Whole training calls are 9–19% shorter. Later training phases are 4–8% faster for
two seeds and 9% slower for the third: a consistent steady-state speedup is still
unproven. All six accuracies match their preceding ordinary-compilation runs;
this is not a statistical quality-equivalence result. Validation, logging,
checkpointing and compilation remain part of whole-call time. No cold-start,
universal model-speed or cross-device claim follows from these runs.

Reports and predictions are local under `/tmp/mini-trainer-mnist-cudagraph-pairs`.
The new continuous profile preserves the same configuration and all unfavorable
results alongside the earlier profiles, rather than replacing their baselines.

The synthetic oracle also reached 100% after checkpoint reload for float and INT8
with the same model mode and compiled MuonAuxAdamW updates. Those reports are at
`/tmp/mini-trainer-cudagraph-oracle`. This checks the simple oracle task, not
normalized-head convergence or general CUDA graph eligibility.

### Embedding publication without graph breaks

`EmbeddingContext.set` now publishes its tensor through dictionary state instead
of assigning a tensor-valued class attribute. This lets Dynamo carry the side
effect through a full model graph, preserving the embedding's gradient path.
Activation, retrieval, nesting checks and exception cleanup retain their existing
interface. Tests compare eager/full-graph input and parameter gradients, including
an embedding auxiliary loss, and verify stable compilation after the classifier's
initial lazy-cache guard settles.

A batch-128 MNIST training trace fell from three compiled forward/backward calls
per batch to one. CUDA graph launches fell from 186 to 62 in the third epoch.
The traces are at `/tmp/mini-trainer-cudagraph-epoch-trace.json` and
`/tmp/mini-trainer-embedding-epoch-trace.json`. Profiling timing is not used below.

The following unprofiled comparisons used a snapshot of `8639c24` as the control,
the same GPU/dependencies as above, dense MNIST, batch 128, 15 epochs, SGD,
FP16 AMP, model and optimizer compilation, CPU cache, zero cache workers, and
one CPU/compiler thread. Float ran before then after; INT8 ran after then before.
Validation tests had finished before these runs. Dataset manifests, held-out
labels and paths match within every pair. Source hashes are:

- Before: `ee66fe6f88471cbbb798366f1ef056478ec569ca79140208d8928149cbbcaca2`.
- After: `ccbfc2dfe1e69eae41f52f7b11bfb5c84641cb167ded827b588cad2196f8a023`.

The lock hash remains `43ad5c7df81212b3bcd536220201f8888666723507c317e8c86dd538fb595745`.
Compiler caches were retained and CUDA nondeterminism was permitted.

| Mode | Seed | Precision | Accuracy before → after | Peak MiB before → after | Median train epoch 3–15 s before → after | Training wall s before → after |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ordinary | 42 | Float | 92.44% → 93.12% | 236.30 → 236.05 | 0.197 → 0.161 | 13.21 → 13.71 |
| Ordinary | 42 | INT8 | 93.26% → 92.84% | 168.95 → 168.95 | 0.274 → 0.228 | 13.85 → 14.94 |
| reduce-overhead | 42 | Float | 92.44% → 93.12% | 193.55 → 217.41 | 0.155 → 0.138 | 11.62 → 11.25 |
| reduce-overhead | 42 | INT8 | 93.26% → 92.84% | 136.96 → 149.60 | 0.174 → 0.184 | 11.62 → 12.38 |
| reduce-overhead | 43 | Float | 92.70% → 93.30% | 193.55 → 217.41 | 0.159 → 0.137 | 12.55 → 11.56 |
| reduce-overhead | 43 | INT8 | 93.10% → 93.02% | 136.96 → 149.60 | 0.179 → 0.155 | 12.19 → 11.74 |

Ordinary compilation improves later-phase time by about 18% for float and 17%
for INT8 in this pair, with essentially unchanged allocation. Whole-call times
increase, so this is not a startup improvement. Under CUDA graph compilation,
float improves in both seeds, while INT8 is mixed and peak allocation rises by
9% for INT8 and 12% for float. INT8 remains slower than equally configured float
on this small-batch model. The change removes a verified graph break; it does not
establish consistent QT speed superiority or universally lower compiled memory.

AMP fusion changes the training trajectory: float accuracy rises by 0.60–0.68
percentage points, while INT8 falls by 0.08–0.42 points. These few short runs
neither establish an intrinsic quality improvement nor prove equivalence.
Reports and predictions are at `/tmp/mini-trainer-embedding-default-pairs` and
`/tmp/mini-trainer-embedding-clean-pairs`. An earlier exploratory comparison at
`/tmp/mini-trainer-embedding-pairs` is retained separately; a short CPU validation
check overlapped that run, so its timings are excluded from this table.

#### Hierarchical Blair check

The same source comparison also ran Blair with 3,704 training images, 912
validation images and the fixed 1,161-image test set. Both sides used TinyConv,
a 64-feature hidden layer, the normalized `HierarchicalClassifier`, the same
reviewed two-level class specification, MuonAuxAdamW, batch 32, five epochs,
FP16 AMP, CPU cache, and model/optimizer compilation with `reduce-overhead`.
INT8 coverage is `fc.hidden` and `fc.linear`; the convolutions remain floating
point. This checks the two-level aggregation path, not every hierarchical head
variant. Manifest, test labels and paths match, and all saved scores are finite.

| Precision | Revision | Species accuracy | Parent accuracy | Peak MiB | Median train epoch 3–5 s | Training wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Float | Before | 65.72% | 79.33% | 72.64 | 0.721 | 22.27 |
| Float | After | 64.86% | 78.12% | 66.38 | 0.679 | 18.47 |
| INT8 | Before | 65.81% | 81.05% | 55.96 | 1.060 | 22.95 |
| INT8 | After | 67.53% | 82.95% | 54.62 | 0.845 | 46.54 |

Later-phase time improves by 6% for float and 20% for INT8, and peak allocation
falls for both. INT8 still takes longer per epoch than float. Its whole-call
time also rises sharply: the first training phase takes 35.60 seconds after the
change versus 13.21 before, consistent with substantial first-use compilation
cost. These caches were not cleared, so this is not a controlled cold-start
comparison. This short single-seed run establishes functional coverage and a
measured steady-phase improvement; it does not establish convergence equivalence,
universal memory behavior, or faster total QT training on Blair. Raw reports,
predictions and the class specification are at `/tmp/mini-trainer-embedding-blair`.

After this change, both synthetic CUDA oracle runs again reached 100% with model
and MuonAuxAdamW optimizer compilation, `reduce-overhead`, FP16 AMP, training and
checkpoint reload. Reports are at `/tmp/mini-trainer-embedding-oracle`.

### Optimizer graph replay: iteration fix and first-use limitation

Revision `0fd9e32` explicitly marks each training iteration when optimizer CUDA
graphs are enabled. Before this fix, the separately compiled optimizer could
access a gradient whose model graph storage had already been retired. Six-batch
regressions now pass for SGD, AdamW and MuonAuxAdamW, with both floating and INT8
weights; the composite optimizer's outer update counter remains intact.

The following local MNIST runs used the same source and lock hashes, dataset
manifest, seed 42, dense model, SGD (learning rate 0.3, momentum 0.9), batch 128,
15 epochs, FP16 AMP, CPU cache, zero loader/cache workers, model compilation with
`reduce-overhead`, and optimizer compilation. Hardware was the RTX 3080 Ti Laptop
GPU with PyTorch 2.12.0/CUDA 13.0; Torch and Inductor each used one thread.
All completed runs reloaded the checkpoint for held-out inference and saved
finite scores. This is a single-seed functional/performance probe, not a quality
gate or convergence comparison.

| Weights | Optimizer graphs | Test accuracy | Peak MiB | Median train epoch 3–15 s | Training wall s |
| --- | --- | ---: | ---: | ---: | ---: |
| Float | Off | 93.12% | 217.41 | 0.193 | 14.56 |
| Float | On | 93.12% | 217.42 | 0.209 | 15.43 |
| INT8 | Off | 92.84% | 149.60 | 0.292 | 37.26 |
| INT8 | On, warmed retry | 92.84% | 167.60 | 0.230 | 18.04 |

**The first INT8 graph-enabled attempt failed**, during the first compiled
backward before any optimizer update, with `These storage data ptrs are not
allocated in pool (0, 1) but should be`. The baseline and subsequent graph-enabled
retry completed. First-use kernel tuning or compilation is a suspected cause,
not an established diagnosis; the warmed retry does not resolve this failure.
Do not interpret these results as reliable cold-start support.

Run order was float off, float on, failed INT8 on, INT8 off, INT8 on retry.
Caches were not cleared, and no tests ran concurrently. The failed attempt
warmed caches, so whole-call times are not a controlled comparison of compilation
cost. Optimizer replay reduced the INT8 later-phase median by about 21%, but
raised its peak allocation by 12%. INT8 with replay still ran about 19% slower
per later epoch than float without replay, while using about 23% less peak
memory. Float replay was slower. Keep optimizer graphs opt-in; these results
do not establish a general QT speed advantage.

Reports, predictions, logs, and the original failures are retained locally in
`tmp-optimizer-cudagraphs/pairs` and `tmp-optimizer-cudagraphs/pairs-fixed`
(ignored generated artifacts, not repository-hosted results). Reproduce each
completed configuration with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
MPLCONFIGDIR=/tmp/mini-trainer-mpl MPLBACKEND=Agg \
.venv/bin/python -m dev.benchmarks.run --output /tmp/mnist-optimizer-graphs \
  --dataset mnist --data-root examples/mnist --seed 42 --model-profile dense \
  --optimizer sgd --learning-rate 0.3 --epochs 15 --batch-size 128 \
  --compile --compile-mode reduce-overhead --compile-optimizer \
  --device cuda:0 --dtype float16 --cache CPU --cache-workers 0 \
  --allow-nondeterministic
```

Add `--quantized-training` for INT8 and `--optimizer-cudagraphs` for optimizer
replay. Fresh-cache failure reproduction and a fix must precede recommending
this combination in a continuous benchmark profile.

#### First-use tuning failure follow-up

Forced retuning reproduced the first-backward pool failure three times with
already compiled kernels. Allocation diagnostics showed that the mismatched
storage reference expired during the graph check's garbage collection. Rejected
Triton candidates had requested 122,880 bytes of shared memory on a GPU limited
to 101,376 bytes; their exception tracebacks retained temporary tensors.

The local benchmark callback now clears tracebacks for the same expected
candidate failures that Triton already scores with infinite timing. This releases
those tensors promptly. Unexpected errors still propagate, kernel arithmetic is
unchanged, and CUDA graph assertions remain enabled. No global garbage collection
or third-party monkey patch is added. A diagnostic two-epoch MNIST run completed
with forced retuning after this change.

Regression coverage includes immediate tensor release without `gc.collect()`,
unexpected-error propagation, and three dense-model forward/backward/update
iterations with fresh local tuning. The latter disables only that tuner's memory
and disk result caches; it preserves installed dependencies and compiled kernel
caches and requires no dataset download:

```bash
CUDA_VISIBLE_DEVICES=0 RUN_CUDA_TESTS=1 OMP_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
bash dev/check.sh test tests/test_quantized_training.py -k fresh_kernel_tuning
```

This addresses the reproduced failure mechanism. The earlier timing table remains
historical evidence, including its failed attempt; it does not become a controlled
cold-start timing comparison or establish a general QT speed advantage.

The production fix also completed the full 15-epoch MNIST configuration above
with forced local retuning: 92.84% held-out accuracy, finite saved scores,
167.60 MiB peak allocation, 0.203 s median train epoch 3–15, and 24.33 s training
wall time. Checkpoint reload and inference completed. Results are retained in
`tmp-optimizer-cudagraphs/tuning-fixed-mnist`; source hash
`23df40b2027a8a18de797f60c93c84aad72653c797a35e0f1883e9efcccbf2e4`.
This verifies the formerly failing path. It forces tuner results to be recomputed,
not a completely empty compiler cache, and has no concurrent matched float run,
so its timing is not evidence of a new speedup.

### Rejected experiment: compiler-visible row requantization

After `48f8058`, a warmed fourth training epoch was profiled on the same dense
MNIST configuration (31 batches of 128, FP16 AMP, CPU cache, zero workers).
INT8 used model and optimizer graph replay; the float reference used model graph
replay and the ordinary compiled optimizer. These are the respective configurations
under investigation, not a controlled test of the optimizer graph flag alone.
Profiler runs are diagnostic and must not be used as wall-time benchmarks.

In the INT8 trace, the scaled matrix kernels accumulated 9.96 ms of device time,
versus 19.89 ms for the large floating update kernels and 7.32 ms for row
requantization. Gradient unscaling and clipping's scaling pass each took about
7.4 ms. Nested compiled-region timings overlap their child kernel timings and
must not be added to them. The traces identify update-buffer traffic and gradient
handling as worthwhile targets; they do not establish a kernel-only speedup.
Local traces and tables are retained under
`tmp-optimizer-cudagraphs/profile-int8` and `profile-float`.

A candidate replaced the opaque row requantization call during fake-tensor
tracing with TorchAO's tensor arithmetic, intending to let Inductor fuse the
floating update and row reduction. Eager execution was unchanged. All 67 CUDA
model/optimizer regression cases passed, but the following real training results
did not justify retaining it. The candidate was reverted.

| Seed | Requantization | Test accuracy | Peak MiB | Median train epoch 3–15 s | Training wall s |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | Existing kernel | 92.84% | 167.60 | 0.144 | 10.95 |
| 42 | Tensor arithmetic candidate | 93.10% | 167.60 | 0.175 | 17.04 |
| 43 | Existing kernel | 93.02% | 167.60 | 0.144 | 11.20 |
| 43 | Tensor arithmetic candidate | 92.78% | 167.60 | 0.146 | 11.88 |

Settings match the 15-epoch SGD/INT8 optimizer-graph MNIST probe above. Both
baseline seeds ran before the candidate was installed; both candidate seeds ran
after its CUDA checks finished. No tests or other benchmark runs overlapped the
timed runs. Caches were not cleared, so the first candidate's whole-call time
includes additional compilation work and is not a controlled cold-start measure.
Peak memory did not fall, later-phase time did not improve, and stochastic
rounding changed the trajectories. Two seeds do not establish quality equivalence.
Reports are retained locally in `tmp-optimizer-cudagraphs/fusion-before-{42,43}`
and `fusion-after-{42,43}`.

The next update-path experiment should avoid constructing the full floating
updated-weight buffer explicitly, while keeping new INT8 codes/scales as
functional outputs and making the final parameter copies visible to the compiler.
It must preserve optimizer state, scalar learning-rate precision, AMP gating,
checkpoint behavior, and stochastic-rounding quality. Simply exposing more tensor
arithmetic did not provide that improvement here.

### Rejected experiment: functional fused weight update

A second candidate reused the native update kernel to produce fresh INT8 codes
and scales directly, followed by compiler-visible parameter copies. It avoided
an explicit floating updated-weight result in that kernel's caller and consumed
learning-rate tensors on CUDA. Six kernel cases matched the existing native
update exactly across FP32, FP16 and BF16, with division-based updates and
noncontiguous inputs. All 67 CUDA model/optimizer cases and 23 kernel cases passed.

Inspection confirmed that compiled AdamW reached the new operator. Compiled SGD
continued through its existing decomposition, including with `foreach=False`, so
this experiment did not establish an SGD optimization. The following AdamW runs
used the dense MNIST profile above, learning rate 0.001, 15 epochs, batch 128,
FP16 AMP, CPU cache, and both model and optimizer graph replay. Baselines ran from
an isolated snapshot of `e5387d8`; the candidate used the working checkout and the
same virtual environment and data. No timed run overlapped another run or tests.

| Seed/run | Existing median train epoch 3–15 s | Candidate median s | Existing / candidate accuracy |
| --- | ---: | ---: | ---: |
| 42, initial pair | 0.152 | 0.162 | 94.14% / 94.34% |
| 43, reversed order | 0.267 | 0.133 | 94.02% / 94.00% |
| 42, warmed before→after | 0.139 | 0.153 | 94.14% / 94.34% |
| 42, warmed after→before | 0.197 | 0.220 | 94.14% / 94.34% |

All runs peaked at 218.10 MiB. Timing varied substantially between runs, so the
second seed alone would give a misleading speedup claim. Both additional warmed
pairs favored the existing implementation by about 10–12%. Whole-call times in
those pairs were 13.05 vs 13.69 s and 16.68 vs 17.11 s. The candidate was reverted;
passing numerical tests alone did not justify extra kernel complexity without a
measured memory or speed benefit. Its patch and reports are retained locally in
`tmp-optimizer-cudagraphs/rejected-functional-update.patch`,
`functional-adam-{before,after}*`, and `adam-repeat-*`.

Update fusion remains a possible future direction, but merely placing the final
weight arithmetic inside the row kernel did not improve this workload. Further
performance validation should also include larger compute workloads, where the
integer GEMMs have more opportunity to offset update and scheduling overhead,
and should retain alternating run order and report variability.

### Larger-batch model and optimizer graph results

The current implementation at `1152b78` was measured with the shared
`qt-cudagraphs` profile and a second pass enabling optimizer graph replay for
both precisions. Each pass used seeds 42, 43 and 44, alternating float/INT8 order
by seed, with the dense MNIST model, SGD (learning rate 0.3), batch 512, 60 epochs,
FP16 AMP, CPU cache and zero workers. Both precisions used model `reduce-overhead`
and optimizer compilation. Hardware was the RTX 3080 Ti Laptop GPU, PyTorch
2.12.0/CUDA 13.0, with one Torch thread and one Inductor compiler thread.

| Seed | Optimizer graphs | Float / INT8 accuracy | Float / INT8 peak MiB | Float / INT8 median train epoch 3–60 s | Float / INT8 training wall s |
| --- | --- | --- | --- | --- | --- |
| 42 | Off | 93.06% / 92.82% | 221.73 / 167.91 | 0.0640 / 0.0568 | 38.26 / 48.87 |
| 43 | Off | 92.76% / 93.38% | 221.73 / 167.91 | 0.0581 / 0.0626 | 31.37 / 28.82 |
| 44 | Off | 92.74% / 92.40% | 221.73 / 167.91 | 0.0604 / 0.0606 | 32.51 / 26.41 |
| 42 | On | 93.06% / 92.82% | 221.74 / 171.92 | 0.0598 / 0.0577 | 29.11 / 28.14 |
| 43 | On | 92.76% / 93.38% | 221.74 / 171.92 | 0.0623 / 0.0565 | 34.21 / 27.80 |
| 44 | On | 92.74% / 92.40% | 221.74 / 171.92 | 0.0595 / 0.0566 | 33.12 / 26.55 |

With optimizer replay enabled, INT8 used 22.5% less peak memory and its later
training phases were 3.6%, 9.3% and 4.9% faster than float with the same setting.
Whole-call times were also lower in all three pairs. Comparing INT8 replay with
the faster float median observed across either setting for each seed leaves
smaller advantages of 3.6%, 2.7% and 4.9%. These are measured benefits in this
compute profile, not guarantees for other models or hardware. Without optimizer
replay, the later-phase speed comparison was mixed.

INT8-minus-float accuracy differences were −0.24, +0.62 and −0.34 percentage
points. All 12 runs completed checkpoint reload and held-out inference with
finite scores. Paired dataset manifests match; all runs share source hash
`23df40b2027a8a18de797f60c93c84aad72653c797a35e0f1883e9efcccbf2e4`
and the same lock hash. No tests or other benchmark runs overlapped these runs.
Caches were not cleared: first-use costs and sequential cache warming affect
whole-call comparisons, and these results do not establish cold-start speed or
statistical quality equivalence. Accuracy still has no automated real-data gate.

Reports and predictions are retained locally in
`tmp-optimizer-cudagraphs/large-current` and `large-optimizer-graphs`. The second
configuration is now reproducible through the shared
`qt-optimizer-cudagraphs` mode and included in the optional QT plus real-data GPU
workflow, with visible summaries and retained artifacts. Use
[the shared command](../dev/benchmarks/README.md#model-and-optimizer-graph-comparison)
for future comparisons rather than treating these timings as fixed thresholds.

### Loading audit on the delivered implementation

The shared loader and reader probes were rerun after `d9ecab6`, using one Torch
thread, seven measured repetitions after warmup, and alternating reference/current
order. All comparisons verified exactly identical batches.

| Path | Workers | Reference samples/s | Current samples/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| CPU cache, synthetic 64×64 | 0 | 241,783 | 625,168 | 2.59× |
| CPU cache, synthetic 64×64 | 1 | 40,090 | 43,843 | 1.09× |
| MNIST streaming, resized to 224×224 | 0 | 4,037 | 4,881 | 1.21× |
| MNIST streaming, resized to 224×224 | 1 | 2,447 | 3,444 | 1.41× |
| Blair streaming, resized to 224×224 | 0 | 2,309 | 2,643 | 1.14× |
| Blair streaming, resized to 224×224 | 1 | 1,956 | 2,448 | 1.25× |

The cached probe used 2,048 generated uint8 images and batch 64, comparing scalar
fetch/stack against batched gathering. The reader probe used 128 real images per
dataset and batch 16 through `get_inference_dataloader`, comparing the retained
Torchvision resize path with the current reader. Worker runs used spawn. Timings
exclude worker startup, cache construction and model compute; the cached probe
also excludes decoding, and both exclude H2D. Reader filesystem caches were warm.
The streaming reader and batching implementation are shared by float and INT8
inference; cached gathering also serves training. These ratios are loading-path
measurements, not end-to-end model throughput claims.

Reports, input file hashes, all repetition times, and equality flags are retained
in `tmp-optimizer-cudagraphs/audit/{cached-*,reader-*}.json`. Reproduce with the
shared `dev.benchmarks.loader` and `dev.benchmarks.reader` commands documented in
[the benchmark guide](../dev/benchmarks/README.md).

The audit also found that automatic worker selection still omitted CPU bandwidth
quotas and Slurm task allocations. The budget helper now includes cgroup v1/v2
limits (including visible ancestors) and `SLURM_CPUS_PER_TASK`, while preserving
explicit overrides and the existing reserve/caps. Regression tests cover quota
and namespace parsing, fractional/unlimited/malformed limits, scheduler budgets,
and affinity fallbacks. This closes an allocation-aware defaulting gap; it does
not infer the instantaneous load or private CPU shares of competing processes.
## EfficientNetV2-S on Blair: initial representative-model comparison

On 2026-09-09, revision `c06470c` completed four local CUDA runs using the actual
EfficientNetV2-S backbone with a symmetric 1280-feature hidden layer and normalized
flat or hierarchical classifiers. All four used the same reviewed Blair manifest
(3,704 train / 912 validation / 1,161 test), seed 42, five epochs, size 128, batch
32, MuonAuxAdamW at head LR 0.01, FP16 AMP, CPU cache, zero workers, no augmentation
and no model/optimizer compilation. Backbone initialization was random, not
pretrained; these are execution/convergence diagnostics rather than a reproduction
of the notebook's pretrained training recipe.

| Head | Precision | Fine accuracy | Parent accuracy | Peak allocated MiB | Median train phase, epochs 3–5 (s) | Training call (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Flat | Float | 56.33% | — | 1230.15 | 13.027 | 75.706 |
| Flat | INT8 | 34.80% | — | 1191.86 | 15.433 | 102.584 |
| Hierarchical | Float | 48.23% | 62.62% | 1230.16 | 15.061 | 93.804 |
| Hierarchical | INT8 | 46.43% | 60.12% | 1188.37 | 16.457 | 99.235 |

The current head-only recipe reduced whole-model parameter storage from 87,407,112
to 82,401,132 bytes (5.73%). Peak allocated memory fell by about 3.1% for the flat
head and 3.4% for the hierarchical head. Later training phases were approximately
18.5% and 9.3% slower respectively. All 170 convolutions remained floating.

The flat accuracy regression is material: -21.53 percentage points, versus -1.81
points for hierarchical fine accuracy and -2.50 for parent accuracy. One short
seed does not establish the cause or statistical generality, and matching seed
values does not guarantee identical stochastic training trajectories under QT.
Nevertheless, these results do not support recommending this recipe for the
representative model. Next compare pretrained initialization and inspect the
optimization/numerical behavior before interpreting broader quality effects.

All four runs trained, reloaded their checkpoints and produced finite held-out
scores. Source, lock and dataset manifest hashes match across the four reports.
Source SHA256 is `8da1c2dbc1218d4d655925e0497fb0598ecd48bcafa6197c4ec8d5328fbb2bf6`;
manifest SHA256 is `1cef6c7d9133d889b6c8eff0ffef29b1db5d6653ab4adf0c005739af8c2921c6`.
Reports, logs, checkpoints and predictions are retained locally under ignored
`tmp-efficientnet-baseline/`. Commands are in the
[benchmark guide](../dev/benchmarks/README.md#representative-efficientnetv2-configuration).

Hardware was the RTX 3080 Ti Laptop GPU; runtime versions are recorded per report.
Runs were sequential in flat-float, flat-INT8, hierarchical-float,
hierarchical-INT8 order without clearing compiler caches. Timings include local
loading/logging effects and do not predict A40/A100/B300 or Spark/desktop speed.
The missing optional dendrogram visualization dependencies emitted warnings;
training and prediction completed. No ONNX or target-hardware inference was tested.

### Pretrained initialization and fixed-batch numerical check

The matched pretrained comparison at revision `f14f62d` used the same settings,
source hash and dataset manifest as above, adding `--pretrained`. This loaded
Torchvision's cached `efficientnet_v2_s-dd5fe13b.pth` backbone, with a newly
initialized symmetric normalized head. Execution order was INT8 then float for
each head; no other runs overlapped the measurements.
The pretrained file SHA256 is
`dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74`.

| Head | Precision | Fine accuracy | Parent accuracy | Peak allocated MiB | Median train phase, epochs 3–5 (s) | Training call (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Flat | Float | 83.46% | — | 1234.15 | 14.673 | 89.091 |
| Flat | INT8 | 83.63% | — | 1191.36 | 15.291 | 88.937 |
| Hierarchical | Float | 78.21% | 90.61% | 1234.16 | 14.558 | 92.052 |
| Hierarchical | INT8 | 80.62% | 91.30% | 1191.37 | 14.994 | 95.987 |

The large flat accuracy regression did not recur with pretrained initialization
in this seed. That neither establishes superiority nor identifies the cause of
the random-initialization regression. INT8 later training phases remained 4.2%
slower for the flat head and 3.0% slower for the hierarchical head; peak allocation
was about 3.5% lower. Whole-call timing is mixed. All checkpoints reloaded and
produced finite held-out scores; coverage is still limited to the two head layers.
Reports and predictions are retained under ignored `tmp-efficientnet-pretrained/`.

After those runs, a separate diagnostic compared deep-copied flat models before
any optimizer update on 32 identical Blair training images, sampled with seed 42.
Dropout and stochastic depth were disabled, BatchNorm remained in training mode,
and both used FP16 AMP with loss scaled by 1024 for backward. This isolates an
initial forward/backward discrepancy, not stochastic update or convergence behavior.

| Initialization | Float / INT8 loss | Score relative L2 error | Backbone gradient relative L2 error | Hidden gradient relative L2 error | Output gradient relative L2 error |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | 3.74278 / 3.74148 | 1.43% | 4.68% | 4.61% | 1.31% |
| Pretrained | 3.77631 / 3.77755 | 1.92% | 9.10% | 9.61% | 1.53% |

Relative errors are L2 difference divided by the floating reference norm,
aggregated over each parameter group. The initial pretrained gradient discrepancy
was larger despite its better eventual accuracy: these initial errors alone do
not explain the earlier result. Preparation uses deterministic rounding; subsequent
updates use stochastic rounding and share the CUDA RNG with stochastic layers.
Further investigation should separate repeated-seed variation, optimizer updates
and stochastic-layer effects before changing training numerics. The diagnostic
script, exact image paths and results are retained beside the reports as
`gradient_probe.py`, `gradient-probe.json` and `gradient-probe.log`.

### Large-class head capacity and initialization

The production envelope includes 10,000–1,000,000 classes. A 25-class head cannot
represent that regime. Initial 100k-class capacity probes failed in both float
and INT8 before training: spherical initialization evaluated `(W @ W.T) @ W`,
requesting a 100k-by-100k FP32 intermediate (37.25 GiB). At one million classes
that intermediate would require 3.64 TiB.

The initializer now uses `W @ (W.T @ W)` when class count exceeds embedding
width. This keeps the smaller Gram matrix (6.25 MiB at width 1280) while preserving
the mathematical spherical-repulsion update. Floating-point association changes,
so initialization is not bitwise identical for these tall heads. Tests compare
100 iterations against the former formula and reject class-by-class allocations
during construction of a 10k-class normalized head. Heads no taller than their
embedding width retain the previous association.

After that fix, eight fresh CUDA processes exercised full EfficientNetV2-S with
random initialization, symmetric normalized heads, 32 synthetic uint8 images at
128x128, FP16 AMP and eager MuonAuxAdamW. Hierarchical models used a synthetic
two-level taxonomy with 100 leaf classes per parent. Three warmup steps preceded
five measured steps; each measured loss was finite and each optimizer update
executed. The probe follows the trainer's forward/zero-grad/backward/clip/update
order, but excludes loading, logging, scheduling, checkpointing and deployment.

| Classes | Head | Float / INT8 parameter MB | Float / INT8 peak allocated MiB | Float / INT8 median step ms |
| --- | --- | ---: | ---: | ---: |
| 10,000 | Flat | 138.56 / 95.29 | 1450.4 / 1373.2 | 89.78 / 78.68 |
| 10,000 | Hierarchical | 138.56 / 95.29 | 1450.4 / 1373.2 | 91.96 / 89.65 |
| 100,000 | Flat | 600.08 / 211.57 | 3910.4 / 4130.3 | 126.87 / 138.36 |
| 100,000 | Hierarchical | 600.08 / 211.57 | 3910.4 / 4130.4 | 144.66 / 144.83 |

MB here is decimal; MiB is binary. At 100k classes, the physical parameter
reduction is substantial, but peak training allocation increases. QT's transient
floating normalization/gradient/update tensors therefore need investigation;
quantized parameter storage does not guarantee lower peak memory. These five-step,
single-process samples are capacity diagnostics, not robust speed estimates or
quality comparisons. Only head layers are quantized. No million-class training,
ONNX runtime or target-hardware performance claim follows from them.

The original failure logs and probe are retained under ignored
`tmp-efficientnet-classes/`; rerun logs, script and JSON measurements are under
`tmp-efficientnet-classes-fixed/`. Both used the local RTX 3080 Ti Laptop GPU.
One-million-class validation should record initialization peak separately from
training peak, then check actual updates, checkpoint/reload and deployment on a
machine with enough memory. Eliminating the quadratic Gram matrix leaves linear
parameter, gradient and temporary storage costs; it does not establish that a
million-class run fits the local 16 GiB GPU.

### Warmed training trace

A separate pretrained 25-class float run captured three warmed training batches
through the actual Blair training loop, including MuonAuxAdamW updates. The trace
contains 137.85 ms of summed CUDA kernel/copy/memset event duration; a single
NCHW-to-NHWC conversion kernel contributes 7.52 ms (5.45%). Depthwise convolution
weight-gradient, batch normalization, SiLU and optimizer kernels are also visible.
This suggests examining layout and backbone costs alongside the large-head cases.

The denominator counts device events once and excludes nested CPU operators and
GPU annotations. It is not wall time or a speedup prediction: profiling perturbs
execution, and summed durations do not account for overlap. Raw trace, grouped
operator data, script and run outputs are retained under ignored
`tmp-efficientnet-profile/`. Do not sum the grouped operator table directly;
it includes nested attribution as well as device events and would double-count.

### Bounded INT8 normalization backward storage

The large-head follow-up isolates a concrete temporary-allocation cost in QT's
weight-normalization Jacobian. The previous expression materialized floating unit
directions and projection/update intermediates. A CUDA row kernel now computes
the same Jacobian while allocating only the returned direction and magnitude
gradients. CPU execution, widths above 16,384 and higher-order differentiation
retain the Torch expression. The kernel does not change quantization bit widths,
optimizer state, row scales, the forward normalization or stochastic rounding.

At 100k rows by 1280 columns with FP32 gradient metadata, three warmups and five
measured calls reduced extra allocated bytes from 1,536,800,768 to 512,400,384.
Median isolated backward duration fell from about 18.5 ms to 2.7 ms. The probe
excludes caller-owned codes, scales, magnitudes, norms and upstream gradients;
it is not an end-to-end training measurement. Reports are retained under ignored
`tmp-quantized-normalization/`.

Fresh full-model capacity probes used the same synthetic-input settings as the
large-class comparison above. Both heads had finite losses and applied every
measured optimizer update:

| Head, 100k classes | Float / INT8 peak allocated MiB | Float / INT8 median step ms |
| --- | ---: | ---: |
| Flat | 3910.4 / 3231.4 | 129.47 / 135.97 |
| Hierarchical | 3910.4 / 3231.4 | 125.64 / 120.12 |

Peak allocation during the five measured steps is now 17.4% lower under QT for
both heads. Model construction and the three warmup steps are excluded from that
peak. Step timings remain mixed and these short sequential samples do not
establish a portable speed gain. Reports are under `tmp-efficientnet-normalization/`.

The CUDA model suite passed 73 cases covering the normalization Jacobian,
negative scales, zero/trainable magnitudes, FP32/FP16/BF16, strided upstream
gradients, compilation, optimizer graph replay, checkpoint and inference behavior.
A separate allocation-bound regression also passed. Source hashing includes the
new kernel so compiled backward graphs cannot reuse the previous implementation.
The full CPU-default suite passed with 351 tests passed, 141 skipped and the
existing EMA expected failure; static lint, formatting and import checks passed.

Both pretrained Blair INT8 heads were rerun for five epochs under the previous
settings and produced finite held-out predictions after checkpoint reload. Flat
accuracy was 84.07% (previous INT8 83.63%); hierarchical fine/parent accuracy was
79.16%/91.73% (previous INT8 80.62%/91.30%). These mixed single-seed changes do not
establish quality equivalence or superiority; reduction-order changes can alter
quantized optimization trajectories. Reports are under `tmp-normalization-blair/`,
with source SHA256 `6aba096dca741e09863487cf4ed96dce21dea44b68b5612625caa3f3beb347d2`.
Repeated-seed convergence, million-class capacity, ONNX deployment and performance
on the intended target machines remain open.

### EfficientNetV2 ONNX CPU export and inference quantization

The pretrained floating Blair checkpoints were exported through the existing
generic API for both symmetric normalized EfficientNetV2-S heads. ONNX Runtime
CPU parity passed at batch sizes 1, 2, 4 and 8, including 16 real held-out images.
The maximum observed absolute score difference was 1.67e-5 for the flat head and
1.29e-5 for the hierarchical head, within the existing combined rtol=1e-4,
atol=1e-5 checks. The test suite now also exports both actual architectures with
random initialization and verifies dynamic batch sizes 1–4 without downloads.

Runtime dependencies were installed at the locked versions (ONNX 1.22.0,
ONNX Runtime 1.29.0, ONNX Script 0.7.1), with constraints preserving the existing
CUDA PyTorch, Torchvision, TorchAO and NumPy versions. This was an environment
installation, not a dependency declaration or lock-file change.
The experiment's complete package-version record is retained in
`tmp-efficientnet-onnx/environment.json`; its protobuf dependency resolved to
7.36.1 rather than the lock's 7.35.0, so this is not a fully locked-environment run.

A separate exploratory post-training quantization trial used ONNX Runtime static
QDQ with per-channel INT8 weights and signed INT8 activations, targeting Conv,
Gemm and MatMul. MinMax calibration used 128 training images sampled with seed 42;
neither validation nor test images entered calibration. The exported graph
already folded BatchNorm into convolution; preprocessing ran shape inference
without further graph optimization before quantization. This follows the starting
point described in [ONNX Runtime's quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
but the result is **not an acceptable deployment recipe**:

| Head | Float / INT8 fine accuracy | Float / INT8 parent accuracy | Float / INT8 graph plus weight bytes |
| --- | ---: | ---: | ---: |
| Flat | 83.55% / 55.73% | — | 88,624,980 / 25,398,200 |
| Hierarchical | 78.12% / 70.28% | 90.61% / 85.62% | 88,645,350 / 25,418,559 |

All 1,161 held-out images produced finite outputs. Float and INT8 were evaluated
through the same CPU provider and preprocessing; the floating accuracy differs
slightly from the earlier CUDA AMP results. Artifacts are roughly 71% smaller,
but the execution profile confirms only 63 QLinearConv operations and two QGemm
operations, alongside 107 floating Conv operations. QDQ nodes and integer stored
weights therefore do not establish integer execution throughout the backbone.

These are exploratory held-out checks, not a confirmatory quality comparison or
a timing study. Subsequent recipe tuning should use validation images, localize
weight/activation error, and investigate incomplete quantized-operator fusion.
Do not select a production recipe from artifact size alone. No ONNX GPU or ARM
execution was tested, and no native QT checkpoint was converted in this trial.

Exports, calibration records with source-image hashes, full predictions, runtime
profiles, report JSON and scripts are retained under ignored
`tmp-efficientnet-onnx/`. Preprocessing remains external to the graph; the local
experiment used the checkpoint's repository preprocessing and does not yet provide
a standalone raw-image deployment recipe. Missing telemetry/cache write access
emitted environment warnings; export and inference completed.

### ONNX activation calibration, execution coverage and macro metrics

A follow-up on the same floating checkpoints separates execution coverage from
quantization error. Recoding signed activation zero points as unsigned values
(`zero_point + 128`), retaining scales and signed weights, gave bitwise-identical
unoptimized outputs on a fixed real validation batch for both heads. With ORT
optimization enabled, all 170 convolutions then executed as QLinearConv, rather
than 63 QLinearConv plus 107 floating Conv. This is a representation-dependent
fusion result on this CPU/provider build, not evidence that other providers behave
the same way. It did not solve the MinMax quality loss.

Diagnostic weight-only and activation-only graphs were evaluated on all 912
validation images. Float / weight-only / activation-only accuracy was
84.10% / 83.22% / 55.92% for flat, 80.92% / 80.92% / 65.68% for hierarchical fine,
and 91.12% / 90.90% / 77.96% for hierarchical parent. These controls isolate error
sources; they are not efficient deployment graphs. Activation quantization is the
larger problem here, although weight and activation errors interact.

One focused Percentile 99.9 calibration trial used the same 128 training images
(seed 42), asymmetric activation ranges, unsigned INT8 activations and per-channel
signed INT8 weights. Histogram collection used 2,048 bins and batches of eight,
accumulating histograms across batches without retaining every raw activation.
Conv, Gemm and MatMul remained the target operations. Validation accuracy recovered
to 82.68% flat and 78.62% / 89.47% hierarchical fine / parent. This follow-up used
validation data for diagnosis, not a new inspection of the test split.

Quality was also evaluated through the local `mini_metrics` checkout at commit
`70cc69adc05362863439277048e06386c1f885e1` (clean working tree). The table uses
ordinary Macro-F1, Macro-Recall, Macro-Precision, Coverage and Theil's U directly
from that package. Values are proportions, not percentages.

| Output / recipe | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat / float | 0.76207 | 0.75442 | 0.78957 | 1.00000 | 0.81479 |
| Flat / signed MinMax | 0.44030 | 0.45594 | 0.62608 | 1.00000 | 0.63969 |
| Flat / unsigned Percentile | 0.76082 | 0.75161 | 0.78966 | 1.00000 | 0.81173 |
| Hierarchical fine / float | 0.70922 | 0.69573 | 0.79877 | 1.00000 | 0.78258 |
| Hierarchical fine / signed MinMax | 0.59050 | 0.58995 | 0.74886 | 1.00000 | 0.72673 |
| Hierarchical fine / unsigned Percentile | 0.69824 | 0.67946 | 0.75561 | 1.00000 | 0.76686 |
| Hierarchical parent / float | 0.85936 | 0.82759 | 0.92038 | 1.00000 | 0.85139 |
| Hierarchical parent / signed MinMax | 0.77617 | 0.74597 | 0.89192 | 1.00000 | 0.79788 |
| Hierarchical parent / unsigned Percentile | 0.84110 | 0.80161 | 0.91253 | 1.00000 | 0.82635 |

Every recipe uses exactly the same 912 validation images and manifest class
mapping, with independent top-1 predictions at each available output level.
There is no threshold optimization, resampling, known-label filtering or inferred
parent output for the flat head. Confidence is softmax maximum; threshold zero
makes Coverage 100% by construction. This does not measure useful abstention.
The flat Macro-F1 decrease is 0.13 percentage points, versus 1.10 / 1.83 points
for hierarchical fine / parent. These single-checkpoint descriptive results do
not establish a production acceptance threshold or statistical equivalence.

For a CSV in the repository's `mini_metric.csv` schema, the exact metric call is:

```python
from mini_metrics.metrics import MacroF1, evaluate_file

metrics = evaluate_file(
    "mini_metric.csv",
    optimal=False,
    threshold=0,
    known_only=False,
    per_class=False,
    simple=True,
    hierarchical=False,
    pattern=r"^(f1|recall|precision|coverage|theilU)$",
    opt_crit=MacroF1,
    verbose=0,
)
```

In this package version, unprefixed `f1`, `recall` and `precision` identify macro
metrics; micro variants have a `micro_` prefix. Pin the metrics revision when
comparing reports. The installed package was not replaced: this experiment
selected the sibling checkout with an explicit process-local `PYTHONPATH`.
The package itself does not depend on that filesystem layout. The publication
bootstrap script provides a reference for a later confidence-thresholded study;
its `optimal=True` mode changes the evaluated sample set by splitting calibration
from evaluation. Such a study should preserve paired splits across recipes and
report Coverage alongside quality, separately from this full-coverage comparison.

Warm CPU inference timing used one ORT intra/inter-op thread, fixed preprocessed
real inputs, three warmups and eleven measured repetitions per batch size.
Recipe execution order alternated forward/reverse each repetition. Medians below
are milliseconds per `Session.run`, excluding image loading, preprocessing,
session construction and model export; these are not end-to-end CLI latencies.

| Head / batch | Float | Signed MinMax | Unsigned Percentile |
| --- | ---: | ---: | ---: |
| Flat / 1 | 23.29 | 35.89 | 11.91 |
| Flat / 8 | 162.81 | 243.15 | 80.67 |
| Hierarchical / 1 | 23.86 | 36.75 | 11.96 |
| Hierarchical / 8 | 175.44 | 252.97 | 84.56 |

A separate execution profile of the Percentile graphs confirms 170 QLinearConv
and two QGemm operations, with no floating Conv. Sigmoid, multiplication,
normalization and other operations still execute in floating point. These are
local Intel i7-12800H x86 CPU measurements with ORT 1.29.0, not Raspberry Pi, ARM,
CUDA-provider or production throughput verification. Calibration and activation
representation both differ between the timed quantized recipes, so timing does
not isolate either change. Quality acceptance, repeated-process timing and target
hardware measurements remain open. Native CUDA QT checkpoint export is still a
separate unsupported path.

Local scripts, scores, CSV inputs, exact metric JSON, calibration histograms,
raw timing samples and execution profiles are retained under ignored
`tmp-onnx-validation/`. The dataset manifest SHA256 is
`1cef6c7d9133d889b6c8eff0ffef29b1db5d6653ab4adf0c005739af8c2921c6`.
This documents an exploratory experiment; those local artifacts are not a shared
continuous-evaluation service or a portable deployment harness.

The portable `dev.benchmarks.onnx_inference` runner now retains graph/external
weight/input hashes, named outputs, raw timings, runtime configuration and actual
operation/provider execution profiles. See the
[target-machine commands](../dev/benchmarks/README.md#portable-onnx-inference-measurements).
A local batch-eight check of the same trained flat graphs measured 181.24 ms float
and 90.57 ms Percentile INT8, with 170 floating Conv versus 170 QLinearConv and two
QGemm operations. The report is retained under ignored `tmp-onnx-portable-flat/`.
This verifies the shared runner on a real model, not repeatability across machines
or a new quality comparison. Regression tests cover external weight provenance,
retained input-contract failures, unavailable providers and an advertised GPU
provider whose graph actually executes entirely on CPU.

### Native INT8 training checkpoint export

The native CUDA QT checkpoints from `tmp-normalization-blair/` now export their
captured integer forward through the generic ONNX API with
`reference_device="cuda:0"`. The private export copy freezes parameters so
PyTorch's tensor-subclass decomposition does not try to assign gradients to
integer storage. The training model and optimizer parameters remain untouched.
The custom scaled INT8 product lowers to MatMulInteger with INT32 accumulation,
retaining the existing row quantizer and floating scales. There is no calibration
set or conversion to a floating-weight classifier.

An initial trained flat export failed the unchanged parity gate: maximum score
error was 0.01264 on the first real image with TF32 allowed. Disabling TF32 in the
CUDA reference removed that failure. The exporter now scopes full-FP32 reference
execution and records this precision choice; it restores caller settings. The
failure is retained in `tmp-native-int8-onnx/export.log`, and the successful
explicit-precision diagnostic in `export-no-tf32.log`.

Both final trained exports passed rtol=1e-4/atol=1e-5 at batches 1, 2, 4 and 8,
using the same eight preprocessed validation images as the preceding portable
runtime check. The maximum observed absolute score difference was 1.72e-5 for
flat and 1.29e-5 for hierarchical fine/parent. A separate runtime profile confirms
two MatMulInteger and 170 floating Conv operations for each model on the local
ONNX Runtime CPU provider. These are native training head products, not the
170-quantized-convolution Percentile recipe from floating checkpoints.

Seven focused tests passed with intentional CUDA access: explicit-device failure,
normalized symmetric flat/hierarchical heads on tiny and actual EfficientNetV2-S
backbones, active-class filtering, dynamic batches, caller-state/TF32 restoration,
restricted checkpoint CLI loading, and zero/tiny-input and zero/negative-scale
numerical cases. Exported bundles, source checkpoint hashes, parity manifests and
runtime profiles are retained under ignored `tmp-native-int8-onnx/verified/`.
The documented export command is in the [ONNX guide](onnx.md#native-int8-training-checkpoints).
The full-validation follow-up below measures `mini_metrics` quality and numerical
limits, including AMP/TF32 comparisons. Target-provider execution and speed,
confidence-threshold equivalence and million-class export capacity remain open.
No new test-set evaluation was used in this work.


### Native ONNX full-validation quality and numerical limits

The native QT checkpoints and ONNX bundles above were compared on all 912 Blair
validation images in batches of eight. Checkpoint and graph/external-weight hashes
were verified, class mappings were checked against the dataset manifest, and
all modes used the same preprocessed inputs. This is inference comparison of each
fixed checkpoint, not a new training comparison or a test-set evaluation.

ONNX Runtime CPU and the full-FP32 CUDA reference made **identical top-1
predictions for every image**, at both hierarchical levels. Their Macro-F1,
Macro-Recall, Macro-Precision, Coverage and Theil's U therefore matched exactly.
Values below are proportions from `mini_metrics` at clean checkout revision
`70cc69adc05362863439277048e06386c1f885e1`, using the preceding full-coverage metric
call with no threshold optimization, filtering or resampling.

| Output / execution | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat / ONNX and CUDA FP32 | 0.769112 | 0.758984 | 0.796193 | 1.000000 | 0.810716 |
| Flat / CUDA AMP FP16 | 0.768593 | 0.758712 | 0.795913 | 1.000000 | 0.810290 |
| Hierarchical fine / ONNX and CUDA FP32 | 0.730698 | 0.720275 | 0.799803 | 1.000000 | 0.787638 |
| Hierarchical fine / CUDA AMP FP16 | 0.726950 | 0.716638 | 0.795216 | 1.000000 | 0.786153 |
| Hierarchical parent / ONNX and CUDA FP32 | 0.867051 | 0.844461 | 0.913180 | 1.000000 | 0.857871 |
| Hierarchical parent / CUDA AMP FP16 | 0.867051 | 0.844461 | 0.913180 | 1.000000 | 0.857871 |

FP32/ONNX accuracy was 83.55% flat and 81.47% / 91.45% hierarchical fine / parent.
AMP changed two flat and two hierarchical fine predictions; parent predictions
were unchanged. AMP fine accuracy was 81.36%; flat accuracy remained 83.55%
despite its two changed decisions. A third mode allowed cuDNN TF32 while keeping
floating matmul TF32 disabled and autocast off; it produced the same decisions
and metrics as the full-FP32 reference in this batch-eight run. Allowing TF32 does
not force a particular cuDNN kernel and is not evidence of TF32 execution. This
result does not establish parity for other batch sizes, devices or kernel choices.

The full split **did not pass universal score parity** at the export defaults
rtol=1e-4/atol=1e-5, despite identical top-1 decisions:

| Output | Images with any score outside tolerance | Score elements outside tolerance | Maximum absolute score error |
| --- | ---: | ---: | ---: |
| Flat | 6 / 912 | 142 | 0.022024 |
| Hierarchical fine | 16 / 912 | 372 | 0.012286 |
| Hierarchical parent | 17 / 912 | 222 | 0.010367 |

The earlier eight-image export check remains valid for its tested inputs, but
must not be read as a guarantee for unseen inputs. Supplying these affected
images to strict export verification would fail; the tolerance was not relaxed.
Coverage is 100% by construction here. Confidence differences can still matter
for abstention, threshold calibration or downstream consumers even when argmax
is unchanged. No production quality acceptance criterion has been established.

An affected-batch diagnostic exposed hidden inputs and actual integer activation
codes without changing weights. At validation index 164, the maximum hidden-input
CPU/CUDA difference was 1.07e-6. One of the hidden Linear input codes differed,
then two output Linear input codes differed; the maximum score difference was
0.009955. The other seven images in that batch had identical integer input codes.
This localizes amplification to quantization boundaries following small floating
input differences. It does not justify changing the trained quantizer or claiming
that all outliers have the same cause.

Exact scores, prediction CSVs, package/source provenance, per-level differences,
metric JSON and the diagnostic are retained under ignored
`tmp-native-int8-validation/`. These are local x86 CPU / laptop CUDA measurements,
not target GPU or ARM verification, and no latency claim is derived from this
quality run. Confidence-thresholded evaluation, repeated-seed training quality,
large-vocabulary quality and the intended target-machine runs remain open.

### Long-contraction INT8 accumulator correctness

Reviewing the million-class training envelope exposed a correctness issue before
full-model capacity testing: an INT8 input-gradient product contracts over the
number of output classes. INT32 accumulation is not safe for arbitrary signed
INT8 values once the contraction exceeds 131,071 elements. Checking only finite
losses or gradients cannot detect saturation.

A bounded local CUDA probe with contraction 150,000 and every operand equal to
127 returned 2,147,483,648 instead of the expected FP32 value 2,419,350,016. The
result was finite. The exact integer product is 2,419,350,000 before FP32 rounding.
This is an adversarial arithmetic regression, not a claim that every large-class
training batch previously saturated.

The backend now keeps the existing tuned kernel for contractions within the
INT32-safe bound. Longer products periodically transfer bounded INT32 partial
sums into INT64 storage inside the kernel, then convert the total to floating
point and apply row/column scales once. This avoids both integer saturation and
the loss of small residuals when large opposite-signed partial sums are converted
to float before cancellation. No floating weight master or floating matrix
multiplication is introduced. The new long-contraction kernel has a fixed tile;
its throughput on the intended accelerators still needs measurement.

ONNX lowering likewise combines bounded MatMulInteger results in INT64 before
scaling. Its conservative chunk limit also bounds raw unsigned-activation times
signed-weight products before zero-point compensation. Ordinary 1280-wide
EfficientNetV2 forward head products retain one MatMulInteger per Linear; the
large-class risk primarily concerns the training input gradient.

Regression coverage includes signed extrema around the 131,071/131,072 boundary,
150k and 1M contractions, strided weights, per-row and scalar/per-column scales,
large-sum cancellation, compiled execution and a 150k-input ONNX Linear. A separate
one-million-output, eight-input Linear with unit weights/inputs verifies both the
input gradient (1,000,000 per element) and weight gradient (2 per element for a
two-sample sum loss). This is a bounded gradient correctness check, **not** a
million-class EfficientNetV2 training, optimizer-memory or deployment benchmark.
Full-model initialization and training capacity remain open.

### Million-class normalized-head initialization memory

The normalized spherical initializer still retained several class-by-width
intermediates after the earlier Gram-matrix fix. A one-million-class, 1280-wide
symmetric normalized `Classifier` exposed this independently of training.
An uncapped WSL run reached 25,619,099,648 allocated bytes (23.86 GiB) on the
16 GiB laptop GPU and was interrupted, rather than treating driver-managed
allocations beyond dedicated memory as a successful capacity result. Per-process
GPU memory was unavailable from `nvidia-smi` in this environment.

The repeat explicitly capped the PyTorch allocator at 90% of device capacity
(about 14.4 GiB). The original implementation failed while requesting another
4.77 GiB temporary, after reaching 15,383,099,904 allocated bytes. The OOM message's
non-PyTorch usage field was nonsensical on this WSL build; the report retains it,
but conclusions use the configured allocator limit and PyTorch allocation counters.

The update now reuses the gradient buffer for subtraction and scaling, preserving
the original arithmetic order, and releases gradient/projection buffers before
the next iteration. This removes additional full-size result buffers and prevents
the previous iteration's intermediates from overlapping the next one. The same
one-million-class head then completed all 100 initialization iterations under the
same cap, with peak allocated/reserved bytes 15,383,099,904 / 15,407,775,744.
The uncapped observation was interrupted; it is not a completed baseline timing.
The cap governs the PyTorch allocator, not independent driver-residency telemetry.

Exact CPU and CUDA tests compare the old and new updates for tall, wide and square
weights over 100 iterations. A CUDA regression also bounds temporary allocations
on a 10k-by-1280 matrix across several iterations, so retaining old gradient and
projection buffers would fail. Initializer parameters, iteration count, seeded
arithmetic and classifier behavior are preserved.

Reports and the capped probe are retained under ignored `tmp-million-head/`.
This establishes normalized-head construction, not full EfficientNetV2 training,
optimizer-state capacity, checkpoint/reload, large-vocabulary quality or target
hardware performance. Those phases require their own evidence.

A subsequent full EfficientNetV2-S flat-model probe used a 92% allocator cap
(about 14.72 GiB), leaving room for the backbone. Construction completed with
peak allocated/reserved bytes 15,471,561,728 / 15,485,370,368. It then failed during
`prepare_quantized_training`, when the existing rowwise quantizer requested a
full-size rounding buffer. No optimizer step ran, so the declared momentum-free
SGD configuration is not a successful training result. The retained report is
`tmp-million-head/full-flat/report.json`. Bounded quantization preparation is the
next memory bottleneck; bypassing the cap or changing initialization iterations
would not resolve it.

Validation for the initializer change passed 381 CPU-default tests with 139 skips
and the existing EMA expected failure. Seven focused CPU/CUDA parity and
allocation cases passed with intentional GPU access. Static checks passed.

### Bounded preparation and laptop-sized capacity checks

Initial native INT8 conversion now applies the existing deterministic rowwise
quantizer in chunks, avoiding full-matrix rounding/clipping temporaries. The
source matrix and final INT8 storage remain resident during conversion; other
preparation validation allocations are unchanged. Exact regression checks cover
FP32, FP16 and BF16 on CPU and CUDA, including transposed weights, zero/tiny rows,
source preservation, RNG preservation and output layout. A CUDA check using a
10k-by-1280 matrix bounds additional conversion allocation below 96 MiB, which the
previous whole-matrix conversion exceeds.

Following the local scale limit, the full-model repeat uses **100,000 classes**,
not one million. Both randomly initialized EfficientNetV2-S configurations use a
symmetric hidden layer, normalized output head, seed 42, batch two of synthetic
128-by-128 RGB images, FP16 AMP (initial scale 128), gradient clipping at 5, and
MuonAuxAdamW (learning rate .01, weight decay zero). The hierarchical case groups
100 consecutive leaves per parent. Each run performs three successful optimizer
updates and verifies finite gradients and changed INT8 codes in the two target
rows. Forward/zero-grad/backward ordering matches the trainer.

The same RTX 3080 Ti Laptop GPU runs under a 92% PyTorch allocator cap. Peaks below
are allocated MiB for each phase, including live model/state storage, rather than
additional scratch space or independently measured physical VRAM residency.

| Head | Construction | Preparation | Training update peak | Changed target codes |
| --- | ---: | ---: | ---: | ---: |
| flat | 1563.6 | 1436.6 | 2899.0 | 2535 |
| hierarchical | 1564.3 | 1437.4 | 2899.8 | 2538 |

Both runs passed. These synthetic checks establish construction, preparation and
updates at this scale, not generalization, throughput improvements, checkpoint
capacity or target-hardware performance. No new million-class full-model run was
attempted; that remains work for a larger machine. The earlier million-class
preparation failure remains a recorded failure, not a subsequently verified pass.
Local probe and phase reports are retained under ignored `tmp-million-head/` as
`full_model_100k.py`, `full-flat-100k/report.json` and
`full-hierarchical-100k/report.json`.

Validation passed static checks and 387 CPU-default tests (146 skips and the
existing EMA expected failure), plus 13 focused CPU/CUDA preparation cases and
91 CUDA-enabled quantized-training/model regressions.


### ONNX CUDA provider placement

A local follow-up on 2026-09-09 used an isolated ONNX Runtime GPU 1.29.0 wheel,
Python 3.13.7, ONNX 1.22.0 and NumPy 2.4.6 on the RTX 3080 Ti Laptop GPU.
The wheel links CUDA 13 libraries; the process used the existing CUDA 13/cuDNN 9
library directories without replacing the working PyTorch or CPU ORT packages.
`use_tf32=0` was explicit. Runtime dependencies must match the actual installed
wheel; see the [ORT CUDA provider guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

The portable runner profiled the retained EfficientNetV2-S exports using the same
eight preprocessed Blair validation images, batch eight, size 128. **Selecting
CUDA did not produce integer GPU execution for either quantized recipe.**

| Export | Main CUDA operations | Main CPU operations | Profiled host-to-device / device-to-host copies |
| --- | --- | --- | ---: |
| Floating flat | 170 Conv, 2 Gemm | Auxiliary operations | 1 / 1 |
| Native QT flat | 170 Conv | 2 MatMulInteger | 3 / 3 |
| Native QT hierarchical | 170 Conv | 2 MatMulInteger | 3 / 3 |
| Calibrated unsigned Percentile flat QDQ | 170 floating Conv, 2 floating Gemm | 171 DequantizeLinear | 172 / 1 |

The QDQ CUDA run additionally executed 340 QuantizeLinear and 650 DequantizeLinear
operations on GPU. This differs from its earlier CPU run, which fused integer
QLinearConv/QGemm operations. These counts describe the observed optimized runtime
graph, not a universal statement about every CUDA/TensorRT version or quantized
representation. The native head's CPU fallback is particularly relevant to the
large-class deployment objective, even though this quality model has only 25 leaves.
Single-process timings are retained in the reports; these placement probes do not
establish speedups on the intended target machines.

An optional `--require-provider-op` check now requires each named runtime operation
type to occur and execute entirely on the requested provider. Requiring Conv and
MatMulInteger rejected the native CUDA run with its two CPU MatMulInteger nodes,
retaining a failed report and profile before timing. Missing/fused-away operation
types also fail; inspect actual profiles when selecting requirements. The default
runner still permits partial CPU execution and records it.

On the eight images, float and both native heads passed CPU/CUDA score comparison
at rtol=1e-4, atol=1e-5 (maximum errors 1.24e-5, 1.34e-5 and 8.59e-6 respectively).
The calibrated flat QDQ export failed: maximum score error 0.45745 and one top-1
change versus its own CPU output. That recipe needs a full CUDA-specific quality
study; its CPU results cannot serve as GPU quality evidence.

The native flat/hierarchical exports were then evaluated on all **912** Blair
validation images, in manifest order, batch eight, against retained matched CPU
ORT and full-FP32 PyTorch references. Manifest/checkpoint hashes and class mappings
were checked; the baseline score file hashes are retained. All predictions stayed
the same. `mini_metrics` revision `70cc69adc05362863439277048e06386c1f885e1`, with
explicit MacroF1 selection criterion and the earlier threshold-zero, no-abstention
policy, gave exactly the same five metrics for all three execution paths:

| Output | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.769112 | 0.758984 | 0.796193 | 1.000000 | 0.810716 |
| Hierarchical leaf | 0.730698 | 0.720275 | 0.799803 | 1.000000 | 0.787638 |
| Hierarchical parent | 0.867051 | 0.844461 | 0.913180 | 1.000000 | 0.857871 |

Strict score parity remains false: versus full-FP32 PyTorch, 6 flat, 18 leaf and
19 parent images exceed tolerance, with maximum errors 0.014928, 0.012346 and
0.010621. Confidence-threshold equivalence and production acceptance remain open.
The initial probe completed flat evaluation then hit a cleanup NameError; after
fixing that probe, hierarchical evaluation completed while preserving the flat
result. No training or performance measurements were inferred from that retry.

Profiles, hashes and output arrays are retained under ignored
`tmp-onnx-cuda-placement-flat/`, `tmp-onnx-cuda-placement-recipes/`,
`tmp-onnx-cuda-placement-cpu-reference/` and `tmp-onnx-cuda-required-int8/`.
Full-validation scores, metric CSVs and reports are under
`tmp-native-int8-cuda-validation/`; its probe is
`tmp-native-int8-validation/run_cuda.py`.

Next, verify an integer GPU execution path with an appropriate provider/kernel
and repeat the quality protocol before making a GPU quantization recommendation.
The native dynamic quantizer and calibrated QDQ recipe have distinct numerical
contracts. HPC training performance, intended desktop/Spark GPU performance and
ARM CPU inference still require their actual hardware; laptop execution does not
close those requirements.

The placement-check increment passed static checks and 390 CPU-default tests,
with 146 skips and the existing EMA expected failure. The real CUDA negative
probe also rejected native-head CPU fallback and retained its diagnostic report.


### TensorRT INT8 calibration candidate

The next local experiment used isolated TensorRT 10.16.1.11 CUDA 13 packages with
ONNX Runtime GPU 1.29.0 and the same RTX 3080 Ti Laptop GPU. The working `.venv`
was not changed. TensorRT is the documented
[ORT GPU quantization path](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html),
but its [standard quantizer](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Quantize.html)
requires constant scales and zero zero-points. Native QT uses dynamic row scales.

Direct parsing rejected the native export at an unsigned intermediate Cast. A
small signed MatMulInteger probe was also rejected as unsupported, so changing
only activation signedness does not establish a native TensorRT export path.
Those failures are retained, not replaced by a calibrated graph with a different
numerical contract.

The separate calibration candidate uses signed symmetric INT8 activation/weight
QDQ, per-channel weights and **floating biases**, applied to Conv/Gemm/MatMul.
Its symmetric ranges derive from the previously recorded asymmetric Percentile
99.9 ranges on 128 training images; it is not a newly fitted symmetric histogram.
Default INT32 bias dequantization was rejected by TensorRT. Disabling bias
quantization made the flat graph parse directly; registering TensorRT's standard
plugins also enabled the hierarchical scatter reductions.

ONNX Runtime's default graph rewrites reintroduced INT32 bias dequantization,
leaving 171 DequantizeLinear operations on CPU and 171 host-to-device copy nodes
around a TensorRT partition. Repeating the same flat graph with graph optimization
disabled produced **one TensorRT partition and no profiled CPU operations**.
The runner now exposes `--optimization disable|basic|extended|all`, records it,
and applies it to both profiling and timing sessions. The default remains `all`.
This does not disable TensorRT engine optimization. The initial provider probe
also exposed a configuration error: this build requires True/False values for
TensorRT boolean options, rather than the string "1"; its failed report is retained.

Directly built engines used a 1 GiB workspace limit, builder optimization level 1,
TF32 disabled, detailed layer inspection, and batch profiles 1–8 at image size 128.
Both normalized symmetric EfficientNetV2-S heads built and executed. Inspection
shows all **170 convolutions with INT8 inputs and weights**, and **both head GEMMs
with INT8 tactics**. Floating bias and auxiliary operations remain. The
hierarchical engine uses three standard scatter plugin layers. These are actual
integer GPU computation checks, not conclusions from QDQ nodes alone. The flat
ORT TensorRT output matched the direct engine exactly on the eight fixed images.
That is not a full ORT-versus-direct-engine equivalence claim.

The direct engines were evaluated on all **912** Blair validation images in
manifest order, batch eight, alongside full-FP32 PyTorch and CPU ORT executions of
the same signed QDQ graphs. Checkpoint/export hashes and class order were checked;
graph/external-weight/calibration/engine hashes are retained. Metrics use local
`mini_metrics` revision `70cc69adc05362863439277048e06386c1f885e1`, explicit MacroF1
selection criterion, threshold zero and no abstention or threshold tuning.

| Output / execution | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat / Float | 0.762073 | 0.754415 | 0.789570 | 1.000000 | 0.814794 |
| Flat / TensorRT INT8 | 0.755863 | 0.744234 | 0.790002 | 1.000000 | 0.804222 |
| Hierarchical leaf / Float | 0.709222 | 0.695729 | 0.798771 | 1.000000 | 0.782579 |
| Hierarchical leaf / TensorRT INT8 | 0.710140 | 0.687581 | 0.775226 | 1.000000 | 0.769531 |
| Hierarchical parent / Float | 0.859365 | 0.827595 | 0.920382 | 1.000000 | 0.851394 |
| Hierarchical parent / TensorRT INT8 | 0.847096 | 0.805315 | 0.921725 | 1.000000 | 0.828815 |

Relative to float, Macro-F1 changes are **−0.62 percentage points flat, +0.09 leaf,
and −1.23 parent**. Recall and Theil's U decrease for all three outputs; the small
leaf F1 increase is not evidence of generally improved quality. TensorRT versus
CPU QDQ changes 13 flat, 10 leaf and 7 parent predictions. Maximum score differences
are 0.46753, 0.52708 and 0.52582, and strict parity fails. This reinforces the need
for provider-specific quality evaluation. No production degradation tolerance has
been fixed. The user accepts degradation of a few percentage points when paired
with a significant inference speed/cost or memory improvement. These quality
results alone do not establish that joint trade-off; matched performance evidence
against a practical floating deployment baseline is required.

This milestone establishes a **separate calibrated INT8 GPU inference candidate**
for both representative heads, not native QT export equivalence. It does not
establish target-machine performance, optimal TensorRT build settings, training
benefits, confidence-threshold parity or large-class deployment capacity. The
probes used a default CUDA stream and ran alongside CPU correctness checks; their
recorded timings are not benchmark evidence. Production measurements require
fresh, uncontended runs with the final stream, build and cache configuration.

Ignored `tmp-trt-probe/` retains parser failures, the candidate manifest, graphs,
ORT reports, engines, detailed `direct-{flat,hierarchical}/layers.json`, the
`build_inspect.py` and `full_validation.py` probes, and the full-validation
scores/metric CSVs/report. Rebuild engines on each target device. The reproducible
runner options and recipe parameters are in the
[developer guide](../dev/benchmarks/README.md#tensorrt-calibration-candidate).

Static checks and 393 CPU-default tests passed, with 146 skips and the known EMA
expected failure. Runtime regressions verify that the selected ORT optimization
level changes the profiled graph while preserving fixture outputs.


### TensorRT INT8 versus FP16: initial paired trade-off

The user's acceptance criterion is conditional: a few percentage points of metric
loss are tolerable with a substantial inference speed/cost or memory benefit.
The comparator therefore includes a TensorRT FP16 deployment baseline, not just
the full-FP32 validation reference. Both heads were built from the same floating
checkpoints and preprocessing contracts. FP16 was enabled for the baseline;
TF32 remained disabled. Workspace (1 GiB), builder optimization level (1), spatial
size (128) and batch profiles (1–8) match the INT8 candidate. These build settings
are a controlled initial comparison, not a tuning result.

FP16 was checked on all 912 Blair validation images using the same five
`mini_metrics` metrics and fixed policy:

| FP16 output | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.759822 | 0.751879 | 0.787498 | 1.000000 | 0.813357 |
| Hierarchical leaf | 0.709222 | 0.695729 | 0.798771 | 1.000000 | 0.782579 |
| Hierarchical parent | 0.859365 | 0.827595 | 0.920382 | 1.000000 | 0.851394 |

FP16 changed three flat predictions relative to FP32 and no hierarchical
predictions. Strict score parity fails for both heads. Compared with this FP16
baseline, INT8 Macro-F1 changes by approximately −0.40 percentage points flat,
+0.09 leaf and −1.23 parent. These remain single-checkpoint quality measurements.

Timing then used three fresh processes per head with no concurrent validation or
benchmark jobs. Each process held both engines, used a non-default CUDA stream,
10 warmups and 31 measured repetitions per model/batch, alternated FP16/INT8 order
within repetitions, and reversed ordering in the middle process. Preallocated
CPU/GPU input and output buffers were reused. Host timing includes copies and
synchronized engine execution, excluding decoding, preprocessing, allocation,
engine loading/building and shape changes.

The ranges below are the three process medians in milliseconds. Ratios are
computed **within each paired process**, then summarized by their median; lower
than one would favor INT8. Variation is retained rather than selecting one run.

| Head | Batch | FP16 host ms range | INT8 host ms range | Median paired INT8 / FP16 |
| --- | ---: | ---: | ---: | ---: |
| flat | 1 | 3.444–3.887 | 4.288–4.384 | 1.233 |
| flat | 8 | 3.371–4.528 | 4.230–5.624 | 1.242 |
| hierarchical | 1 | 3.440–4.008 | 3.899–5.207 | 1.170 |
| hierarchical | 8 | 3.609–3.823 | 4.134–4.559 | 1.152 |

There is **no observed host-latency improvement** in these local small-batch
probes. CUDA-event durations frequently exceeded their enclosing synchronized
host durations; those counters are retained for diagnosis but excluded from
GPU-only timing conclusions. No cause has been established for that discrepancy.
The host measurements are limited observations from this WSL laptop and build
configuration, not evidence about A100/B300, Spark/desktop GPU, or ARM performance.

Serialized engines shrink from 46,089,148 to 26,406,668 bytes (flat) and 46,177,388
to 26,519,700 bytes (hierarchical), approximately **43% smaller**. TensorRT's
reported execution-context memory for the profile changes from 5,669,888 to
5,586,944 bytes for both heads, only **1.46% lower**. These are engine-file size
and a runtime-reported context requirement, not total GPU residency or peak
process memory. They do not establish a substantial total-runtime-memory benefit.

Thus quality is in the potentially tolerable range, but the desired speed or
runtime-memory trade-off is **not yet demonstrated** against FP16. The storage
benefit is real. Larger batches and large-class heads need separate paired probes,
and the requested target machines remain necessary for deployment conclusions.
The tested real-data heads have 25 leaves; they do not model 100k-class head costs.

Ignored artifacts: `tmp-trt-probe/direct-{flat,hierarchical}-fp16/`,
`full-validation-fp16/`, `paired-{flat,hierarchical}-{1,2,3}/`,
`paired-summary.json`, and the `build_fp16.py`, `full_validation_fp16.py`, and
`paired_inference.py` probes. Reports retain engine/source hashes and all timing
samples. This follow-up changes documentation only; all six benchmark processes
and both full-validation cases completed with finite outputs.

### Native INT8 checkpoint to calibrated TensorRT deployment

The explicit `mt_export --materialize-int8-training` path connects a trained
native INT8 checkpoint to floating export and subsequent static calibration.
It does not silently substitute the native export: the source checkpoint hash,
removed dynamic activation quantization and conversion recipe are recorded in
the manifest. See the [export contract](onnx.md#explicit-materialization-for-deployment-calibration).

This study uses the actual native checkpoints behind the verified native ONNX
baselines: `tmp-normalization-blair/{flat,hierarchical}/training/weights/last.pt`.
Their hashes were checked against `tmp-native-int8-onnx/verified/{head}/manifest.json`,
and the source files remained unchanged. These are different training runs from
the initial floating-checkpoint TensorRT study above, so its engines and metrics
are not reused as baselines here.

Both EfficientNetV2-S heads are normalized with symmetric hidden layers. The new
exports passed floating ONNX verification on CPU. The maintained preparation
command reproduced all retained NPZ hashes for the 128 training calibration
images and 912 held-out validation images. Percentile 99.9 calibration used the
same signed symmetric INT8 activation/per-channel weight, floating-bias recipe,
with asymmetric histograms and one CPU thread. Dynamic activation row quantization
from native training is absent from this new artifact.

TensorRT 10.16.1.11 on the RTX 3080 Ti Laptop built both INT8 engines and matched
FP16 baselines from the same materialized models. Settings were profile 1–8,
128px inputs, builder optimization 1, workspace 1 GiB, and TF32 disabled. Detailed
inspection verified **170 INT8 convolutions and two INT8 head GEMMs in each INT8
engine**, including INT8 inputs/weights or integer GEMM tactics. This establishes
integer GPU execution for the converted artifacts, not the native dynamic graph.

All four stages were evaluated on all 912 validation images, using the maintained
collector and mini_metrics comparison with threshold zero and no tuning. Native
means the previously verified native integer ONNX graph on CPU; materialized
FP32 means ONNX CPU execution before recalibration. FP16 and INT8 mean direct
TensorRT execution. The metrics are:

| Head / stage | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat / native | 0.769112 | 0.758984 | 0.796193 | 1.000000 | 0.810716 |
| Flat / materialized FP32 | 0.770778 | 0.762868 | 0.796316 | 1.000000 | 0.810397 |
| Flat / TensorRT FP16 | 0.772581 | 0.765220 | 0.797016 | 1.000000 | 0.811644 |
| Flat / TensorRT INT8 | 0.769551 | 0.762561 | 0.789897 | 1.000000 | 0.805218 |
| Hierarchical leaf / native | 0.730698 | 0.720275 | 0.799803 | 1.000000 | 0.787638 |
| Hierarchical leaf / materialized FP32 | 0.735466 | 0.725632 | 0.801459 | 1.000000 | 0.791476 |
| Hierarchical leaf / TensorRT FP16 | 0.732321 | 0.722396 | 0.800159 | 1.000000 | 0.789964 |
| Hierarchical leaf / TensorRT INT8 | 0.736413 | 0.730936 | 0.777745 | 1.000000 | 0.791877 |
| Hierarchical parent / native | 0.867051 | 0.844461 | 0.913180 | 1.000000 | 0.857871 |
| Hierarchical parent / materialized FP32 | 0.870255 | 0.847898 | 0.915910 | 1.000000 | 0.859540 |
| Hierarchical parent / TensorRT FP16 | 0.870844 | 0.848284 | 0.916666 | 1.000000 | 0.862149 |
| Hierarchical parent / TensorRT INT8 | 0.869788 | 0.849624 | 0.913274 | 1.000000 | 0.857367 |

Materialization changes 5 flat, 9 hierarchical leaf and 4 parent predictions
against native. The final INT8 candidate changes 65, 64 and 21 respectively.
Relative to native, INT8 F1 changes are +0.044, +0.572 and +0.274 percentage points,
while hierarchical leaf precision drops 2.206 points. Relative to matched FP16,
the corresponding F1 changes are −0.303, +0.409 and −0.106 points, with a 2.241-point
hierarchical leaf precision drop. Small F1 gains in a single-checkpoint comparison
are not evidence that quantization generally improves model quality.

INT8 engine sizes are 26,325,140 bytes (flat) and 26,492,548 bytes (hierarchical),
versus 46,455,028 and 46,154,172 for FP16. Reported context requirements are
5,586,944 versus 5,669,888 bytes. These are file/context observations, not total
runtime-memory savings. No new latency benchmark or target-hardware performance
claim is made; a worthwhile trade-off must still be verified against FP16 on
the intended workload and device.

The conversion regressions cover ordinary/normalized weights, negative/zero scales,
source immutability, masked flat/hierarchical checkpoint exports and explicit native
default behavior. A further shared-magnitude regression exposed a sign-placement
issue; signs now belong to the directions, while zero-scale rows use zero
magnitudes. Compatible shared magnitudes work, and incompatible tied roles/views
fail explicitly. The two real checkpoints have no nonpositive normalized direction
scales, so their materialized parameter values are unchanged by this correction.
All 12 focused materialization tests passed. `bash dev/check.sh all` passed static
checks and 470 tests, with 152 skips and one known EMA expected failure. Those
CPU regression checks supplement the real TensorRT execution above; they do not
establish target GPU or ARM correctness.

All artifacts are retained under `tmp-native-materialized/`: source-linked floating
exports, prepared inputs, calibrated QDQ graphs, detailed engine inspection,
per-batch scores, prediction CSVs and native/FP16 comparison reports. The experiment
provides a distinct, measured QT-checkpoint-to-INT8-GPU route. It does not establish
unchanged native quantization, distributed training, million-class capacity,
ARM execution or target-machine cost benefits.

### Paired inference quality orchestration

`dev.benchmarks.inference_pair` now composes baseline collection, candidate
collection and the five-metric comparison in sequential fresh processes. Separate
Python interpreters can supply the inference and metric dependencies. It retains
commands, logs, child report hashes, failure status and a Markdown summary, and
rejects existing output directories. See the
[command documentation](../dev/benchmarks/README.md#paired-inference-quality-pipeline).

Four full 912-image Blair comparisons replayed the retained artifacts above:
native integer ONNX versus materialized ONNX on CPU, and TensorRT FP16 versus
INT8, each for flat and hierarchical heads. Both CPU pairs and the flat TensorRT
pair reproduced prediction CSVs byte-for-byte. The hierarchical TensorRT baseline
had one confidence value differ by 4.28e-12; all other CSV fields and the candidate
CSV were unchanged. All four pairs reproduced every discrete prediction, metric
value and metric delta exactly. This is orchestration reproducibility evidence,
not a new quality improvement, timing result or target-hardware qualification.

Ignored outputs are `tmp-inference-pair-{flat,hierarchical}/` and
`tmp-inference-pair-trt-{flat,hierarchical}/`. The replays reuse existing models,
engines and prepared inputs and do not duplicate them or retain full score arrays.
The small hierarchical oracle also exercises the real subprocess pipeline;
failure and output-reuse regressions verify that incomplete runs cannot report
successful evaluation. Preparation, calibration, engine building, placement and
performance still need orchestration before this is a complete deployment pipeline.
Validation: `bash dev/check.sh all` passed static checks and 473 tests, with
152 skips and one known EMA expected failure. The focused collector/pipeline
suite passed 12 tests with one optional CUDA skip. The real TensorRT replays
were separate intentional GPU runs; ordinary CPU checks do not establish GPU support.

### Isolated Linux CPU memory and inference study

`dev.benchmarks.onnx_cpu_memory` measures one CPU model per fresh Linux interpreter,
with no profiling session or other model loaded. It records session load, first
inference, all warm timings, and resident-memory snapshots before/after runtime
import, input loading, session creation, first inference, warmup and measurement.
The [command guide](../dev/benchmarks/README.md#single-process-onnx-cpu-memory-probe)
defines the memory accounting and target handoff procedure.

A regression exposed inherited `ru_maxrss`: after a parent allocated 150 MiB,
its fresh child reported 167,700 KiB through `getrusage`, while `/proc/self/status`
reported a 13,404 KiB VmHWM for the child's new address space. The probe therefore
uses `smaps_rollup` RSS/PSS snapshots and approximate post-exec VmHWM, rather than
attributing a parent's earlier peak to model inference. It records memory before
ONNX graph provenance inspection and output serialization. Interpreter, runtime,
inputs/outputs and finite-value checks remain part of the measured process.

The paired quality runner separately evaluated the materialized floating ONNX
and signed calibrated QDQ models from the native-checkpoint study on all 912
Blair validation images, using ONNX Runtime CPU. These are CPU runtime results,
not reused TensorRT predictions. The five mini_metrics results are:

| Head / stage | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat / float | 0.770778 | 0.762868 | 0.796316 | 1.000000 | 0.810397 |
| Flat / INT8 | 0.764407 | 0.756174 | 0.786411 | 1.000000 | 0.802939 |
| Hierarchical leaf / float | 0.735466 | 0.725632 | 0.801459 | 1.000000 | 0.791476 |
| Hierarchical leaf / INT8 | 0.738374 | 0.732758 | 0.774078 | 1.000000 | 0.795192 |
| Hierarchical parent / float | 0.870255 | 0.847898 | 0.915910 | 1.000000 | 0.859540 |
| Hierarchical parent / INT8 | 0.869179 | 0.848718 | 0.914401 | 1.000000 | 0.858741 |

Flat F1 decreases 0.637 percentage points; hierarchical leaf F1 increases 0.291
points while precision decreases 2.738 points. Parent F1 decreases 0.108 points.
The candidate changes 67 flat, 55 leaf and 16 parent predictions. Calibration
remains training-only; validation thresholds are fixed at zero without tuning.
Local quality artifacts are `tmp-edge-quality-{flat,hierarchical}/`.

After all regression and quality processes finished, three fresh-process trials
per model used batch one, 128px inputs, one intra-op/inter-op thread, three warmup
runs and 31 measured runs. Float/INT8 order was reversed in the middle trial.
The machine was the local i7-12800H x86 WSL environment, ONNX Runtime 1.29; CPU
affinity remained 0–19, rather than a pinned core. Session construction and first
inference were recorded separately. No filesystem-cache eviction, thermal
control or ARM emulation was performed.

| Head / model | Warm median ms, trials 1/2/3 | Final RSS MiB, trials 1/2/3 | Approximate peak RSS MiB, trials 1/2/3 |
| --- | --- | --- | --- |
| Flat / float | 29.277 / 26.685 / 26.726 | 192.270 / 196.605 / 192.512 | 193.691 / 198.086 / 193.980 |
| Flat / INT8 | 42.963 / 42.599 / 44.455 | 104.992 / 106.230 / 105.867 | 104.812 / 106.117 / 105.668 |
| Hierarchical / float | 27.914 / 38.539 / 27.565 | 193.785 / 190.727 / 192.156 | 195.164 / 192.207 / 193.547 |
| Hierarchical / INT8 | 45.976 / 40.411 / 45.793 | 105.406 / 105.633 / 105.980 | 105.359 / 105.395 / 105.926 |

INT8 final resident memory was 44.6–46.0% lower across these pairs. Warm median
latency was higher in all pairs: candidate/baseline ratios 1.468/1.596/1.663 for
flat and 1.647/1.049/1.661 for hierarchical. These are separate-process median
ratios, not adjacent per-inference paired ratios. Reported swap was zero. The
approximate status high-water counter can be slightly below the more precise
smaps snapshot; retain both sources rather than treating them as identical
accounting. This demonstrates local process-memory savings, not a speed gain or
an ARM production result.

Separate post-measurement profiling found **63 QLinearConv and two QGemm**
operations, with **107 floating Conv** operations, on CPU for each candidate.
The same signed QDQ/floating-bias recipe executed 170 integer convolutions under
TensorRT; its CPU execution is hybrid. A CPU-specific calibration/fusion study
is therefore the next useful step before target ARM qualification, rather than
assuming that the TensorRT recipe is also the best edge recipe.

`tmp-edge-process-probe/` retains the trial driver, all raw timings/memory reports,
output arrays, batch-one input provenance and separate placement profiles. No
models or source datasets were duplicated. All 13 focused ONNX benchmark tests
passed; `bash dev/check.sh all` passed static checks and 476 tests, with 152 skips
and one known EMA expected failure. Hardware-specific validation, sustained-load
thermal behavior and end-to-end preprocessing costs remain unverified.

### CPU-specific activation and bias calibration

This follow-up isolates the previous CPU recipe's incomplete integer coverage.
It uses the same native-trained checkpoints, materialized floating exports,
128 training calibration images, percentile 99.9 histograms and 912 validation
images. Both new candidates reproduce the previous **parsed calibration ranges
exactly**; JSON file hashes differ because of key ordering. No validation data
or metric thresholds were used to select ranges.

1. Signed symmetric INT8 activations with INT32 biases changed only the bias
   option from the previous TensorRT-oriented recipe. It still executed 63
   QLinearConv, 107 floating Conv and two QGemm operations on CPU. Bias quantization
   alone did not resolve the coverage gap. Its full-data quality results are
   retained rather than discarded.
2. Unsigned asymmetric UINT8 activations with symmetric per-channel INT8 weights
   and INT32 biases used the maintained calibrator's CPU defaults. Both models
   executed **170 QLinearConv and two QGemm operations, with no floating Conv**,
   under the local CPU provider. This is a separate recipe from the TensorRT
   candidate, not a change to the package's training/export defaults.

The paired quality runner evaluated both candidates independently against the
same materialized float baseline. Candidate metric values are below; the float
values are in the preceding CPU study.

| Candidate / output | Macro-F1 | Macro-Recall | Macro-Precision | Coverage | Theil's U |
| --- | ---: | ---: | ---: | ---: | ---: |
| Signed, INT32 bias / flat | 0.764407 | 0.756174 | 0.786411 | 1.000000 | 0.802939 |
| Signed, INT32 bias / hierarchical leaf | 0.735748 | 0.730662 | 0.772329 | 1.000000 | 0.792383 |
| Signed, INT32 bias / hierarchical parent | 0.867726 | 0.845839 | 0.912249 | 1.000000 | 0.856400 |
| Unsigned, INT32 bias / flat | 0.764523 | 0.761055 | 0.783247 | 1.000000 | 0.805098 |
| Unsigned, INT32 bias / hierarchical leaf | 0.733602 | 0.727803 | 0.771896 | 1.000000 | 0.793045 |
| Unsigned, INT32 bias / hierarchical parent | 0.868766 | 0.848444 | 0.913857 | 1.000000 | 0.857750 |

For the unsigned candidate, Macro-F1 changes against float are −0.626, −0.186 and
−0.149 percentage points for flat, leaf and parent outputs. Hierarchical leaf
precision decreases 2.956 points, despite a small recall increase. Coverage remains
one. The unsigned candidate changes 66, 60 and 18 predictions respectively;
the signed/INT32-bias candidate changes 67, 59 and 17. These remain single-trained-
checkpoint comparisons, not a general quality-improvement claim.

After calibration, profiling and quality processes finished, three fresh-process
trials per model measured the unsigned candidate against newly run float
baselines. Settings match the preceding CPU study: batch one, 128px, one intra-op
and inter-op thread, three warmups and 31 timings, reversing recipe order in the
middle trial. Input hashes, settings and reported environments match within
every pair. The i7-12800H WSL x86 CPU affinity remained 0–19; no claim of controlled
thermals, disk-cold startup or ARM emulation is made.

| Head / model | Warm median ms, trials 1/2/3 | Final RSS MiB, trials 1/2/3 | Approximate peak RSS MiB, trials 1/2/3 |
| --- | --- | --- | --- |
| Flat / float | 23.484 / 25.838 / 24.330 | 196.812 / 191.426 / 191.430 | 198.230 / 192.855 / 192.848 |
| Flat / unsigned INT8 | 11.022 / 12.236 / 11.983 | 103.586 / 103.852 / 103.137 | 102.738 / 103.059 / 102.273 |
| Hierarchical / float | 22.611 / 27.651 / 22.634 | 198.129 / 191.621 / 191.625 | 199.609 / 193.039 / 194.797 |
| Hierarchical / unsigned INT8 | 12.703 / 12.605 / 12.049 | 103.414 / 103.242 / 104.441 | 102.551 / 102.453 / 103.645 |

Candidate/baseline median latency ratios are 0.469/0.474/0.493 for flat and
0.562/0.456/0.532 for hierarchical: **44–54% lower warm latency**, with **45–48%
lower final resident memory** across the six pairs. These are separate-process
median ratios, not adjacent timing pairs. Session load was usually slower for
INT8 (139–181 ms versus 109–150 ms for float), so this supports repeated inference
after startup rather than a universal end-to-end speed claim. Memory sources and
their different accuracy are as documented in the preceding study.

This establishes a useful local x86 quality/resource trade-off and a concrete
candidate for target-device validation. It does not establish ARM execution,
sustained thermal behavior, image/preprocessing throughput, large-class heads,
GPU inference speed or HPC training benefits. Do not extrapolate these percentages
to Raspberry Pi, Spark/RTX or A40/A100/B300 systems.

Reproduction uses the maintained [CPU recipe](../dev/benchmarks/README.md#cpu-specific-qdq-candidate-for-efficientnetv2),
paired quality runner and isolated memory probe. Retained ignored artifacts are
`tmp-cpu-bias-{flat,hierarchical}/`, `tmp-cpu-u8-{flat,hierarchical}/` and
`tmp-cpu-u8-trials/`, including calibration reports/ranges, placement profiles,
all prediction CSVs and five metrics, raw timing/memory reports and the trial
driver. Batch-one inputs and source provenance are reused from
`tmp-edge-process-probe/`. All four calibration and quality cases, four placement
checks, and twelve timing processes completed successfully. Static checks passed;
this increment changes documentation only, so the full runtime suite was not rerun.

### Composed CPU deployment validation

`dev.benchmarks.cpu_deployment` now runs the full held-out quality comparison,
baseline/candidate operation inspection and repeated fresh-process resource
trials in one command. It links graph/external-weight hashes across phases,
rejects changed timing inputs, checks paired environments/settings and retains
logs, child reports and a combined Markdown summary. Candidate operator
requirements are explicit, not hardcoded by architecture. See the
[command guide](../dev/benchmarks/README.md#composed-cpu-deployment-comparison).

Both real heads completed the command with the unsigned CPU candidates and the
same materialized float baselines, all 912 held-out Blair images, batch-one
resource inputs, one thread, three trial pairs, three warmups and 31 repetitions.
The full regression suite finished before these runs; the two complete model
comparisons ran sequentially. Predictions and all five metric values reproduced
the preceding CPU recipe study exactly for both baseline and candidate CSVs.
Each candidate again executed 170 QLinearConv and two QGemm operations with no
floating Conv. Every phase's artifact identity checks passed.

| Head | Candidate/baseline warm latency ratios, trials 1/2/3 | Final RSS ratios, trials 1/2/3 |
| --- | --- | --- |
| Flat | 0.489 / 0.474 / 0.588 | 0.540 / 0.518 / 0.536 |
| Hierarchical | 0.454 / 0.453 / 0.491 | 0.520 / 0.534 / 0.545 |

These fresh x86 runs replicate the local latency/memory benefit. They are
separate-process median comparisons, not adjacent inference-pair measurements,
and do not establish sustained thermal behavior or target ARM performance.
Raw warm timings, startup observations, approximate memory peaks, output arrays,
placement profiles and quality reports are retained in
`tmp-cpu-deployment-{flat,hierarchical}/`. Existing model and input bundles are
reused; they are not copied into these report directories.

Regression coverage exercises the real subprocess pipeline with a small
two-level oracle, alternating execution order, rejection of existing outputs,
missing required operations and changed timing inputs. `bash dev/check.sh all`
passed static checks and 479 tests, with 152 skips and one known EMA expected
failure. This composes evaluation of existing CPU deployment artifacts;
preparation/calibration, GPU orchestration, durable CI hosting and profile-specific
acceptance gates remain unfinished.

### Maintained 100k-class BF16 training comparison

`dev.benchmarks.large_head_training` turns the earlier capacity probe into a
maintained command with independent synthetic-input RNG, applied-update checks,
failure reports and separate full/frozen-backbone modes. It preserves the head's
intentionally frozen parameters. Valid unused normalized-head BatchNorm parameters
are reported explicitly; they do not imply a missing gradient in an active branch.
See the [command guide](../dev/benchmarks/README.md#large-head-training-capacity-command).

The local study ran 24 fresh GPU processes: normalized symmetric EfficientNetV2-S
flat/hierarchical heads, 100,000 leaf classes, full/frozen backbone, float/native
INT8 storage, and seeds 42/43/44. Both paths used BF16 autocast, FP32 gradients and
eager MuonAuxAdamW (learning rate 0.01, zero weight decay, norm clipping at 5).
The hierarchical taxonomy groups leaves in consecutive groups of 100. Batch size
was 32 at 128px, with three warmup and five measured updates on each fixed batch.
Floating parameters remain FP32; native INT8 quantizes the two head Linear weights
and saved Linear inputs. Convolutions, gradients and optimizer states remain floating.

The backbone was initialized without pretrained weights in every case. Frozen
mode is therefore a synthetic capacity diagnostic, not realistic pretrained
fine-tuning: it evaluates the backbone on every batch and trains the head, without
caching embeddings. Loss is leaf cross-entropy plus parent cross-entropy for the
hierarchical case. These are new inputs and BF16 profiles; earlier FP16 results
are not reused as baselines. Float/INT8 order reverses for seed 43.

All input hashes, settings (apart from quantization), parameter counts and reported
environments matched within each pair. All 72 warmup and 120 measured optimizer
updates were applied. The only unused trainable parameters in every run were the
normalized head's inactive BatchNorm weight and bias. No frozen parameters received
gradients. Measurements began after the full regression suite and GPU smoke checks
had exited, and all GPU cases ran sequentially on the RTX 3080 Ti Laptop with
PyTorch 2.12.0+cu130 and TorchAO 0.17.0.

| Head / mode | Float median update ms, seeds 42/43/44 | INT8 median update ms, seeds 42/43/44 | Float measured peak GiB | INT8 measured peak GiB |
| --- | --- | --- | ---: | ---: |
| Flat / full | 137.349 / 110.825 / 148.278 | 167.257 / 143.157 / 151.248 | 3.819 | 3.154 |
| Hierarchical / full | 132.841 / 145.857 / 152.000 | 132.494 / 120.835 / 134.396 | 3.820 | 3.154 |
| Flat / frozen | 68.440 / 78.138 / 103.929 | 88.448 / 63.521 / 80.553 | 2.800 | 3.093 |
| Hierarchical / frozen | 90.369 / 73.076 / 91.959 | 72.296 / 92.531 / 92.770 | 2.801 | 3.094 |

Measured-phase CUDA allocated peaks were identical across the three seeds for
each configuration. Native INT8 reduced the full-model peak by **17.4%**, but
**increased the frozen-backbone peak by 10.5%** in the original probe. This frozen
result was subsequently traced to a retained floating parameter in the probe;
see the correction below. Parameter storage decreased from
about 0.559 to 0.197 GiB in both modes; that storage reduction is not proof of a
runtime-memory reduction. The frozen-mode peak needs allocation profiling before
choosing an optimization; this comparison alone does not identify its cause.

INT8/float update-time ratios were 1.218/1.292/1.020 for flat/full,
0.997/0.828/0.884 for hierarchical/full, 1.292/0.813/0.775 for flat/frozen and
0.800/1.266/1.009 for hierarchical/frozen. These short local trials do not establish
a general speedup or stable target throughput. No device clock/thermal controls
or cold compiler-cache controls were imposed. Timings use synchronized host
boundaries around preprocessing, forward, backward, clipping and optimizer work;
scalar loss reporting follows the timed interval. Setup/warmup allocated peaks
and measured allocated/reserved peaks are retained separately. These are PyTorch
allocator measurements, not total device/process memory.

`tmp-large-head-bf16/` retains the paired driver, logs, reports and comparison JSON.
The probe saves no checkpoints or datasets. CPU float32 full/frozen diagnostics
cover both head types, while separate FP16 and 100k-class hierarchical BF16 native
INT8 CUDA smoke cases checked execution. `bash dev/check.sh all` passed static
checks and 482 tests, with 152 skips and one known EMA expected failure.
Actual HPC/desktop hardware, realistic pretrained fine-tuning, longer training,
convergence, loading, checkpoint/resume, scheduler and distributed behavior still
require the corresponding integrated profiles; this capacity probe does not
certify those parts of the goal.

### Correcting the frozen-head allocation comparison

Allocation tracing of the 100k-class BF16 probe placed the frozen-mode peak in
AdamW, which updates the final classification layer. A CUDA allocator snapshot
also identified a live 512,000,000-byte allocation from the original floating
head constructor after INT8 preparation. The probe's backbone-freezing loop
retained its last `parameter` local after preparation replaced that parameter.
The corrected probe releases that reference before preparation. This was a
measurement artifact, not a required floating master weight for native training.
GPU lifetime regressions check that replaced parameters have been released before
optimizer construction for both flat and hierarchical heads.

The corrected flat frozen run measured 2.616 GiB allocated peak versus 2.800 GiB
for float, a 6.6% reduction, instead of the original 3.093 GiB INT8 peak. The
hierarchical frozen pair measured 2.617 GiB INT8 versus 2.801 GiB float, also 6.6%
lower. Each uses
the same seed 42, 100k classes, batch 32, image size 128, three warmup and five
measured updates, normalized symmetric EfficientNetV2-S and BF16 settings. The
full-model comparison remained 3.819 GiB float versus 3.154 GiB INT8. Reports and
diagnostic allocation snapshots are retained under ignored `tmp-mixed-update/`.
CPU regression checks overlapped the corrected diagnostic runs, so these runs
support allocation conclusions only, not new timing claims or target certification.

A separate update-path improvement handles FP16/BF16 optimizer updates to FP32
INT8 weights without materializing a floating weight matrix. It preserves the
update dtype's multiplication rounding before FP32 addition. CUDA regressions
compare exact codes, scales and RNG consumption across repeated updates and
transposed inputs, and forbid dequantization during the fused update. This closes
a Muon fallback but did not change the measured full-model or frozen peak by
itself; it must not be credited with the probe correction's memory reduction.

Static checks passed. The CPU-default regression run passed 482 tests, with 156
skips and the known EMA expected failure. Separate intentional CUDA runs passed
the 11 focused update/version checks and both new parameter-lifetime checks.
The broader affected CUDA suite passed all 95 tests after isolating its existing
fresh-tuning regression in a subprocess with compiled caches disabled at startup.
Initially that test failed because no tuning callback ran; clearing only the
tuner cache or disabling graph caches late in a shared process was insufficient.
The isolated test retains its tuning, finite-gradient and optimizer assertions.

### Maintained image preparation reproduction

`dev.benchmarks.prepare_inputs` now generates both calibration and held-out NPZ
manifests from source images, the benchmark dataset inventory and export metadata.
It uses existing image decoding/resizing with zero workers and one CPU thread by
default. Class ordering, selected source levels, source hashes, declared split
separation, output bindings, tensor shape/dtype and explicit score semantics are
checked. A trusted preprocessing factory handles custom transforms; the default
uses existing architecture loaders without loading the trained classifier head.
See the [commands and scope](../dev/benchmarks/README.md#maintained-image-input-preparation).

The real reproduction used the retained EfficientNetV2-S flat/hierarchical export
manifests and Blair source inventory. Calibration selection used
`random.Random(42).sample(train_records, 128)`; validation retained all 912 records
in manifest order. Source image hashes were checked before and after preparation.

| Head / split | Images | Batches of eight | Retained NPZ file hashes reproduced |
| --- | ---: | ---: | --- |
| Flat / calibration train | 128 | 16 | Every batch |
| Flat / validation | 912 | 114 | Every batch |
| Hierarchical / calibration train | 128 | 16 | Every batch |
| Hierarchical / validation | 912 | 114 | Every batch |

Calibration filenames and selection order matched the historical calibration
records. Validation samples, class labels and level/output bindings matched the
maintained collector's previous input manifests exactly. Calibration IDs are now
canonical integer strings; that metadata change does not change the input tensors.
The new files are retained in `tmp-prepared-real/{flat,hierarchical}-{train,val}/`,
with reports and reproduction checks. These exact input bytes already passed the
preceding calibration and full-dataset inference replays, so those downstream
GPU/CPU runs were not repeated for this preparation-only milestone.

Ten focused regressions passed, including byte-reproducible preparation, a partial
final batch, class-order and source-hash failures, declared cross-split duplicates,
and preservation of caller RNG state. The default loader receives backbone
metadata without constructing a million-class trained head; the regression checks
that loader boundary without allocating such a head. This does not remove the
backbone construction cost of the current architecture getters or establish
reproduction of an unrecorded custom preprocessing transform. Continuous job
orchestration and target-hardware verification remain outstanding.

Static checks and the full CPU-default suite passed: 458 passed, 152 skipped and
the known EMA expected failure. No new GPU correctness or performance claim is
made by this input-preparation milestone.

### Maintained full-dataset inference reproduction

`dev.benchmarks.dataset_inference` now collects complete held-out prediction
tables with ONNX Runtime or TensorRT and produces the paired quality evaluator's
manifest directly. It validates the shared dataset identity contract before
execution, binds named outputs to ordered class lists and explicit score semantics,
checks every batch's identity/hash/shape, and optionally retains batch scores.
The CPU path imports neither PyTorch, TensorRT nor mini_metrics. See the
[input schema and composed commands](../dev/benchmarks/README.md#maintained-full-dataset-prediction-collection).

For the Blair replay, all 912 validation source images were checked against their
recorded dataset hashes. Inputs were regenerated with each retained floating
checkpoint's preprocessing at 128px, after checking checkpoint class mappings
against the dataset manifest. Each head produced 114 batches of eight; the
synthetic regression separately exercises a final partial batch and static
secondary inputs. Real inputs/manifests are retained in `tmp-heldout-inputs/`.

Both heads were then executed through the maintained collector using the retained
FP16 TensorRT engine, retained signed-QDQ INT8 TensorRT engine, and the maintained
calibration command's signed-QDQ ONNX model on CPU. All six runs evaluated all
912 images. They were correctness runs; CPU tests were active during part of the
work, so no timing or memory-benefit claim is made.

| Head/backend | Discrete predictions versus retained results | Maximum leaf score difference | Maximum parent score difference |
| --- | --- | ---: | ---: |
| Flat TensorRT FP16 | All identical | 0 | — |
| Flat TensorRT INT8 | All identical | 0 | — |
| Flat ONNX CPU INT8 | All identical | 0 | — |
| Hierarchical TensorRT FP16 | All identical | 0 | 1.19e−7 |
| Hierarchical TensorRT INT8 | All identical | 0 | 1.19e−7 |
| Hierarchical ONNX CPU INT8 | All identical | 0 | 0 |

The small TensorRT parent-score differences mean full bitwise score reproduction
is not established; they remain below the earlier absolute parity tolerance of
1e−5 and cause no argmax changes. Four generated comparison manifests (INT8 GPU
and CPU versus FP16 for each head) ran through the maintained mini_metrics
evaluator. Candidate metrics matched the retained reports within 3.33e−16.
Confidence is now computed with a float64 softmax for declared logits; CSV bytes
are therefore a separate contract from raw scores and discrete predictions.

Collection reports, scores, bundles, paired metric reports and reproduction
checks are retained under `tmp-heldout-collection/{flat,hierarchical}-{fp16,int8,cpu_int8}/`.
These results verify the maintained collection/evaluation connection on the local
x86 CPU and laptop GPU. They do not establish ARM runtime behavior, target GPU
performance, native QT checkpoint deployment or production acceptance. Source
image preprocessing/batch preparation for this replay still used a local script;
promoting that preparation and composing continuous jobs remain necessary.

Static checks and the full CPU-default suite passed: 447 passed, 152 skipped and
the known EMA expected failure. A subsequent focused run including the new CPU
dependency-isolation regression passed 23 tests with one GPU skip. All nine
focused tests present in the prepared TensorRT run passed, including real
multi-input/two-level execution at both batch sizes.

### Maintained paired mini_metrics reproduction

`dev.benchmarks.quality_compare` now validates prediction CSVs against an explicit
held-out sample/level/class manifest and evaluates Macro-F1, Macro-Recall,
Macro-Precision, Coverage and Theil's U. It preserves literal class names,
canonicalizes reordered rows, rejects missing/duplicate samples and changed labels,
and retains model/dataset provenance plus imported mini_metrics source hashes.
See the [input contract and commands](../dev/benchmarks/README.md#maintained-paired-quality-comparison).

The real-data replay compared the retained TensorRT INT8 predictions with the
retained TensorRT FP16 baseline for both heads across all 912 Blair validation
images. It reused existing inference CSVs; no new model execution, score-parity
check or timing measurement was performed. Fixed threshold zero, no abstention,
no threshold optimization, ordinary per-level metrics and explicit `MacroF1`
selection preserve the previous evaluation policy.

| Level | F1 delta | Recall delta | Precision delta | Coverage delta | Theil's U delta | Changed predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat leaf | −0.003960 | −0.007645 | +0.002504 | 0 | −0.009136 | 61 |
| Hierarchical leaf | +0.000918 | −0.008148 | −0.023545 | 0 | −0.013048 | 67 |
| Hierarchical parent | −0.012269 | −0.022280 | +0.001343 | 0 | −0.022578 | 25 |

Deltas are candidate minus baseline in the original metric units; an F1 delta
of −0.003960 is approximately −0.396 percentage points. Coverage remains one for
all rows under this complete known-label, no-abstention policy. These quality
changes must be considered alongside measured efficiency, not accepted alone.

An initial exact-equality replay failed only on last-bit Macro-F1 differences.
Inspection of the imported mini_metrics implementation showed unordered class-set
iteration followed by floating summation. The maximum difference from historical
results was 3.33e−16; every result was within the explicit 1e−12 reproduction
tolerance. No metric implementation or expected historical result was changed.
Two fresh processes with `PYTHONHASHSEED=0` reproduced all metrics and deltas
exactly. The runner records the hash-seed setting. Undefined metrics, such as
Theil's U with a single observation, are explicit JSON nulls with an accompanying
list, not invalid JSON NaNs or silently substituted zeros.

The initial replay is retained under `tmp-quality-compare/`; fixed-seed manifests,
canonical CSVs, package hashes and reports are under `tmp-quality-compare-fixed/`,
including both heads and their fresh-process repeats. The local sibling package
was selected explicitly through `PYTHONPATH`; the command itself assumes no local
checkout layout. All 14 focused tests passed with both the installed mini_metrics
release and the local checkout. Static checks and the full CPU-default suite
passed: 439 passed, 151 skipped and the known EMA expected failure.
Full-dataset inference collection and calibration
input preparation still need to be connected to the maintained evaluation commands.

### Maintained paired TensorRT timing reproduction

`dev.benchmarks.tensorrt_pair` now measures arbitrary existing engine pairs with
named preprocessed inputs, alternating adjacent execution order, raw host durations
and median paired candidate/baseline ratios. It supports pageable or pinned host
IO, records hashes and final outputs, and retains completed pairs on later errors.
It deliberately omits CUDA-event timing. See the
[commands and measurement scope](../dev/benchmarks/README.md#maintained-paired-tensorrt-timing-command).

The local validation reused the retained FP16 and signed-QDQ INT8 engines on the
RTX 3080 Ti Laptop, TensorRT 10.16.1.11 and PyTorch 2.12.0+cu130. Each head used
three fresh sequential processes, reversing execution order in the second run,
with 10 warmup pairs and 31 measured pairs per process. Inputs were the same eight
128px images, with pageable preallocated host IO. Both engines/contexts coexisted;
the CPU and GPU test suites had finished before timing began.

| Head | Run 1 median paired INT8/FP16 ratio | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Flat | 1.270 | 1.247 | 1.169 |
| Hierarchical | 1.247 | 1.210 | 1.189 |

Every ratio exceeded one: these local runs again found no INT8 host-latency
advantage. These are transfer/execution/synchronization observations, not
device-only time or evidence about the intended target GPUs. Unlike the earlier
scratch runner, this command inserts no timing events, so the measurement code
is not identical. Do not interpret differences from the earlier table as model
performance regressions. No new full-dataset quality measurement was made.

All final named outputs from all six processes matched the respective retained
FP16/INT8 engine smoke references exactly. Raw trials, hashes, loading observations
and outputs are in `tmp-trt-probe/pair-maintained-{flat,hierarchical}-{1,2,3}/`;
the collected summaries are in `tmp-trt-probe/pair-maintained-summary.json`.
No total runtime-memory reduction was established.

Static checks and the full CPU-default suite passed: 425 passed, 151 skipped,
and the known EMA expected failure. All 11 focused tests passed in the prepared
GPU environment, including actual multiple-input execution with pageable and
pinned buffers and retained deserialization failures. Pinned IO correctness was
tested; the real-model timing table above uses pageable IO only.

### Maintained ONNX calibration reproduction

`dev.benchmarks.onnx_calibration` now regenerates QDQ artifacts from explicit
ordered NPZ batches and a provenance manifest. It supports MinMax/Percentile
calibration, signed/unsigned activations, separate calibration/quantizer symmetry,
per-channel weights and floating/quantized biases. All calibration sessions use
explicit CPU thread limits from creation, and ORT's shape-inference sidecars stay
inside a private source snapshot in the output directory. See the
[manifest and commands](../dev/benchmarks/README.md#maintained-onnx-calibration-command).

The local reproduction used ONNX 1.22.0 and ONNX Runtime 1.29.0, one CPU thread,
and the exact retained 128 Blair training samples in their original 16 batches
of eight. Inputs were regenerated with each floating checkpoint's preprocessing
at 128px. Neither validation nor test samples were used. Both heads used the
signed symmetric activation/per-channel weight, floating-bias TensorRT recipe
with asymmetric Percentile 99.9 histogram collection.

| Head | Exactly equal tensor ranges | Exactly equal initializer arrays | Exactly equal graph nodes |
| --- | ---: | ---: | ---: |
| Flat | 339 | 1,374 | 1,366 |
| Hierarchical | 339 | 1,378 | 1,380 |

Names, input/output definitions and node order also matched the earlier
`tmp-trt-probe/{head}-float-bias.onnx` candidates. Both new graphs passed ONNX
checking and finite-output CPU smoke execution. Snapshot/external-file paths
change serialization hashes, so this is graph/tensor reproduction, not a claim
of identical file bytes. No new engine timing or full-dataset quality measurement
was made for this command milestone; the preceding results remain attributed to
their original engines and runs.

Input manifests and batches are retained under `tmp-onnx-calibration-inputs/`;
new models, caches, smoke outputs and reports are under
`tmp-onnx-calibration-{flat,hierarchical}/`. The maintained command validates a
declared calibration split, unique sample IDs, input hashes and dimensions; it
cannot independently prove that a caller's provenance excludes held-out data.
Dataset preparation, paired timing and full mini_metrics evaluation still need
to be connected into the maintained continuous pipeline.

Validation passed static checks, all ten focused calibration tests, and the full
CPU-default suite: 416 passed, 149 skipped and the known EMA expected failure.
This milestone did not rerun GPU tests or establish new target-hardware results.

### Maintained TensorRT engine reproduction

`dev.benchmarks.tensorrt_build` replaces the one-off engine build/inspection probe
with a documented command. It accepts arbitrary named execution inputs, explicit
min/opt/max profiles, precision flags and workspace/build settings. It records
model/external-weight/input hashes, environment, detailed layers, engine size and
context-memory requirement, and one set of actual outputs. An optional reference
checks exact names/shapes/dtypes and explicit numerical tolerances. Parser and
parity failures retain failed reports and available artifacts; a fresh output
directory is required.

The command was exercised on both signed floating-bias EfficientNetV2-S QDQ
candidates above, using batch eight, profile 1–8, size 128, workspace 1 GiB, builder
optimization level 1, and TF32/FP16 flags disabled. All flat and hierarchical
outputs passed comparison to the retained direct-engine outputs at rtol=1e-4,
atol=1e-5. Inspection again shows 170 INT8 convolutions and two head GEMMs per
engine. Reports and inspection are retained under ignored
`tmp-trt-probe/maintained-{flat,hierarchical}/`.

These checks establish reproduction and one-input execution, not new throughput,
full-dataset quality or every-shape parity results. TensorRT can choose different
tactics on rebuild; compare the exact engine hashes when interpreting timing
reports. The command is generic across model/head names but explicitly rejects
shape-tensor input profiles, host/nonlinear IO bindings and unresolved/empty
outputs that its smoke-test buffer handling does not yet support.

CPU contracts cover profile ranges, exact integer reference comparison, floating
parity and CLI help without TensorRT/PyTorch imports. Intentional CUDA checks cover real multi-input
engine construction, successful parity, retained mismatches, parser failure
messages and external-weight provenance. See the
[command and environment instructions](../dev/benchmarks/README.md#maintained-tensorrt-build-and-inspection-command).

Validation passed static checks and 406 CPU-default tests (149 skips and the
known EMA expected failure), plus all 16 focused checks in the prepared CUDA/
TensorRT environment and both real-model reference comparisons above.
