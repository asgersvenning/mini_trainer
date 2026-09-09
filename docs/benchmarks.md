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
