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
