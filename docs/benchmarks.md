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
