# Reproducible benchmark progression

Start with a synthetic oracle, then MNIST, then hierarchical Blair. CPU is the fast
continuous check; GPU profiles validate additional behavior. Birds and iNaturalist
2021 remain optional larger follow-ups. Notebook settings and timing claims are
references, not acceptance criteria.

See [visible results and coverage](../../docs/benchmarks.md) for initial observations,
known issues, reporting and storage limits.

## Shared pipeline commands

Use the existing environment; these commands never install or synchronize packages:

```bash
bash dev/check-benchmarks.sh cpu /tmp/benchmarks-cpu
bash dev/check-benchmarks.sh gpu /tmp/benchmarks-gpu
BENCHMARK_DATA_ROOT="$PWD/examples" \
BLAIR_CLASS_SPEC="$PWD/examples/blair/blair_model/class_spec.json" \
    bash dev/check-benchmarks.sh real /tmp/benchmarks-real
bash dev/check.sh test tests/test_benchmark_synthetic.py tests/test_benchmark_datasets.py
```

Each results directory must be new. Set `BENCHMARK_PYTHON` to use a different prepared
interpreter. CPU runs the synthetic float32 profile. GPU runs synthetic float32,
float16 AMP and bfloat16 AMP with CUDA caching. Real runs MNIST on CPU and CUDA,
then hierarchical Blair on CUDA. Profiles run sequentially, retain failures, and
return a nonzero exit code if any profile fails. No requested GPU silently falls
back to CPU. The real profile requires existing datasets and a saved Blair class
specification; it performs no downloads or online taxonomy queries.

Individual runs are configurable:

```bash
.venv/bin/python -m dev.benchmarks.run --output /tmp/oracle \
    --seed 42 --threads 1 --device cpu
.venv/bin/python -m dev.benchmarks.run --dataset mnist \
    --data-root examples/mnist --output /tmp/mnist-amp --epochs 5 \
    --device cuda:0 --dtype float16 --cache CUDA --allow-nondeterministic
```

`--num-workers` defaults to zero and is always explicit in reports. CUDA caching
forces zero effective loader workers. Threads default to one. Cache modes are
`NONE`, `RAM` and `CUDA`. The CPU float32 profile and synthetic GPU profiles use
strict deterministic algorithms. Real GPU profiles explicitly allow nondeterministic
operations because CUDA adaptive-pooling backward lacks a deterministic implementation.
AMP, CUDA caching and algorithm determinism are separate report fields.

## Dataset and training contracts

The synthetic generator uses four balanced classes encoding two binary factors in
red and green channels of 8-by-8 RGB images. Intensities are 32 or 224 with bounded
noise of at most 16. Thresholding channel means at 128 is an exact 100% oracle;
uniform guessing achieves 25% in expectation. Blue contains nuisance noise. A parent
label encodes the red factor, ready for a future synthetic hierarchical profile.

Independent streams keyed by seed, split, class and sample generate 128 training,
32 validation and 64 test images. The synthetic runner uses flat labels, a channel-mean
backbone and an unnormalized linear classifier. Fixed 12-epoch runs train with
MuonAuxAdamW at learning rate 0.1. The exact oracle quality gate fails if held-out
accuracy is below 100%.

MNIST and Blair use a tiny randomly initialized convolutional backbone, five epochs
and learning rate 0.01 in the shared profile. MNIST uses 28-pixel inputs and a flat
head; Blair uses 64-pixel inputs and the normalized hierarchical head. Blair's
reviewed specification must exactly cover the training class names. It is included
in the manifest so mappings cannot change silently across runs.

For real datasets, 20% of each class's unique training files become validation data,
using seeded shuffling. Byte-identical duplicates stay together. Training files
identical to supplied test files are excluded from the index and recorded; the
source files and official test split remain untouched. Conflicting cross-split
content fails validation. Hashes cover encoded file bytes, not perceptual duplicates.

Every profile uses the actual mini_trainer training, checkpoint reconstruction and
inference loader. The final checkpoint is evaluated after a fixed epoch budget;
held-out labels are never used for checkpoint or hyperparameter selection. There
is no augmentation, EMA or regularization in this baseline. Real-data completion
has no quality acceptance threshold yet. Declare such thresholds and comparisons
before attempting improvements; do not tune against these test scores.

## Reproduction artifacts

Each run writes `report.json`, `predictions.npz` with sample paths and per-level
scores/targets, dataset/split manifests, and a normal training directory containing
resolved configuration and checkpoints. Synthetic images can be regenerated;
real images remain external. Scores retain the model's evaluation-forward semantics.
The report contains source, lockfile, dataset and checkpoint hashes, versions,
configuration, hardware, coverage flags, quality and synchronized training-call wall
time. Peak CUDA allocated memory covers training, not total device/process memory.

Wall time includes initialization, training, validation, logging and checkpoint
writes. It excludes interpreter startup, inventory/generation and final inference.
It is not pure throughput or loader-wait time. Dedicated profiling and repeated,
matched experiments are still needed to establish speed improvements.

Tests verify exact repeated CPU predictions within one environment, matching dataset
hashes, independent splits, oracle correctness, explicit GPU failure and taxonomy
contracts. They do not guarantee byte-identical checkpoints or cross-version/GPU
numerical identity. Reports explicitly mark unexercised features; do not infer full
module coverage from a passing profile.

## Continuous execution and visible reporting

`.github/workflows/benchmarks.yml` runs the CPU oracle on pull requests, master pushes
and weekly. It uses a locked CPU environment and uploads results even on failure.
The Actions run page contains a readable summary and artifacts with 90-day retention.
The README links to that history. The source checkout is not modified to publish results.
These are initial reporting facilities; durable archival and a historical dashboard
remain follow-up work.

GPU execution uses a configured self-hosted runner labeled `self-hosted`, `linux`,
`gpu`. It creates a disposable uv environment with an explicitly selected CUDA extra,
leaving the runner's existing environments alone. Enable it manually using the `gpu`
workflow input, or set repository variable `ENABLE_GPU_BENCHMARKS=true` for weekly
execution. Set `BENCHMARK_CUDA_BACKEND` for scheduled runs (default `cu130`).

For real-data jobs, enable input `real_data` together with `gpu`, or set
`BENCHMARK_REAL_DATA=true`. Set repository variables `BENCHMARK_DATA_ROOT` and
`BLAIR_CLASS_SPEC` to stable dataset and specification paths on that runner, outside
the checkout. The workflow does not assume these resources already exist and cannot
validate GPU behavior on a CPU-only hosted runner. No GPU job has been dispatched
from this development session; the equivalent local commands have been exercised.

References: [Actions job summaries](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary)
and [artifact retention](https://github.com/actions/upload-artifact#retention-period).
