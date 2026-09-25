# Benchmark workflows

Start with the synthetic CPU oracle, then GPU profiles, then MNIST and hierarchical
Blair. These exercise training, checkpoint reconstruction and inference. Notebook
timings are illustrations, not acceptance criteria.

| Task | Guide |
|---|---|
| Paired training, native INT8 and loading/capacity probes | [Training](training.md) |
| ONNX/TensorRT preparation, calibration, quality and resources | [Inference](inference.md) |
| Target execution, compact history and optional publishing | [Reporting](reporting.md) |
| Measured findings and limitations | [Findings](../../docs/benchmarks.md) |
| Unfinished qualification | [Quantization roadmap](../../docs/quantization-roadmap.md) |

## Shared pipeline commands

Use a prepared environment and a new output directory for each run:

```bash
bash dev/check-benchmarks.sh cpu /tmp/benchmarks-cpu
bash dev/check-benchmarks.sh gpu /tmp/benchmarks-gpu
BENCHMARK_DATA_ROOT="$PWD/examples" \
BLAIR_CLASS_SPEC="$PWD/examples/blair/blair_model/class_spec.json" \
    bash dev/check-benchmarks.sh real /tmp/benchmarks-real
```

`BENCHMARK_PYTHON` selects an alternative interpreter; otherwise the harness uses
`.venv/bin/python` without installing or synchronizing dependencies.

| Profile | Workload |
|---|---|
| `cpu` | Synthetic float32 oracle |
| `gpu` | Synthetic float32, FP16 AMP and BF16 AMP with CUDA caching |
| `real` | MNIST on CPU/CUDA, then hierarchical Blair on CUDA |

Profiles run sequentially, retain failures and return nonzero if any run fails.
A requested GPU never silently falls back to CPU. Real-data runs require existing
datasets and a reviewed Blair class specification; they do not download data or
query taxonomy online.

For individual settings, use `python -m dev.benchmarks.training.run --help`:

```bash
.venv/bin/python -m dev.benchmarks.training.run --dataset mnist \
    --data-root examples/mnist --output /tmp/mnist-amp --epochs 5 \
    --device cuda:0 --dtype float16 --cache CUDA --allow-nondeterministic
```

Default workers are zero and runtime threads one. CUDA caching forces zero loader
workers. Synthetic profiles use deterministic algorithms; real GPU profiles allow
nondeterminism because adaptive-pooling backward lacks a deterministic CUDA path.
AMP, caching and determinism are separate settings and report fields.

## Dataset and training contracts

The [synthetic generator](data/synthetic.py) has independent seeded splits and an
exact 100% oracle. Its runner fails the quality gate below 100% held-out accuracy.
It is a correctness test; real-data profiles currently have no quality acceptance
threshold. Fix budgets and quality criteria before comparing changes.

For real data, a seeded 20% of each class's unique training files forms validation.
Byte-identical duplicates stay together; training copies of test files are excluded
from the index and recorded without changing source files or the official test
split. Conflicting labels for identical content fail validation. Hashing detects
encoded-byte equality, not perceptual duplicates. Blair's reviewed class mapping
must exactly cover training classes and is retained in the manifest.

The final checkpoint is evaluated after a fixed epoch budget; test labels never
select checkpoints or hyperparameters. Baseline profiles disable augmentation,
EMA and regularization. Exact model and optimizer recipes live in
[the runner](training/run.py) and [shared harness](../check-benchmarks.sh).

## Reproduction artifacts

Retain `report.json`, `predictions.npz`, dataset/split manifests and the training
directory with resolved configuration/checkpoints. Reports record source,
environment, input and model identities; scores retain evaluation-forward semantics.
Synthetic images are regenerable; real images remain external.

Reported training-call wall time includes setup, training, validation, logging and
checkpoint writes, but excludes interpreter startup, data inventory and final
inference. CUDA allocated peak is not total process/device memory. Neither measure
is pure inference throughput. Cross-environment numerical identity and complete
feature coverage are not implied by a passing profile.

## Continuous execution and visible reporting

[Dataset CI](../../.github/workflows/benchmarks.yml) runs the CPU oracle on PRs,
master pushes and weekly, with a locked CPU environment. Actions summaries and
artifacts include failures; artifacts expire after 90 days.

GPU jobs require a self-hosted Linux runner labelled `gpu`. They install a
disposable environment with an explicitly selected CUDA extra.

| Configuration | Purpose |
|---|---|
| Manual `gpu` / `ENABLE_GPU_BENCHMARKS=true` | Enable GPU jobs manually / weekly |
| Manual `cuda_backend` / `BENCHMARK_CUDA_BACKEND` | CUDA extra; default `cu130` |
| Manual `real_data` / `BENCHMARK_REAL_DATA=true` | Include real-data profiles; requires GPU job |
| `BENCHMARK_DATA_ROOT`, `BLAIR_CLASS_SPEC` | Existing dataset/specification paths outside checkout |
| Manual `qt` / `BENCHMARK_QT=true` | Include paired native INT8 profiles; requires GPU job |

A configured workflow is not evidence of a successful target run. The separate
[history publisher](reporting.md#opt-in-public-history-publisher) remains opt-in
and needs live qualification; it does not make ordinary dataset artifacts permanent.

## Source layout and command migration

| Package | Responsibility |
|---|---|
| `training/` | Dataset training, QT/PTQ probes, large heads, prediction adapter |
| `data/` | Generation/indexing, loading, caching, reading and transfers |
| `inference/` | Input preparation, ONNX/TensorRT, paired quality/resources |
| `reporting/` | Summaries, immutable history and storage |

Development CLI modules use these package paths; shared `dev/check-*.sh` commands
are unchanged. Preserve `models.py` for saved checkpoint identities,
`prepare_inputs.py` for saved preprocessing factories and `_int8_weight.py` for
older probe imports. Historical source hashes must not be relabelled as new runs.
