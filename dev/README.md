# Development guide

Use the existing uv environment from [installation](../README.md#local-installation).
Checks never synchronize it; explicitly choose the PyTorch backend when installing
dependencies. Start with the [roadmap](../docs/roadmap.md) for priorities.

| Task | Guide |
|---|---|
| Locate behavioral coverage | [Test suite map](../tests/README.md) |
| Work on independent branches | [Worktrees](worktrees.md) |
| Run training/inference experiments | [Benchmark commands](benchmarks/README.md) |
| Compare branches on UCloud | [UCloud setup and execution](ucloud/README.md) |
| Qualify native INT8 or PTQ/QAT | [Quantization roadmap](../docs/quantization-roadmap.md) |
| Maintain the MAMBO release | [Release inputs and workflows](releases/mambo_v3/README.md) |
| Change loading or optimizer behavior | [Runtime contracts](#training-and-loading-contracts) |
| Reproduce publication experiments | [Research workflows](../publication/experiments/README.md) |

From the repository root:

```bash
bash dev/check.sh static
bash dev/check.sh test tests/core/test_config.py
bash dev/check.sh all
bash dev/check.sh test --cov=mini_trainer --cov-report=xml --cov-report=term
bash dev/check-wheel.sh
```

`static` checks Ruff formatting/lint and import contracts without importing the
package. `test` forwards arguments to pytest; `all` runs static checks before tests.
The harness also works by absolute path. Apply formatting with
`.venv/bin/python -m ruff format mini_trainer tests dev`.

Tests hide CUDA and use a headless plotting backend by default. Set
`CUDA_VISIBLE_DEVICES=0` for intentional GPU checks; some also require
`RUN_CUDA_TESTS=1`. Slow backbone tests require `RUN_SLOW_TESTS=1` and may download
weights. CPU DDP tests need localhost sockets. Report dependency/sandbox failures
separately from assertion failures. The optional
`tests/utils/run_compatibility_tests.py` can mutate the architecture blacklist;
it is not a routine validation command.

## Reviewing a change

Preserve public defaults, output formats and checkpoint identities during refactors.
Use existing regression coverage before adding tests; cover observable behavior
and meaningful failures, not private layout or configuration literals. Run focused
tests plus static checks, and the full suite for changes spanning training, loading,
checkpointing or the validation harness. Record skipped and untested boundaries.
Documentation-only edits need content/link and diff checks, not model execution.

## Installed package and dependency checks

`bash dev/check-wheel.sh [python-version-or-path]` builds and installs the wheel
with locked core CPU dependencies in a disposable environment outside the checkout.
It checks minimal imports, CLI help, packaged metadata and a tiny training/reload/
prediction round trip without optional integrations. It needs uv and cached or
downloadable dependencies; it does not sync `.venv`.

CI runs source and wheel checks on Python 3.12–3.14 using the committed lock.
The scheduled/manual dependency workflow upgrades its disposable lock before
running the same checks. To propose an upgrade, explicitly update the lock,
review it, sync the intended backend and validate. Local checks use the installed
environment and do not establish agreement with the lock.

## Behavioral coverage and known limits

The [test map](../tests/README.md) owns subsystem coverage.
Checkpoint tests compare model/optimizer/scheduler/scaler state and uninterrupted
versus resumed CPU float32 training with fixed order and no stochastic transforms.
They do not qualify active AMP restoration or arbitrary stochastic continuation:
checkpoints omit RNG and sampler state.

EMA continuation retains a strict expected failure for incompatible classifier
cache-buffer shapes after evaluation. Preserve that regression until fixed; see
[the roadmap](../docs/roadmap.md). Passing CPU checks does not establish CUDA,
AMP, optional-backbone or distributed behavior.

## ONNX checks

Export tests use offline initialized models and require the `export` extra;
optional backbones need their own dependencies. See [export contracts](../docs/onnx.md).

`bash dev/check-onnx.sh [export-environment/bin/python]` exports a classifier and
checks predictions in a disposable ONNX Runtime environment without PyTorch or
mini_trainer. It installs the export environment's runtime version and needs cache
or registry access; the working environment is unchanged.

## Reproducible dataset benchmarks

Follow the [benchmark progression](benchmarks/README.md): synthetic oracle,
MNIST, then hierarchical Blair. Keep test splits separate from configuration and
checkpoint selection. A passing correctness profile is not a throughput claim.

## Agent-only changes and CI

Follow [agent workspace policy](../.agents/README.md) for file placement and separate
`agent:` commits. The prefix does not disable checks. CI classifies the entire diff:
only `AGENTS.md`, Markdown under `.agents/` and `.agents/.gitignore` qualify as
agent-only. Mixed or unreadable changes run checks; scheduled/manual jobs are
unchanged. Do not use `[skip ci]` to bypass required checks.

[The classifier](ci_scope.py) and [its tests](../tests/core/test_ci_scope.py) own exact
path/rename handling. Required-check behavior on the first hosted agent-only PR
still needs verification under the actual branch-protection settings.

## PR change statistics

[The workflow](../.github/workflows/pr-change-summary.yml) reports PR changes by
content group. Only root `README.md` is included in its code headline; other Markdown
has a separate row. Its `featureMarkdown` set defines exceptions.

It reads GitHub metadata without checking out or executing PR code, flags incomplete
diffs above 3,000 files and classifies renames by destination. Counts describe volume,
not quality or effort.

## Training and loading contracts

These constraints matter when changing runtime behavior. The linked implementations
and tests own exact cases; benchmark commands are in the [training guide](benchmarks/training.md).

### Optimizer step contract

Scheduler and EMA updates follow successful optimizer steps, including zero-LR or
zero-gradient updates. Parameter changes, scale equality and return values do not establish success.
Fused AMP optimizers may enter `step()` on overflow; the trainer observes their
`found_inf` skip flag. The deprecated `step(..., grad_scaler=...)` protocol is rejected
with scaling enabled. [Tests](../tests/training/test_optimizer_steps.py) cover overflow,
recovery and hook cleanup; checkpoint tests cover continuation. Run the CUDA cases with:

```bash
RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 bash dev/check.sh test tests/training/test_optimizer_steps.py -k cuda
```

### CUDA batch transfer lookahead

`mt_train --cuda-prefetch`, `mt_predict --cuda-prefetch` and Python `cuda_prefetch=True` opt into
one-batch transfer lookahead. CPU targets are rejected; CUDA-cached data bypasses it.
Sampling/order are preserved and preprocessing stays on the caller's compute stream.
CPU hooks may execute earlier, affecting shared RNG interleaving. Callers using
another CUDA stream own that handoff. Use the
[transfer probe](benchmarks/training.md#capacity-and-bottleneck-probes) to measure benefits.

### Direct pinned cache batches

CUDA with `cache="CPU"` and `num_workers=0` gathers directly into owned pinned
batches (`LazyDataset(pin_batches=True)`). Workers use ordinary gather followed by
parent-side pinning; they must not initialize CUDA pin allocators. Retaining or
modifying a batch must not corrupt the cache or subsequent batches.

### Direct collation of stacked batches

Repository loaders avoid splitting gathered batches into per-sample views.
External `LazyDataset.__getitems__` still returns sample lists for default collators.
Preserve sampler/RNG, shuffle/drop-last and worker behavior; see
[loader tests](../tests/data/test_loader.py).

### Model compilation

`--compile-mode` requires `--compile`; choices are `default`, `reduce-overhead`,
`max-autotune` and `max-autotune-no-cudagraphs`. Optimizer compilation is separate.
Measure startup, warm throughput, memory and held-out quality; speedup is not guaranteed.

### Optimizer compilation

`--compile-optimizer` defaults off. Custom training calls
`mini_trainer.training.compilation.compile_optimizer` after scheduler construction
and checkpoint restoration. The first real update initializes state eagerly;
subsequent updates use tensor learning rates without specializing each numeric rate.
AMP skip decisions stay with the trainer; saved scalar rates support eager resume.
Explicit foreach Adam/AdamW requires `capturable=True`; incompatible settings fail
before mutation.

### Optimizer CUDA graphs

`--compile-optimizer --optimizer-cudagraphs` requests Inductor replay on one CUDA
device; custom code uses `compile_optimizer(optimizer, cudagraphs=True)`.
Scheduler calls, overflow decisions and Muon's outer counter stay outside capture.
An already compiled optimizer cannot change its graph setting.

Numeric rates retain float64 precision; explicitly typed tensor rates retain their
dtype in checkpoints. Fused SGD/Adam/AdamW graph kernels require explicitly selected
float32 rates; never silently coerce them. Foreach/capturable restrictions still apply.
[Implementation](../mini_trainer/training/compilation.py) and
[tests](../tests/training/test_optimizer_steps.py) define the exact state contract;
graph overhead and INT8 performance need workload-specific evidence.

### Automatic CPU budgets

Defaults use the smallest available process CPU count, affinity mask, visible
Linux cgroup quota (including ancestors) and positive `SLURM_CPUS_PER_TASK`.
Fractional quotas round down; unusable signals fall back to the others. Explicit
counts, including zero, are preserved.

| Consumer | Automatic budget | Override |
|---|---|---|
| Training DataLoader | CPUs minus 4, rounded down to even, bounded 0–16 | `--num_workers` |
| Prediction DataLoader | Same rule, capped at 32 | `--num_workers` |
| Training cache preparation | Same rule, capped at 16; zero is serial | `--cache-workers` / `cache_workers` |

CUDA-cached datasets force zero DataLoader workers. These limits do not measure
competition from other jobs; set explicit per-rank budgets when sharing resources.
