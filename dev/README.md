# Development checks

Tests are grouped by subsystem; see the [test suite map](../tests/README.md).
Use [local worktrees](worktrees.md) to develop independent branches concurrently.
For quantization work, follow the [branch roadmap](../docs/quantization-roadmap.md)
and [benchmark command index](benchmarks/README.md).
For a mounted global_lepi dataset on a manually allocated UCloud node, use the
[paired branch training comparison](ucloud/README.md), including fresh environment
setup, single-GPU/DDP launch and a checkpoint-based ONNX follow-up.

Use the uv-managed environment from the [installation guide](../README.md#local-installation).
The default sync includes the development dependency group. Choose a PyTorch backend
explicitly when syncing; checks use `.venv` without changing installed packages.

Run from the repository root:

```bash
bash dev/check.sh static
bash dev/check.sh test tests/core/test_config.py
bash dev/check.sh all
bash dev/check.sh test --cov=mini_trainer --cov-report=xml --cov-report=term
bash dev/check-wheel.sh
```

The script resolves the repository relative to itself, so it also works when invoked
by absolute path from another directory. With no arguments it runs static checks.
`static` checks lint, formatting, and the dependency contracts in `pyproject.toml`.
Formatting failures include the proposed diff; apply it with
`.venv/bin/python -m ruff format mini_trainer tests dev`, then rerun static checks.
CI uses the locked formatter; local checks use the installed version without syncing.
It does not import training code. `test` passes remaining arguments to pytest;
`all` runs static checks first and stops if they fail.

Tests default to hidden CUDA devices and a headless plotting backend. Set
`CUDA_VISIBLE_DEVICES=0` explicitly for tests intended to exercise a GPU. The CPU DDP
integration test launches two processes and requires localhost sockets; a restricted
sandbox may need to allow that execution. Failures caused by missing dependencies or
restricted execution should be reported separately from assertion failures.

The slow backbone tests are skipped unless `RUN_SLOW_TESTS=1`. Enabling them can
require downloaded weights or cached configurations. The separate
`tests/utils/run_compatibility_tests.py` utility also writes compatibility results
and may change the architecture blacklist; it is not part of routine checks.

## Reviewing a change

- Explain the observable behavior and compatibility guarantees.
- Run focused behavioral tests for changed code, plus static checks. Use the full
  suite for changes crossing training, loading, checkpointing, or harness boundaries.
- Keep synthetic tests independent of datasets, model downloads, and service accounts.
- For refactors, retain current defaults and output formats. Add regression tests
  where behavior is not already covered; avoid tests that merely repeat implementation.
- Record any untested GPU, backend, export, or optional-dependency cases explicitly.

## Installed package and dependency checks

`bash dev/check-wheel.sh` explicitly builds a wheel and installs it into a disposable
CPU environment using the core dependencies in `uv.lock`. It does not sync `.venv`.
It needs uv and access to dependencies, either cached or downloadable. Supply a Python
version or interpreter path as its first argument to select another supported Python.

The check runs outside the source tree with isolated Python imports. It verifies
minimal imports, every console entry point's help, the packaged architecture blacklist,
and a tiny image-folder training/reload/prediction round trip. Optional integrations
must be absent. Temporary environments and outputs are cleaned up when the check exits.

CI runs the source suite and the installed-wheel check on Python 3.12, 3.13, and 3.14.
Regular CI uses the tracked `uv.lock` with `uv sync --locked`. Local checks continue
to use the existing environment; they do not imply that it matches the lockfile.
For an explicit reproducible CPU setup, run `uv sync --locked --extra all --extra cpu`.
Select your CUDA extra instead when maintaining a GPU environment.

The separate scheduled/manual dependency-compatibility workflow runs `uv lock --upgrade`
in its disposable checkout, then the same checks. It does not update the committed lock.
To propose an upgrade locally, run `uv lock --upgrade` (or `uv lock --upgrade-package NAME`),
inspect the lock diff, explicitly sync the desired backend, and validate before committing.

## Behavioral coverage and known limits

Loader regressions live in `tests/data/test_loader.py`. They simulate restricted
CPU affinity, Python/platform fallbacks, and explicit worker counts without starting
large worker pools. Small image and RAM-cache fixtures check output compatibility,
and the distributed loader check verifies sampler/spawn configuration. The full CPU
DDP integration test remains the runtime check for distributed training.

`tests/training/test_checkpoint_contract.py` compares live and reloaded predictions, verifies
model/optimizer/scheduler/scaler state at the continuation boundary, and compares final
checkpoint contents after uninterrupted and resumed CPU float32 training. It retains
the original epoch budget and uses fixed data order with stochastic transforms disabled.
The disabled float32 scaler has empty state; this does not test active AMP restoration.
Checkpoints currently omit RNG and sampler state, so arbitrary stochastic runs are not
guaranteed to continue identically.

EMA state restoration is checked independently. Full EMA continuation currently has a
strict expected-failure regression: evaluation populates nonpersistent classifier cache
buffers with shapes that differ from the training model's buffers at the next EMA update.
See the [roadmap](../docs/roadmap.md) for the follow-up. Expected failures remain visible
in pytest output and become failures if they unexpectedly pass.

## ONNX checks

`tests/export/test_onnx.py` requires the `export` extra; optional backend cases additionally
require `timm`, `transformers` and `bioclip`. CI's `all` extra includes these.
Tests use randomly initialized offline models and check all classifier head families,
representative backbones, state preservation, masks and structured outputs.

`bash dev/check-onnx.sh [path/to/export-environment/bin/python]` exports a classifier
and compares predictions in a disposable environment containing ONNX Runtime and
its dependencies, with no PyTorch or mini_trainer. It explicitly installs the runtime
version from the export environment and needs registry access or cached packages.
It does not synchronize `.venv`. CI runs this in addition to the shared test harness.

## Reproducible dataset benchmarks

The [benchmark progression](benchmarks/README.md) starts with a fast synthetic
classification task with a known oracle and independent train/validation/test
splits. Its runner uses the actual training, checkpoint and inference paths.
MNIST and hierarchical Blair have explicit real-data profiles; their test data must
remain separate from configuration and checkpoint selection.

## Optimizer step contract

`trainer._optimizer_step` preserves the successful-step gate previously supplied
by MuonAuxAdamW's `_step_count`. The scheduler and EMA averaging update advance only
after a completed step that AMP did not reject. EMA still receives the original
batch-based index (`batches_per_epoch * epoch + batch_index`); its update-rate and
distillation schedules have not been redefined. EMA's separate cache bug is not fixed.

The trainer uses a temporary public optimizer post-step hook. Ordinary GradScaler
omits `optimizer.step()` on overflow, so that hook does not run. Native fused AdamW
and SGD enter `step()` even on overflow and skip inside the kernel: the hook captures
GradScaler's transient `optimizer.found_inf` tensor before it is removed. Only this
AMP-aware path reads the device flag for the Python scheduler/EMA decision. The
ordinary Muon/AdamW/SGD path adds no scaler-value readback or parameter comparison.

A completed step is not defined by whether parameters changed: zero learning rate
or zero gradients still count. Scale equality and the optimizer's return value
cannot establish success. Temporary hooks are removed even if the step raises,
and no new counter enters optimizer checkpoints. MuonAuxAdamW retains its own counter.

Custom optimizers must follow the ordinary Optimizer/GradScaler step contract or
the native `found_inf` AMP contract. The deprecated `step(..., grad_scaler=...)`
protocol is rejected before execution when scaling is enabled because its internal
skip cannot be observed reliably. Arbitrary custom internal no-ops are not inferred
by inspecting parameters. This boundary must be rechecked when PyTorch changes its
fused AMP contract.

`tests/training/test_optimizer_steps.py` uses real GradScaler overflow, scale growth and
recovery on CPU and optionally CUDA, including native fused AdamW/SGD. It checks
parameters, optimizer state, scheduler state and EMA call indices, as well as
zero-LR steps, scale underflow and hook cleanup. Checkpoint regressions additionally
compare uninterrupted/resumed MuonAuxAdamW, AdamW and momentum SGD training.

```bash
bash dev/check.sh test tests/training/test_optimizer_steps.py tests/training/test_checkpoint_contract.py
RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 bash dev/check.sh test tests/training/test_optimizer_steps.py -k cuda
```

The configured GPU benchmark workflow runs these CUDA regressions too. An explicitly
requested CUDA test fails when no device is available; ordinary CPU CI skips those
hardware cases. These are optimizer/AMP tests, not an EMA-functionality claim.

References: [optimizer post-step hooks](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Optimizer.register_step_post_hook.html)
and [GradScaler](https://docs.pytorch.org/docs/2.12/amp.html).

### CUDA batch transfer lookahead

`mt_train --cuda-prefetch` and `mt_predict --cuda-prefetch` opt into one-batch
transfer lookahead. Python loader builders accept `cuda_prefetch=True` with a CUDA
`device`. The returned object still inherits `DataLoader`, with the same sampler,
length, worker settings and repeated-epoch behavior; its batches are already on
the requested CUDA device. Shape, dtype and order are preserved. Model
preprocessing/augmentation stays on the caller's compute stream, so this works
independently of float32, AMP or INT8 model execution.

The option defaults off, rejects CPU targets, and is bypassed for an already
CUDA-cached dataset. It stages one additional batch and records stream usage so
the allocator cannot recycle batch storage before consumption completes. It does
not increase worker counts or add CPU reader threads. Custom CPU hooks may run a
batch earlier; stochastic hooks sharing global RNG state can therefore change
their interleaving with caller code. Returned batches may outlive iteration;
callers using them on another CUDA stream must establish their own stream handoff.
See [PyTorch stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-streams).

This is an opt-in throughput/memory tradeoff. Actual gains depend on the balance
between transfer and compute; compare peak allocation as well as wall time using
[the transfer probe](benchmarks/training.md#capacity-and-bottleneck-probes).

### Direct pinned cache batches

For a CUDA target with `cache="CPU"` and `num_workers=0`, the shared training
loader now gathers cached rows directly into pinned batch storage. This removes
the intermediate pageable batch and its second copy during pinning. Batches own
their storage: modifying one cannot change the cache, and keeping an older batch
cannot cause it to be overwritten by a later iteration. Sampling, label order,
shape and dtype are unchanged.

This applies automatically with either ordinary transfer or `--cuda-prefetch`.
Raw `LazyDataset` users can request `pin_batches=True` for the same behavior.
Worker processes always use the ordinary gather path and DataLoader's parent-side
pinning; this option never initializes the CUDA pin allocator in a worker. CPU
training, CUDA-cached datasets and scalar indexing keep their existing behavior.

### Direct collation of stacked batches

Repository loaders now retain the gathered batch tensors through collation,
avoiding creation of one image/label view per sample. Their batch sampler tags
index lists for this internal path; dataset identity, shuffle/drop-last behavior,
distributed sampler access and RNG consumption are preserved.

Ordinary external `LazyDataset.__getitems__` calls still return actual sample
lists, including direct `torch.stack` compatibility. An external DataLoader that
reuses the repository batch sampler with its default collator materializes sample
views on demand. Tests cover image-only and image/label batches with both direct
loading and CPU caches, including spawned workers and CUDA prefetch.

A one-thread cache benchmark with 4,096 uint8 RGB 28×28 images, batch size 128,
and seven alternating trials measured approximately 0.52 million samples/s before
this change and 2.22 million afterward. This isolates cached iteration; larger
images, decoding, transfer and model compute change the overall benefit. See the
[integrated measurements](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/archive/benchmark-history.md#larger-batches-and-direct-collation).

### Model compilation

`mt_train --compile --compile-mode reduce-overhead` selects a PyTorch model
compilation mode. The Python training entry points accept `compile_mode`, and
`dev.benchmarks.training.run` accepts the same CLI flag and records it in success and
failure reports. An explicit mode requires `--compile`; omitting it preserves
ordinary `torch.compile(model)` behavior. Optimizer compilation remains separate.

Supported modes are `default`, `reduce-overhead`, `max-autotune`, and
`max-autotune-no-cudagraphs`. PyTorch's CUDA graph modes can reduce launch overhead
for eligible graphs, but capture is not guaranteed and workspace caching can
increase memory. Measure both float and INT8 with the same mode, including
compilation time, later training phases, peak allocation and held-out quality.
See the [PyTorch compilation modes](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html).

### Optimizer compilation

`mt_train --compile-optimizer` opts into compiling optimizer updates independently
of model `--compile`. The benchmark runner accepts the same option and records it
in result and failure reports. It defaults off; compilation overhead and graph
breaks can outweigh any steady-state benefit, so measure the intended workload.

The first real optimizer call initializes lazy state eagerly. Subsequent calls
use compilation with tensor learning rates, allowing scheduler changes without
specializing a graph for every numeric rate. AMP overflow handling and scheduler
advancement remain controlled by the trainer. MuonAuxAdamW compiles its child
optimizers while keeping its outer step counter in Python; Muon's compilation
preserves the explicit BF16 casts in its Newton-Schulz iterations.

For custom training, call `mini_trainer.training.compilation.compile_optimizer`
after constructing the scheduler and restoring checkpoint state. Saved learning
rates remain ordinary scalars, so a checkpoint can resume without compilation.
Explicit `foreach=True` Adam/AdamW requires `capturable=True`; unsupported
combinations fail before the helper changes the optimizer. Default and explicit
`foreach=False` updates do not need that setting.

Regression coverage compares eager and compiled SGD, AdamW, their native fused
variants, and MuonAuxAdamW on CUDA, including overflow skips, scheduler changes,
parameter updates and optimizer state. This does not establish support for every
custom optimizer or a real-workload INT8 speedup. Quantized update dispatch and
its performance remain a separate validation boundary.

### Optimizer CUDA graphs

`mt_train --compile-optimizer --optimizer-cudagraphs` additionally requests CUDA
graph replay for optimizer updates. It requires parameters on one CUDA device
and the default Inductor backend. It is independent of model compilation;
`--compile --compile-mode reduce-overhead` can enable model graphs as well.
The benchmark runner accepts and records the same optimizer option.

For custom training, call `compile_optimizer(optimizer, cudagraphs=True)` after
scheduler construction and checkpoint restoration. The first real update still
initializes state eagerly, under the trainer's AMP gating. Subsequent updates use
device-resident learning rates. Scheduler updates, overflow decisions and Muon's
outer step counter remain outside graph capture. Changing the graph setting of
an already compiled optimizer is rejected rather than silently ignored.

Numeric rates retain float64 precision; explicitly supplied tensor rates retain
their dtype. Checkpoints save numeric values plus a `_mini_trainer_lr_dtype`
marker for non-default precision, allowing compiled restoration to recover that
choice. Ordinary eager loading still accepts the numeric rates. Native fused
SGD/Adam/AdamW tensor-rate kernels require an explicitly selected float32 tensor
rate (or its recorded checkpoint marker); unsupported rates fail before mutation.
The existing foreach/capturable restrictions still apply.

INT8 updates use compiler-visible arithmetic and final storage copies during AOT
fake-tensor tracing. This avoids an opaque in-place operator carrying CPU scalar
inputs into CUDA graph partitions. The native row update remains available in
eager execution. Regression tests check actual optimizer-only replay, arithmetic,
AMP skips, rate precision, same-optimizer restoration and eager checkpoint resume.
Capture eligibility, extra gradient copies, graph workspace memory and first-use
compilation costs still depend on the optimizer and workload. Measure both float
and INT8 with the same options; enabling graphs alone is not evidence of a speedup.

## Agent-only changes and CI

Material primarily for coding agents lives in [`.agents/`](../.agents/README.md),
with root [`AGENTS.md`](../AGENTS.md) as the entry point. Durable notes use the
[shared format](../.agents/notes/README.md); scratch work belongs in ignored
`.agents/local/`. Developer-facing documentation remains in `docs/` and `dev/`.

Use separate `agent:` commits for agent instructions and notes. This prefix marks
purpose, not authorship, and does not disable checks. CI workflow changes, executable
helpers and application changes receive normal separate commits.

The CI and dataset benchmark workflows skip pushes that change only `AGENTS.md`,
Markdown inside `.agents/`, or `.agents/.gitignore`. Other Markdown, code, scripts,
configuration and workflow changes still run the usual checks. On pull requests,
a small `scope` job compares the complete PR diff; agent-only changes skip costly
jobs while preserving job statuses. Mixed changes run checks regardless of commit
messages. Scheduled and manual benchmarks keep their existing behavior.

`dev/ci_scope.py` uses Git and the Python standard library, with no environment
installation. Missing/unreadable/empty comparisons run checks conservatively;
renames inspect both old and new paths. If classification fails, downstream checks
still run unless the workflow was cancelled. The push patterns are also checked
against the classifier by `tests/core/test_ci_scope.py`.

Do not use `[skip ci]` as an alternative: GitHub documents that workflow-level
skipping can leave [required PR checks pending](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushpull_requestpull_request_targetpathspaths-ignore).
Agent-only edits need content/link review and `git diff --check`; changes to the
classifier or workflows need their focused checks. Release-tag, scheduled and
manual workflows are not disabled by agent commit messages.

## Automatic CPU budgets

Automatic loader and cache worker counts now use the smallest detected process
CPU count, affinity mask, visible Linux cgroup CPU quota, and positive
`SLURM_CPUS_PER_TASK` allocation. Cgroup v1 and v2 ancestor limits are included;
fractional CPU quotas are rounded down before applying the existing four-CPU
reserve and worker caps. Explicit worker counts, including zero, remain unchanged.
Unreadable, unlimited, or malformed quota data falls back to the other signals.

These are resource ceilings, not a measurement of contention from other jobs.
For a deliberately shared allocation, set worker counts explicitly when needed.
The relevant interfaces are documented by the
[Linux kernel](https://docs.kernel.org/admin-guide/cgroup-v2.html#cpu-interface-files)
and [Slurm](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_CPUS_PER_TASK).

CI runs on pull requests targeting `master` and on pushes to `master`. Feature
branches such as `quant` use PR checks, avoiding duplicate push/PR jobs. New
commits cancel superseded runs for the same PR or branch. The x86 quantization
job forces AVX2 to verify portability without relying on runner VNNI support.

## PR change statistics

`pr-change-summary.yml` maintains one bot comment per PR, with file and line counts
for the core module, tests, benchmark tooling, CI, packaging, and other areas.
Markdown is excluded from the headline except root `README.md`, whose onboarding
instructions are part of the user interface. Other Markdown remains visible in a
separate row. The workflow's `featureMarkdown` set is the explicit exception list;
add a path there when its content is itself a delivered feature.

The workflow reads GitHub PR metadata only. It uses `pull_request_target` with
permission to comment, performs no checkout and never executes PR content. It
starts working once installed on the PR base branch. Counts use GitHub's PR diff,
classify renames by destination, and flag incomplete results above the API's
3,000-file limit. They measure change volume, not quality or development effort.
