# Development checks

Use the uv-managed environment from the [installation guide](../README.md#local-installation).
The default sync includes the development dependency group. Choose a PyTorch backend
explicitly when syncing; checks use `.venv` without changing installed packages.

Run from the repository root:

```bash
bash dev/check.sh static
bash dev/check.sh test tests/test_config.py
bash dev/check.sh all
bash dev/check.sh test --cov=mini_trainer --cov-report=xml --cov-report=term
bash dev/check-wheel.sh
```

The script resolves the repository relative to itself, so it also works when invoked
by absolute path from another directory. With no arguments it runs static checks.
`static` checks lint, formatting, and the dependency contracts in `pyproject.toml`.
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

Loader regressions live in `tests/utils/test_loader.py`. They simulate restricted
CPU affinity, Python/platform fallbacks, and explicit worker counts without starting
large worker pools. Small image and RAM-cache fixtures check output compatibility,
and the distributed loader check verifies sampler/spawn configuration. The full CPU
DDP integration test remains the runtime check for distributed training.

`tests/test_checkpoint_contract.py` compares live and reloaded predictions, verifies
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

`tests/test_onnx.py` requires the `export` extra; optional backend cases additionally
require `timm`, `transformers` and `bioclip`. CI's `all` extra includes these.
Tests use randomly initialized offline models and check all classifier head families,
representative backbones, state preservation, masks and structured outputs.

`bash dev/check-onnx.sh [path/to/export-environment/bin/python]` exports a classifier
and compares predictions in a disposable environment containing ONNX Runtime and
its dependencies, with no PyTorch or mini_trainer. It explicitly installs the runtime
version from the export environment and needs registry access or cached packages.
It does not synchronize `.venv`. CI runs this in addition to the shared test harness.
