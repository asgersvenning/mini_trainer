# Working on mini_trainer

## Start here

- Read `README.md`, `docs/roadmap.md`, and the relevant code and tests before editing.
- Read `[tool.importlinter.contracts]` in `pyproject.toml` before changing imports.
- Check `git status --short` and preserve unrelated work. Keep each change focused and reviewable.
- This file is the repository-wide agent entry point. Read the focused
  [contribution guidance](.agents/rules/code-contribution.md) when making changes;
  this file takes precedence where they conflict.
- Follow [.agents/README.md](.agents/README.md) for agent material and commit boundaries.
  Read only task-relevant notes and skills; they are not all startup context.

## Parallel work and worktrees

- Before creating or using a linked worktree, or coordinating concurrent agents,
  read [.agents/rules/worktrees.md](.agents/rules/worktrees.md). Setup commands are
  in [dev/worktrees.md](dev/worktrees.md).
- Assign each concurrent implementation task one branch, one absolute worktree
  path and a bounded write scope. Pass that path explicitly to tools and agents;
  never assume their default working directory changed with yours.
- Preserve other worktrees and shared resources. Do not switch another agent's
  branch, modify its files, or change a shared environment while it is in use.
- Only the designated integrator merges completed work into the target branch,
  after reviewing commits and validating the combined result. Worktree setup is
  not itself an instruction to spawn agents.

## Priorities

Follow the order in `docs/roadmap.md`: development safeguards; behavior-preserving
simplification; ONNX export and Hugging Face packaging; training efficiency;
`mini_metrics` and continuous evaluation; additional dataset formats.
Complete a bounded, validated increment before moving to the next priority.

## Environment and validation

- Use the existing uv-managed `.venv`. Run its executables directly or use
  `uv run --no-sync`. Never implicitly synchronize dependencies while running checks.
- When dependency installation is needed, explicitly select the intended PyTorch
  backend as described in the README. Do not replace a working CUDA installation
  with CPU or different CUDA wheels as a side effect of validation.
- Run `bash dev/check.sh static` for Ruff lint, formatting, and import contracts.
  Architecture validation is static; it must not import the package or initialize CUDA.
- For behavioral changes, run `bash dev/check.sh test <affected test paths>`.
  Run `bash dev/check.sh all` for changes spanning training, loading, checkpointing,
  or the validation harness. Runtime tests are permitted and necessary to check behavior.
- For packaging or optional-dependency changes, run `bash dev/check-wheel.sh` to
  validate a minimal installed wheel outside the checkout. This explicitly installs
  locked CPU dependencies into a disposable environment; it does not sync `.venv`.
- The harness hides CUDA from tests by default. For intentional GPU verification,
  explicitly set `CUDA_VISIBLE_DEVICES`. CPU tests do not establish GPU correctness.
- Slow backbone tests require `RUN_SLOW_TESTS=1` and may need model downloads.
  The compatibility runner can update the backbone blacklist; use it only when
  that mutation is part of the task.
- Report checks run, failures, skips, and limits honestly. Do not weaken checks or
  alter expected results merely to make a refactor pass.

## Compatibility and architecture

- Keep public imports, signatures, CLI defaults, return types, tensor shape/dtype/device,
  class ordering, serialization, and checkpoint/resume behavior stable during refactors.
- Separate behavior fixes and opt-in features from mechanical cleanup. Establish
  regression coverage for affected behavior before substantial restructuring.
- Respect the declared dependency layers and acyclic sibling contract. Use single-dot
  relative imports within a package and absolute imports across directories.
  Do not add import-contract exceptions to conceal a new cycle.
- Keep required dependencies small. `pyproject.toml` is the source of truth for the
  existing dependency set; new third-party integrations should use optional extras
  and lazy imports with actionable missing-dependency errors.
- Prefer existing builders and extension points over new parallel abstractions.
- Training changes must preserve optimizer, scheduler, EMA, gradient accumulation,
  AMP, distributed sampling, and resume semantics unless the task explicitly changes them.
- For export and evaluation, treat preprocessing, class mappings, score semantics,
  and model provenance as part of the model's interface.

## Scope and artifacts

- Put durable agent-only handoffs in `.agents/notes/` using its standard format;
  keep temporary work in ignored `.agents/local/`. Do not scatter session notes
  through `docs/`, `dev/` or the repository root.
- Commit agent-only instructions and notes separately with an `agent:` prefix.
  Source, tests, CI changes and developer-facing docs use separate normal commits,
  even when authored by an agent. Inspect explicitly staged paths before committing.
- Documentation-only agent changes need link/content and diff checks, not model
  tests. Executable helpers, workflows and mixed changes require affected checks.
- Treat `publication/` as reproducible research: preserve scripts, inputs, seeds,
  and recorded outputs unless the task specifically calls for changing them.
- Keep downloaded models, datasets, credentials, and generated outputs out of commits.
- The sibling `../mini_metrics` checkout may inform integration work; do not assume it
  is installed or make package imports depend on that local filesystem layout.
- When a workflow proves reusable, document it in `dev/` or a focused repository
  skill. Skills should point to shared commands instead of duplicating CI logic.
- The focused maintenance workflow is in
  `.agents/skills/mini-trainer-maintenance/SKILL.md`.
- Finish with the concrete changes, validation results, and remaining limitations.
