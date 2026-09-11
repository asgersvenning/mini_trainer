# Parallel local development

Use one Git worktree per active feature. Worktrees have independent tracked files,
indexes and checked-out branches; commits, refs and Git configuration are shared.
The current prototype exploration starts from `quant`, whose diagnostic functions
are the reference for that work. Choose the base explicitly for other features.

From the main checkout:

```bash
git worktree list
git worktree add -b feature/my-feature .worktrees/my-feature quant
```

`/.worktrees/` is ignored on branches carrying this guide. For older branches,
add `/.worktrees/` once to the main checkout's `.git/info/exclude` (local only,
shared by worktrees). Do not copy uncommitted edits into another branch unless
they belong to its task. Existing main-checkout changes stay in place.

## Reuse dependencies without changing them

When both branches use the same dependencies, a worktree can reuse the existing
environment through a `.venv` symlink. This shares installed packages; it does
not provide independent environments. No branch using that environment should
install, remove or synchronize packages while others are using it.

From the main checkout, before entering the new worktree:

```bash
ln -s "$PWD/.venv" .worktrees/my-feature/.venv
cd .worktrees/my-feature
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -c 'import mini_trainer; print(mini_trainer.__file__)'
bash dev/check.sh static
bash dev/check.sh test tests/utils/test_dendrogram.py
```

The import check must point into the new worktree. Set `PYTHONPATH` separately
in each worktree's shell: it also gives Python subprocesses the correct source
root. Prefer `.venv/bin/python -m ...` over installed console scripts, which can
otherwise resolve an editable installation in the original checkout.

Do not run `uv sync`, `uv pip install` or ordinary `uv run` against a shared
environment. Use its executables directly, or `uv run --no-sync`. The shared
environment remains subject to the README's explicit PyTorch-backend rules.

If a branch needs different dependencies, first unlink only its `.venv` symlink
(`test -L .venv && unlink .venv`), then create a separate environment using the
README's explicit installation command and intended CPU/CUDA backend. Never
delete the symlink target. Separate environments need their own package/import
verification; CPU validation does not establish GPU correctness.

## Keep experiments isolated

- Use a separate terminal/session per worktree and check `pwd` and
  `git branch --show-current` before edits, checks or commits.
- Keep generated reports in that worktree's ignored `tmp/`. Pass existing large
  checkpoints and datasets by absolute path rather than copying them into Git.
- Do not run multiple experiments against the same output directory or change a
  shared dependency environment concurrently. GPU, RAM and disk resources remain
  shared even though the source trees are separate.
- Stage explicit paths and inspect the staged diff. A worktree does not protect
  another worktree from operations on shared branch refs or repository config.
- Commit focused work before merging or cherry-picking it. Do not switch the
  main checkout away from ongoing work just to inspect a feature branch.

When a feature is integrated and its worktree has no changes or needed outputs:

```bash
git worktree list
git -C .worktrees/my-feature status --short
git worktree remove .worktrees/my-feature
```

Inspect and retain ignored reports before removal. Do not use `--force` to
discard work. Branch deletion is a separate decision after integration.

Agents must also follow the [worktree coordination rules](../.agents/rules/worktrees.md).
