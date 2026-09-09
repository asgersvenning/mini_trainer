---
name: mini-trainer-maintenance
description: Maintain mini_trainer with compatibility-focused code inspection, installed-package validation, and checkpoint regression checks. Use for repository refactors, packaging, CI, and training-state maintenance.
---

# mini_trainer maintenance

Read the root [AGENTS.md](../../../AGENTS.md) and select the affected contracts before editing.
Use the shared commands in [dev/README.md](../../../dev/README.md); do not copy their
implementation into a new harness or implicitly sync the working environment.

## Choose checks by boundary

- Packaging or optional dependencies: `bash dev/check-wheel.sh` builds and installs
  a wheel with core CPU dependencies, then checks it outside the checkout. An editable
  installation with all extras cannot establish minimal-install compatibility.
- Checkpoint or training state: inspect `train.py` restoration and `trainer.py` saving
  together. Run `tests/training/test_checkpoint_contract.py` plus the existing training and
  CPU DDP integration tests. Compare predictions and state, not only file existence.
- Data refactors: preserve train/inference worker caps, batch sampling, cache overrides,
  class ordering, and label shape/device behavior. Choose focused tests from `tests/`.
- Imports: read the contracts in `pyproject.toml` and use the static harness; no model
  construction is needed to validate dependency direction.

## Interpret evidence

Deterministic continuation requires the original total epoch budget, fixed data order,
no stochastic augmentation/dropout, and compatible EMA settings. The existing checkpoint
format does not save random-generator or sampler state, so passing this controlled test
is not a guarantee of identical continuation for arbitrary stochastic training.

Read the known-failure notes in [docs/roadmap.md](../../../docs/roadmap.md) before
interpreting expected failures. Keep strict expected-failure regressions visible and
remove their markers when the underlying behavior is fixed; do not broaden markers to
hide unrelated failures. CPU float32 tests do not establish CUDA or AMP correctness.

The backbone compatibility utility can mutate the blacklist and download models.
Use it only when those actions belong to the requested maintenance task.
