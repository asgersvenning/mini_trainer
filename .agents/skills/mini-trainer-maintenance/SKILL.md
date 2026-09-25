---
name: mini-trainer-maintenance
description: Maintain mini_trainer through repository cleanup, compatibility-focused refactoring, packaging, CI, and training-state validation.
---

# mini_trainer maintenance

Read the root [AGENTS.md](../../../AGENTS.md) and select the affected contracts before editing.
Use the shared commands in [dev/README.md](../../../dev/README.md); do not copy their
implementation into a new harness or implicitly sync the working environment.

## Repository reduction campaigns

Use this mode for repository-wide cleanup, not every small code change. Follow
[retention guidance](../../README.md#maintenance) and
[test guidance](../../rules/code-contribution.md#keeping-tests-useful).

- Start from tracked content and the user's content groups. Preserve a fixed
  revision, per-file character counts (including whitespace), exclusions and
  classification rules; apply identical rules to baseline and current content.
  Distinguish prose/source from generated evidence, figures, locks and outputs.
- Build a hierarchical map with short evidence-backed assessments of navigation,
  responsibility boundaries, signal-to-detail ratio and ongoing maintenance value.
  Use a coarse 1–5 scale: 1 obstructive, 2 substantial cleanup, 3 mixed,
  4 clear with minor issues, 5 lean and sufficient. Mark unreviewed areas;
  character counts and passing tests do not establish quality. Read the actual
  content and relevant consumers before scoring it or calling it a reduction
  target; inventories and filenames only identify where to inspect next.
- Select the highest-value bounded campaign from that map. Check consumers,
  generators, public contracts and provenance before retiring content. Prefer
  consolidation or removal over a new abstraction or archive of the same clutter.
- After each campaign, validate affected contracts, recount by the same groups
  and reassess the hierarchy, including newly exposed targets. Record remaining
  work explicitly; a successful bounded campaign is not repository-wide completion.

Keep detailed inventories, score rationales and campaign snapshots in ignored
`.agents/local/`. Track only reusable guidance and concise navigation that helps
contributors. Reduction is an auxiliary measure; preserve useful coverage and
reproducibility, and do not count minification or relocation as simplification.

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
