# Quantization cleanup handoff

Status: completed
Updated: 2026-09-10
Scope: local quantization evidence retained after the 2026-09-09 cleanup
Related: [artifact restore guide](../../docs/quantization-artifacts.md), original report at commit `f5c69e7cab2bfde8a5467026b293858b93e628f9`

## Context and decision

The cleanup consolidated local evidence under ignored
`local-evidence/quantization-2026-09-09/`. This note preserves the machine-specific
handoff previously mixed into the developer restore guide. Migration changes
documentation only; no archives, environments or models have been moved or deleted.

## Evidence and limits

The original report recorded:

- 5,629 archived evidence files and 60 retained model/bundle files verified.
- 75 temporary directories and completed pytest outputs removed.
- About 0.64 GiB of archived evidence and 2.67 GiB of retained bundles.
- About 34.46 GiB reclaimed and 99.37 GiB available immediately after cleanup.
- 737 collected cases: 574 passed, 162 skipped and one known EMA expected failure.
  Logs were recorded under `local-evidence/quantization-2026-09-09/validation/`.
- The working `.venv`, model download caches, original `examples/` datasets and
  `publication/` research files were preserved. Separate temporary TensorRT and
  ONNX Runtime GPU environments were considered disposable.

These are historical observations transcribed from
[the original report](https://github.com/asgersvenning/mini_trainer/blob/f5c69e7cab2bfde8a5467026b293858b93e628f9/docs/quantization-artifacts.md),
not current disk-space, archive-integrity or test-suite claims. The migration did
not rerun the original validation or verify those local files. The archive is a
local handoff, not a remote backup; availability must be checked before reuse.

## Next actions

None for the completed cleanup. Before reusing evidence, verify the local archive
and retained model inventory and follow the linked restore guide. Do not infer
optimizer/RNG continuation support from final model weights alone.
