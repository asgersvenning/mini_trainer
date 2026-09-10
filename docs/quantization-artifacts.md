# Quantization artifact retention

The repository's historical benchmark notes describe experiments, not a guarantee
that every temporary binary remains in the checkout. Cleanup consolidates local
evidence under ignored `local-evidence/quantization-2026-09-09/`; no model, dataset
or generated result is committed by this operation.

Historical cleanup counts, disk-space observations and the test snapshot are
preserved in the [dated agent handoff](../.agents/notes/2026-09-09-quantization-cleanup.md).
They do not establish current artifact availability or test status. This guide
covers the retained layout and restore procedure.

## Retained locally

- `reports-inputs.tar.gz`: reports, metrics/predictions, prepared NPZ inputs,
  experiment scripts/configurations, logs, traces and small synthetic image files.
  Original `tmp-*` relative paths are preserved inside the archive.
- `retained/`: final `last.pt` models for the flat/hierarchical three-seed studies
  and the longer hierarchical pair; floating/materialized and calibrated ONNX
  deployment bundles for both representative heads. Their original relative
  directory structure is preserved.
- `inventory.json`: archived/retained paths, sizes and SHA-256 digests, plus the
  discarded-file inventory and runtime-directory totals. Archive content and
  retained models are verified before their temporary source directories are removed.
- `README.md`: local cleanup totals and restore instructions.

Prepare optional GPU runtimes explicitly on the target using the
[inference guide](../dev/benchmarks/inference.md); the archive does not include a
portable working environment.

## Removed as disposable

Intermediate and duplicate checkpoint/model exports, synthetic large-head binary
models, laptop-specific TensorRT engines, temporary package installations, and
completed pytest outputs. Final optimizer-resume snapshots are not retained in
this compact handoff: exact continuation from discarded experiments requires
regenerating the run. Retained final model weights support inference/quality
re-evaluation, not a claim of complete optimizer/RNG restoration.

## Restore selected evidence

Run from the repository root. List the archive before choosing paths:

```bash
tar -tzf local-evidence/quantization-2026-09-09/reports-inputs.tar.gz
tar -xzf local-evidence/quantization-2026-09-09/reports-inputs.tar.gz \
  tmp-cpu-deployment-flat tmp-cpu-deployment-hierarchical
```

This restores reports and their child evidence for archival/review without engines
or model weights. To rerun inference, restore the corresponding inputs and copy
the selected bundle from `retained/` to the intended location. Rebuild TensorRT
engines on the target. Original reports keep original provenance paths and hashes;
do not rewrite them to make discarded binaries appear available.

The archive is a local handoff, not a remote backup or a live result-hosting service.
Transfer it deliberately with the required retained models when target access is
available. Public compact history contains only the selected reporting projection.
