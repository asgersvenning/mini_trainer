# Quantization artifact retention

The repository's historical benchmark notes describe experiments, not a guarantee
that every temporary binary remains in the checkout. Cleanup consolidates local
evidence under ignored `local-evidence/quantization-2026-09-09/`; no model, dataset
or generated result is committed by this operation.

The 2026-09-09 cleanup verified 5,629 archived evidence files and 60 retained model/
bundle files, then removed 75 temporary directories and completed pytest outputs.
The archive occupies about 0.64 GiB and retained bundles 2.67 GiB. Approximate net
space reclaimed was 34.46 GiB; about 99.37 GiB was available immediately afterward.
The reorganized suite retained all 737 cases: 574 passed, 162 skipped and one
known EMA expected failure. Detailed collection/check logs are retained in
`local-evidence/quantization-2026-09-09/validation/`.

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

The working `.venv`, model download caches, original `examples/` datasets, and
`publication/` research files are preserved. The separate temporary TensorRT and
ONNX Runtime GPU installations are disposable; prepare the optional runtime
explicitly on the target using the [inference guide](../dev/benchmarks/inference.md).

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
