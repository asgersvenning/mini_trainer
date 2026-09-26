# MAMBO V3 deployment freeze

26 September 2026. Publication preparation is implemented; installed-candidate
records and hashes gate the publisher handoff. Packages, artifacts and tags have
not been published. The selected weight license is **CC BY-NC-SA 4.0**; adapter code
remains MIT. Final artifact identities and installed checks are recorded in
[final qualification](final-qualification.md).

## Current publication preparation

- Integrated the `minitrainer` distribution rename, retaining `mini_trainer` imports.
  Both lockfiles preserve dependency versions. Isolated installed-wheel imports,
  CLI, training, checkpoint reload and inference passed.
- Training publication now accepts `packages/minitrainer/vVERSION` release events only;
  manual dispatch prepares artifacts without publication. Model and demo publication
  have separate workflows, per-product environment gates and verified upload inventories.
  [Hierarchical branch/tag routing](../README.md) is shared across releases.
- The CPU Space uses the release API with runtime, scope, custom-list, TTA and
  top-K controls and names for all 17,212 taxa. Runtime reuse is qualified on a
  real image; final installed-candidate qualification is recorded with its artifacts.
  Browser reuse is assessed separately below.
- Publication documentation, explicit model/Space staging and installed qualification
  are implemented. Follow the ordered handoff for account setup and public release;
  inspect the candidate records before approving its artifacts.

## Browser scope

The existing prototype at `b174426` packages a WASM embedding explorer, including
prototype coordinates, a separate worker and a different EXIF/alpha policy. It
does not already supply the release preset/TTA interface. Integrating it would
require another browser preprocessing and configuration qualification effort.
This release preparation therefore delivers the Python-backed Space; WebGPU
remains an explicitly unqualified follow-up, reusing that prototype where useful.

## Release contract

- Distribution `mambo-v3`, version `0.3.0`; Python import `mambo_deploy`.
  Maintenance releases retain the V3 trained model; future generations use a
  separate package. Pin the package version for reproducible application builds.
- PyTorch and standard ONNX; global (`full`) scope by default; legacy and updated
  regional presets, custom class lists, independent rank predictions and optional
  unit embeddings. Automatic precision and optional `rotation30_pad25_3` TTA.
- The deployment package alone owns `mambo_predict`. API/CLI global defaults,
  bounded streaming and incremental CLI output are qualified. The native V2
  compatibility facade retains its native/CUDA defaults.
- All 25 geographic lists retain their evaluated membership. Updated lists use
  at least 3 regional and 25 global metadata rows; legacy lists are unchanged.
  Deduplication or membership changes require a future preset revision.
- No additional throughput campaign, retraining, quantization or exhaustive platform
  qualification. Shared-core changes require a separate branch and reviewed merge.

## Completion evidence

| Requirement | Authoritative record |
| --- | --- |
| Installation, inputs/outputs, meaningful defaults, V2 migration and changelog | [Deployment README](../../../deployment/README.md), [integration details](../../../docs/mambo-integration.md) |
| Geographic filters, counts and reconstruction | [Preset catalogue](../../../docs/model-presets.md), `preset-definitions.toml`, `preset-manifest.toml`; pinned-Parquet reconstruction leaves all lists unchanged |
| Trained assets, preprocessing, vocabulary and notices | Bundle `release.json`, `MODEL_PROVENANCE.toml`, `NOTICES.md`, `MODEL_LICENSE.txt`, `CODE_LICENSE`; immutable per-file hashes |
| Installed runtime, offline/download and API/CLI checks | [Final qualification](final-qualification.md), candidate `qualification/validation.json` and linked retained reports |
| Flemming and complementary in-domain metrics | README figures and linked evidence; mini_metrics calibration/reporting separation, all ranks, both confidence settings and full/support >5 metrics |
| Laptop, CPU, B200 request/streaming comparisons | README figures, [current HPC evidence](../../../docs/mambo-hpc-evidence.md); only measured points updated, historical boundaries retained |
| Reusable protocols and evidence | [Evidence policy](evidence-policy.md), public tables/provenance and retained private prediction/confidence archives |
| Browser and other integration opportunities | README's Beyond Python section; described as integration paths, without claiming untested platform support |
| Distribution assets and publication procedure | Local candidate wheels, source distribution, bundle/archive, evidence, manifest and checksums; [publication and rollback handoff](publication.md) |

## Provenance limits

The trained checkpoint, original export files and best epoch 30 are verified.
The training Git revision and original `initial_seed42.pt` bytes/hash were not
retained. Preparation source reconstructs that file's role: torchvision DEFAULT
EfficientNetV2-S (ImageNet-1K) plus a new hierarchical head, seed 42, saved before
production training. It is not a second trained MAMBO model. The model card and
notices distinguish this source reconstruction from a verified starting-file
identity; no training revision or hash has been invented.

## Validation and handoff

Reuse qualified runtime execution when wheel runtime payloads, weights,
preprocessing and class lists are identical. Check rebuilt metadata and installed
payloads after notice-only changes; do not repeat model quality or speed campaigns.
Static/import-contract checks, focused deployment contracts and the minimal
installed training-wheel check are recorded in the qualification history.

The publisher reviews the concrete artifacts, notices, limitations and source
commit using [publication.md](publication.md). Tagging, package upload, model-asset
upload and public-pointer promotion are separate actions and remain unauthorized.
