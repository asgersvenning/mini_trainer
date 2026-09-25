# MAMBO deployment freeze preparation

Status: preparation, 25 September 2026. Throughput investigation is closed for this
release. The measured runtime baseline is `503de96`; later changes update experiment
setup and evidence. This document identifies the remaining consolidation work;
it does not declare the candidate frozen or published.

## Retain the measured behavior

Keep PyTorch and standard ONNX, presets/custom lists, independent rank predictions,
embeddings, automatic precision, optional `rotation30_pad25_3` TTA, and bounded
streaming. Keep package version 0.3.0 and the current model artifacts/preset identities
while consolidating. Quantization and further HPC scalability work are deferred.
Shared `mini_trainer` changes still require a feature/fix branch and reviewed merge.

## Consolidation sequence

1. **Audit the deployment boundary.** Keep runtime responsibilities in the existing
   modules: bundle/download validation; preprocessing/augmentation; backend execution
   in `predictor`; hierarchy/results; and streaming/transfers/result worker. Check
   unused paths and duplicated work against both request and streaming callers.
   Remove only demonstrably dead or redundant code. Preserve result ownership,
   shutdown/error handling and optional-runtime imports.
   Public API/CLI changes remain possible when they remove a concrete integration
   obstacle; document their V2 migration impact and validate the affected contract.
   Do not treat the current public surface as already frozen.
   Profiling and experiment setup remain under `dev/releases/mambo_v3`, outside the
   deployment wheel. Do not redesign the pipeline during freeze preparation.
2. **Consolidate integration documentation and workflow.** Qualify a minimal path:
   install one runtime → construct a predictor or invoke the CLI → supply ordinary
   images → consume ordinary records. No training checkout, dataset metadata,
   campaign config or GPU setup should be needed for ONNX/CPU. Check the documented
   paths with original images and a custom class list, predictions/embeddings, an
   existing application environment and an offline bundle. Explain scope/runtime/TTA
   decisions; leave tuning and kernel diagnostics in linked details.

   Implemented integration decisions: the `mambo-v3` distribution alone owns
   `mambo_predict`; API/CLI default to global; CLI writes batches incrementally and
   publishes only complete outputs; the streaming read-window default grows with
   batch size; the native extra includes `timm`. Arrays retain explicit CHW input
   to avoid guessing ambiguous layouts. These changes need final installed-bundle
   qualification below.
   Retain V2 entry-point and format compatibility where promised; distinguish that
   from identical vocabularies, scores or embeddings.

   **Documentation ownership:** The [deployment README](../../../deployment/README.md)
   owns installation, configuration and integration examples. Preset scope belongs
   in [the catalogue](../../../docs/model-presets.md); complete numbers/provenance
   belong in the [Flemming](../../../docs/mambo-deployment-evidence.md),
   [in-domain](../../../docs/mambo-indomain-evidence.md) and
   [current HPC](../../../docs/mambo-hpc-evidence.md) evidence pages. Keep performance
   figures and the configuration table in the README, long evidence tables in linked
   pages, and historical diagnostics out of the integration path. Mark old measurements/instructions as historical
   rather than erasing their provenance. Preserve CPU/GPU, request/streaming and
   V2/V3 comparison categories; update only measured values, and never imply older
   CPU/laptop or smaller-batch results were rerun.
3. **Refresh candidate metadata once documentation settles.** The checked-in
   `deployment/mambo_deploy/default_bundle.json` embeds an older README and model
   card. The card still describes full task metrics as outstanding. Update the
   builder's card text and regenerate with the existing `build_bundle.py` and
   `package_download_metadata.py` workflows. Verify preset files, source URLs,
   all file hashes, and version/model/artifact identities. Documentation changes
   alter the descriptor-derived cache revision; record the final revision rather
   than repeatedly regenerating it during editorial work. Ensure links work from
   the actual distribution location as well as the repository.
4. **Build and qualify the final installed candidate.** Build the deployment and
   matching training wheels once. Follow [deployment qualification](deployment-qualification.md)
   for ONNX-only CPU installation, relocated/read-only offline bundle, API/CLI,
   presets/custom lists, predictions/embeddings and existing CUDA environments.
   Include streaming ownership, early close and error propagation in affected
   tests. Run static checks and the required wheel check. Reuse unchanged model
   quality evidence and the completed B200 smoke; do not launch another quality,
   throughput or hardware campaign without a concrete compatibility failure.
5. **Record the freeze manifest.** Capture source commit, wheel hashes, bundle
   revision/hashes, versions, tested runtime environments, known limitations and
   publication/rollback assets. Confirm training revision/best-epoch provenance
   and weight/data redistribution notices, which remain open in the current model
   card. Qualify only the OS/runtime combinations actually checked; additional OS
   support is not implied. Tagging, uploading and promotion are separate from
   preparing these reviewable assets.

## Evidence already ready

- Full Flemming and in-domain comparisons through `mini_metrics`, including rank
  metrics, calibrated/unthresholded results, coverage and support >5.
- Latest full-B200 four-variant request/streaming timings, memory and provenance
  now linked from the README. Earlier CPU/V2 and laptop evidence remains labelled.
- Current focused runtime/static checks and targeted CUDA preprocessing evidence
  recorded in the [pipeline report](pipeline-probe.md).
- Deterministic timing figure generated by `hpc_speed_report.py`; no new benchmark
  execution is needed to reproduce it.

Freeze completion requires the final installed artifacts and documentation to agree.
Historical qualification is supporting evidence, not a substitute for checking the
final wheels and embedded metadata.

## Integration increment evidence — 25 September 2026

The package is now model-generation-specific (`mambo-v3`, Python import
`mambo_deploy`), version 0.3.0. The root training package no longer registers the
same executable. Existing candidate installations need a fresh environment (or
removal of `mambo-deploy`) to avoid two distributions owning the import directory.
No published package was changed.

Focused contracts: 96 passed, four GPU-dependent skips, across the initial run
and correction of a generator stub in the new read-window test. Static checks and
the required minimal installed training-wheel check passed. The renamed deployment
wheel builds; final installed ONNX/native bundle checks remain ahead. No performance
or quality evaluation was rerun.

The automatic model cache now stores verified weight bytes by SHA-256 and reuses
them across metadata revisions (hard links where possible, ordinary copies
otherwise). Offline mode can materialize packaged metadata but still forbids
network downloads. Nine focused cache/download tests pass.
