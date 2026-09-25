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
   batch size; the native extra selects the matching training-package series. Arrays retain explicit CHW input
   to avoid guessing ambiguous layouts. These changes passed the installed-bundle
   qualification recorded below.
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
3. **Refresh candidate metadata after any final owner decisions.** The checked-in
   descriptor now embeds the current README, maintained model card, notices and
   verified epoch/checkpoint provenance. Model identity is `MAMBO_v3`, artifact
   revision 3, distribution `mambo-v3` and default preset `full`. The model-weight
   license and initialization lineage remain explicitly unresolved. Regenerate
   with `build_bundle.py` and `package_download_metadata.py` after resolving them. Verify preset files, source URLs,
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
   publication/rollback assets. The verified training log confirms best epoch 30; the exact training Git
   revision is absent from the retained checkpoint/config/log. Do not substitute
   packaging-time revision. Resolve weight-license and initialization notices. Qualify only the OS/runtime combinations actually checked; additional OS
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
wheel builds; the final installed ONNX/native bundle checks are recorded below. No performance
or quality evaluation was rerun.

The automatic model cache now stores verified weight bytes by SHA-256 and reuses
them across metadata revisions (hard links where possible, ordinary copies
otherwise). Offline mode can materialize packaged metadata but still forbids
network downloads. Nine focused cache/download tests pass.

## Candidate assembly

[Publication handoff](publication.md) describes the prepared artifacts and the
separate human publication step. `prepare_candidate.py` builds only local outputs
from a clean committed checkout. [Evidence policy](evidence-policy.md) defines
what subsequent releases retain and when older results can be reused.

`model-provenance.toml` records checksum-verified console and epoch-summary sources.
The log is 60,812,963 bytes (retrieved in full after a truncated first read was
correctly rejected by its checksum). It records best epoch 30. Weight licensing
and upstream initialization attribution require owner input; questions are pending.
No missing source identity has been invented.

## Final installed candidate

The [installed-candidate record](final-qualification.md) now covers clean ONNX CPU,
read-only offline API/CLI with TTA/embeddings, actual automatic ERDA downloads and
offline reuse, native CPU, and native/ONNX laptop CUDA. Installed package payloads
were compared with the built wheels; the two-package install has one CLI owner.
No performance/quality campaign was rerun. Runtime source is `0bfb5d7`.

Artifacts are prepared in `local-evidence/mambo-v3-release-candidate/`; publication
remains prohibited. Model-weight licensing and initialization attribution are
pending owner decisions. Training source revision is explicitly unknown in the
retained materials. After those decisions, refresh final notices/metadata and the
artifact inventory; do not substitute a historical checkout or invent permission.

## Final preparation audit

| Requirement | Current evidence / remaining work |
| --- | --- |
| Simple distribution and stable model selection | `mambo-v3` distribution, embedded immutable asset hashes, supplied-wheel installation and future PyPI commands; global default. No repository checkout needed for consumer installs. |
| API/CLI and V2 migration | Installed checks above; README documents inputs/outputs, independent rank labels, embeddings, TTA, region/custom lists and migration. Sole CLI owner and bounded output writing verified. |
| Quality and speed presentation | README retains Flemming, in-domain, laptop and HPC figures; linked pages identify protocols and historical/current timing boundaries. No new measurements required. |
| Preset definitions | Selected V3 row-count policy; all 25 lists reconstructed unchanged from pinned metadata. Updated definition hashes and embedded descriptor verified. |
| Offline/download behavior | Installed automatic download/offline/relocation checks plus nine cache tests. Download checker now requires an empty cache. |
| Reusable evidence and portability limits | `evidence-policy.md`, model card and linked evidence; browser integration described as a path, not a tested platform. |
| Packaging and handoff | Local wheels, source distribution, bundle, checksums and qualification records; `publication.md` covers human publication and rollback. |
| Final notices and immutable candidate | Pending model-weight license and initialization attribution. Then refresh artifacts and verify only the changed metadata/installation boundary, reusing identical runtime evidence. |

The refreshed preset metadata is staged in source and
`local-evidence/mambo-freeze/preset-policy-bundle/`; the earlier installed wheel
identities remain historical qualification evidence. Do not describe those wheel
hashes as the final notice-complete release. No publication was performed.
