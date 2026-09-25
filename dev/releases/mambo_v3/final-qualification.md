# MAMBO V3 installed-candidate qualification

25 September 2026. Prepared runtime source: `0bfb5d7a7ac04afdaa392a8494191d8a160954ac`. No publication, tag or public pointer change was performed.

Local artifacts: `local-evidence/mambo-v3-release-candidate/`. The directory contains deployment wheel/source distribution, matching training wheel, a 406 MiB compressed offline bundle, public evidence and a complete artifact inventory. Qualifications are retained in its `qualification/` directory.

## Verified boundaries

| Check | Evidence |
|---|---|
| Clean ONNX-only install | Python 3.13.7, ONNX Runtime 1.30.0, NumPy 2.5.3, Pillow 12.3.0; torch and mini_trainer absent. |
| Offline and relocation | Read-only relocated bundle, Python socket connections blocked, prediction/embedding API and streaming CLI with default TTA; bundle bytes unchanged. |
| Automatic downloads | Installed global-default predictor retrieved public ERDA assets, produced predictions and unit embeddings, then reused them with downloads prohibited. |
| Native CPU | Installed PyTorch 2.14.0+cu130; four real images, global/regional/custom lists, prediction and embedding modes. |
| Laptop CUDA | RTX 3080 Ti Laptop; installed Torch 2.14.0+cu130 and ORT GPU 1.30.0; same four-image contracts with default TTA. Both ONNX graphs used optimized profiles without failed probes. |
| CLI ownership | With both release wheels installed, only mambo-v3 registers mambo_predict, targeting mambo_deploy.cli:run. |
| Installed bytes | All installed wheel payload files except installer-rewritten RECORD match the prepared wheel archive bytes. |
| Source contracts | Global defaults, streaming window, incremental output ordering, embedding file shape, error cleanup, cache/offline behavior and output provenance are covered by focused tests. |
| Packaging/static | Minimal installed training-wheel check passed; standalone wheel and sdist built; Ruff/format and import contracts passed. |

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| `mambo_v3-0.3.0-py3-none-any.whl` | `684b4f12bb3183390ccbbf0b3f510bef54c3ded08c4e0cec64a77f16d4e3efac` |
| `mini_trainer-0.3.0-py3-none-any.whl` | `0cf47254f962803b786c50310ca4ee40fe4710beaa0a9534386b73cd08f6883e` |

The manifest hashes the expanded bundle, archive, wheels, source distribution and public evidence. `qualification/validation.json` binds the test records to the installed wheel identities. Qualification report hashes are retained there; raw input paths remain in local evidence only.

## Scope and remaining decisions

These are execution and packaging checks, not new accuracy or speed experiments. Existing Flemming, global-lepi, laptop and B200 evidence remains the source for published model comparisons. Four images do not establish general accuracy, embedding quality or cross-platform correctness. Windows/macOS and other accelerators have not been newly qualified.

The checkpoint and best epoch 30 are verified. The retained training material does not identify the exact training Git revision; the documented packaging checkout is not a substitute. The initialization checkpoint lineage and required attribution remain an owner question. A model-weight license is also awaiting designation. These are explicit finalization blockers, not grounds to alter the validated runtime or rerun the performance campaign.

The preset row-count policy remains the documented candidate rule (3 regional / 25 global for updated lists, legacy membership unchanged). Final packaging must make its selected status explicit without changing membership or silently deduplicating observations.

Final owner-driven documentation/notice changes will require refreshed bundle metadata and artifact hashes. Reuse these runtime checks for byte-identical code and weights; verify the rebuilt metadata/installation boundary instead of rerunning model evaluation.
