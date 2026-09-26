# MAMBO V3 installed-candidate qualification

25 September 2026. **Ready for publication review; nothing published.**
Candidate source: `97521aca80b714a3c728c5782a9fa2f13eed652e`.
Local artifacts: `local-evidence/mambo-v3-release-candidate-final/`.

The review set contains the standalone deployment wheel/source distribution,
matching training wheel, compressed offline model bundle, public evidence,
qualification records, `release-candidate.json` and `SHA256SUMS`.
Weights use CC BY-NC-SA 4.0; adapter code remains MIT. No tags, uploads or public
pointers were changed.

## Verified boundaries

| Check | Evidence |
|---|---|
| Clean ONNX-only install | Python 3.13.7, ONNX Runtime 1.30.0, NumPy 2.5.3, Pillow 12.3.0; torch and mini_trainer absent. |
| Offline and relocation | Read-only relocated bundle, Python socket connections blocked, prediction/embedding API and streaming CLI with default TTA; bundle bytes unchanged. |
| Automatic downloads | Installed global-default predictor retrieved public ERDA assets, produced predictions and unit embeddings, then reused them with downloads prohibited. |
| Native CPU | Installed PyTorch 2.14.0+cu130; four real images, global/regional/custom lists, prediction and embedding modes. |
| Laptop CUDA | RTX 3080 Ti Laptop; installed Torch 2.14.0+cu130 and ORT GPU 1.30.0; same four-image contracts with default TTA. Both ONNX graphs used optimized profiles without failed probes. |
| CLI ownership | With both release wheels installed, only mambo-v3 registers mambo_predict, targeting mambo_deploy.cli:run. |
| Source contracts | Global defaults, streaming window, incremental output ordering, embedding file shape, error cleanup, cache/offline behavior and output provenance are covered by focused tests. Nine cache/download tests passed after the final metadata update. |
| Packaging/static | Minimal installed training-wheel check passed; standalone wheel and sdist built; Ruff/format and import contracts passed. |
| Final notice-bearing wheel | Installed outside the checkout; all 17 payload files match its archive. Offline metadata bootstrap, global default, sole CLI owner, license identifiers/text and expanded/embedded bundle agreement passed. |

## Artifact identities and evidence reuse

| Artifact | SHA-256 |
|---|---|
| `mambo_v3-0.3.0-py3-none-any.whl` | `6e84e3ee5478771926bf4e18eb92c695ac44921f7d306ad522f4ec2f0b9fc511` |
| `mini_trainer-0.3.0-py3-none-any.whl` | `0cf47254f962803b786c50310ca4ee40fe4710beaa0a9534386b73cd08f6883e` |
| Bundle `release.json` | `3ec2f0483f3412f11fdae4f33b8f95cabb29032cf060df79733d7854b236974e` |

The manifest covers 129 files, including qualification records. `SHA256SUMS` also
covers the manifest itself. All inventory hashes passed; all 45 compressed bundle
files match the expanded bundle. The source distribution's module payloads match
the wheel.

Execution qualification used source `0bfb5d7a7ac04afdaa392a8494191d8a160954ac`.
`qualification/validation.json` binds those retained reports to the current wheels:
the training wheel is identical, and deployment runtime payloads are identical.
Only the embedded descriptor, README in wheel metadata and checksum record changed.
All model files, preprocessing and class-list bytes remain unchanged. Current
installed metadata checks cover the new notices/license and selected preset policy.
This is evidence reuse, not a claim that inference was rerun after documentation edits.
The earlier wheels and raw reports remain under
`local-evidence/mambo-v3-release-candidate/`.

## Scope and provenance limits

Existing Flemming, global-lepi, laptop and B200 evidence remains the source for
published model comparisons. No quality or speed campaign was rerun. Four-image
runtime checks do not establish general accuracy, embedding quality or arbitrary
platform compatibility. Windows/macOS and other accelerators were not newly qualified.

The checkpoint and best epoch 30 are verified. The retained materials do not identify
the exact training Git revision; the packaging checkout is not a substitute.
Preparation source reconstructs the starting checkpoint recipe: torchvision DEFAULT
EfficientNetV2-S (ImageNet-1K) and a new normalized hierarchical head, seed 42.
The original starting-file hash and run-specific preparation manifest are absent.
The model card and notices disclose this limitation without asserting a verified
initial-file identity.

All 25 geographic lists retain evaluated membership: updated lists require at least
3 regional / 25 global metadata rows; legacy lists are unchanged. Reconstruction
from the pinned Parquet confirmed all counts and memberships. Any future distinct-
observation counting or membership change requires an identified preset revision.

[Publication and rollback handoff](publication.md) describes the separate human
publication step. Package ownership and authenticated uploads are publisher actions;
no public release was created by this preparation.
