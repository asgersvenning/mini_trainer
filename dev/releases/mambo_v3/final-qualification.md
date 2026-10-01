# MAMBO V3 installed-artifact qualification

Publication remains an owner action. The current preparation path builds the
candidate from a clean committed checkout, qualifies its installed wheels, and
only then stages public files. The authoritative source revision and artifact
hashes are the candidate's `release-candidate.json` and `SHA256SUMS`.

## Nemo maintenance 0.3.1 qualification (1 October 2026)

The maintenance implementation lives on `release/mambo-v3`; it retains the
MAMBO_v3 weights, ONNX graphs, class order, preprocessing recipe and presets.
Local scratch evidence is under `.agents/local/nemo/` in the maintenance checkout.
Candidate manifests remain the authority for exact source and wheel hashes.

- Full static/runtime harness: **960 passed, 168 skipped, one expected failure**.
  The shared environment lacked installed `mt-trainer` metadata; the successful
  run used an isolated installed wheel alongside the checkout, without changing
  the shared environment. The earlier incomplete-environment run is not passing evidence.
- Follow-up native regressions cover independent checkpoint loading, ACS metadata
  access, stored masks/full vocabulary and head-only legacy backbone restoration.
  The optional BioCLIP dependency was unavailable in that initial run; see the
  actual Meghan qualification in the follow-up below.
- Minimal installed training wheel: native prediction without `mambo_deploy`,
  plus existing training/reload/CLI checks passed on CPU.
- Isolated ONNX/Hub wheel: no Torch/trainer dependency; real Hub commit
  `9dcb52c37f2d912915e76d74a7cd4f7074bf7930` loaded and predicted, then reloaded
  offline. Hugging Face Hub 2.0 and ONNX Runtime 1.30 exercised shared-cache blobs
  and colocated external weights. The new model card parsed through `ModelCard`.
- Four retained Flemming images: Torch on RTX 3080 Ti Laptop and ONNX CPU agree
  on top-1 at all ranks for global, Europe and custom selections. Synthetic
  four-image tests separately passed TTA, masks and embedding contracts on both.
- A fixed-input GPU comparison retained identical logits and embeddings across
  the original and delegated Torch paths; confidence differences were below
  `9e-8`. This is bounded regression evidence, not a new throughput or accuracy claim.

Publication of 0.3.1 is pending. The original release remains immutable. Use
`packages/mt-trainer/v0.3.1` followed by `models/mambo-v3/v0.3.1`; the
[maintenance publication instructions](publication.md#nemo-maintenance-031)
identify the workflow and unchanged-model checks. The live Hub currently retains
its original card until maintenance publication advances it to a new commit.

## Pre-publication review corrections (1 October 2026)

The follow-up keeps publication pending. Candidates staged from `0bacea5` predate
these corrections and must be rebuilt from the final reviewed commit.

Native checkpoint recognition now uses tensor contents, architecture metadata and
class order rather than the filename or constructor argument. Its fingerprint is
recorded beside the immutable file checksum in `model-provenance.toml` and generated
into the native bootstrap. Class selection remains separate from model identity.
The native CLI preserves explicit checkpoint scope and uses the same Nemo
preprocessing as the Python predictor. The CLI retains the generic inference
runner and its explicit dtype/collector options.

ONNX embedding startup also prepares ordinary prediction by reusing that graph's
logit output. Hub asset selection follows runtime profiles, independent of origin
URLs. Regression coverage includes both behaviors, native local-file/dictionary
preprocessing, CLI output, and legacy BF16 preprocessing on an FP32 CPU model.

Actual Meghan European checkpoint qualification used `open-clip-torch==3.3.0`
and PyTorch `2.14.1+cpu` in an isolated environment. The head-only checkpoint
restored the pretrained BioCLIP-2 backbone and predicted a retained Flemming image:
3,014 active species, 12,632 full species, input size 512, embedding width 768.
Labels and confidences matched the existing model prediction utility with FP32
inputs; embeddings were finite. Compilation was disabled for this eager CPU check.
This is a loading/inference qualification, not a V2 accuracy or GPU qualification.

The full static/runtime harness passed: **969 passed, 168 skipped, one expected
failure**. Final focused release checks after the checkpoint-identity and CLI
regression refinements: **214 passed, 8 skipped**. Static/import contracts
and the minimal installed training wheel passed. On a retained real image, Nemo
CPU predictions and embeddings were identical for default, local-file and
checkpoint-dictionary loading. A real ONNX embedding session served both ordinary
prediction and embedding prediction with identical confidences and no second
session. Scratch reports are `meghan-qualification.json` and `loading-parity.json`
under `.agents/local/nemo/`.

## Current candidate records

Local output: `local-evidence/mambo-v3-publication-candidate/`. The corresponding
Action output is `model-candidate`; the explicit upload set is
`model-publication`. A manifest with `qualification != "passed"` cannot be
staged. Inspect these retained records rather than treating this page as a
completion marker:

| Record | What it establishes |
| --- | --- |
| `qualification/validation.json` | Exact installed wheel hashes, source identity, fixture type and completed contract checks |
| `qualification/download.json` | Actual pinned ERDA downloads through the installed ONNX-only package, global defaults, embeddings and offline cache reuse |
| `qualification/cpu-none.json` | PyTorch/ONNX global, regional/custom-list and embedding contracts on four images |
| `qualification/cpu-rotation30_pad25_3.json` | The same contracts with the default TTA recipe |
| `qualification/demo.json` | UI construction and real model calls through both backends, dynamic preset/custom/TTA controls, top-K and runtime reuse |
| `qualification/environment.txt` | Resolved runtime versions in the isolated qualification environment |
| `publication/*/publication.json` | Exact staged file inventory for GitHub, the Hub model or the Space |

The CLI check writes JSON, evaluation CSV and unit embeddings using TTA and cached
offline weights. The ONNX-only installation is checked before installing Torch or
the training package. Dependency consistency is checked after installing both
backends. Qualification leaves the working development environment unchanged.

## Evidence reuse and limits

The prior September 25 candidate (`97521ac`) qualified offline/read-only bundle
use and laptop RTX 3080 Ti CUDA execution, including embeddings and TTA. Its
runtime execution reference was `0bfb5d7`. Those records remain historical and are
not relabelled as measurements of the newly built packages. The package identity
migration preserves `mini_trainer` imports; the new `configure()` interface changes
scope/TTA without replacing model sessions. Current installed CPU qualification
covers that interface.

Trained weights, graphs, preprocessing, preset membership and evaluation policy
are unchanged. Existing Flemming/in-domain accuracy and laptop/B200 measurements
remain applicable within their documented boundaries; no full quality or speed
campaign is repeated. Synthetic CI images establish runtime contracts only; local
qualification can supply retained real images. Neither establishes new accuracy
or universal Windows/macOS/edge/CUDA compatibility.

Training revision and original initialization-file identity remain unavailable;
see the model card's provenance limits. The Space is qualified locally; public
hosting, credentials, registry installation and live cross-links require the
owner's publication and post-publication checks in [the handoff](publication.md).
