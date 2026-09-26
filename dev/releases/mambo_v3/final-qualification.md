# MAMBO V3 installed-artifact qualification

Publication remains an owner action. The current preparation path builds the
candidate from a clean committed checkout, qualifies its installed wheels, and
only then stages public files. The authoritative source revision and artifact
hashes are the candidate's `release-candidate.json` and `SHA256SUMS`.

## Current candidate records

Local output: `local-evidence/mambo-v3-publication-candidate/`. The corresponding
Action output is `mambo-v3-candidate`; the explicit upload set is
`mambo-v3-publication`. A manifest with `qualification != "passed"` cannot be
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
