# Nemo installed-artifact qualification

Qualify each final candidate from a clean release commit before publication.
`release-candidate.json` identifies its source and artifact hashes; a candidate
with `qualification != "passed"` cannot be staged. Follow the
[publication procedure](publication.md) to build, qualify and stage it.

## Candidate records

The `model-candidate` Action artifact contains the following evidence. Local
preparation writes the same files to the chosen candidate directory.

| Record | What it establishes |
| --- | --- |
| `qualification/validation.json` | Installed wheel hashes, source identity and completed contract checks |
| `qualification/download.json` | ONNX-only installation, pinned downloads, embeddings and offline cache reuse |
| `qualification/cpu-none.json` | PyTorch/ONNX global, regional/custom-list and embedding contracts |
| `qualification/cpu-rotation30_pad25_3.json` | The same contracts with TTA |
| `qualification/demo.json` | Both runtimes and the preset, custom-list, TTA and top-K controls |
| `qualification/environment.txt` | Resolved versions in the isolated qualification environment |
| `publication/*/publication.json` | Exact staged GitHub, Hub model and Space inventories |

Qualification exercises the CLI and checks ONNX-only use before installing Torch.
It uses a separate CPU environment and never synchronizes the working `.venv`.
Synthetic CI images check runtime behavior; local qualification can use four
retained real images with `--dataset`. Neither is a new accuracy benchmark.

## Evidence reuse and limits

Maintenance preserves model weights, preprocessing, vocabulary and presets, so
existing [deployment evidence](../../../docs/mambo-deployment-evidence.md) and
[in-domain results](../../../docs/mambo-indomain-evidence.md) retain their original
scope. Reuse them without relabelling historical measurements as new runs.
CPU qualification does not establish GPU correctness or universal platform support.

Keep local diagnostic scripts, intermediate test totals and environment reports
outside tracked documentation. Final candidate records remain the authority for
release qualification. Training provenance limitations are recorded in
[model-provenance.toml](model-provenance.toml). Public package, model-page and Space
checks follow publication; local preparation does not establish their live state.
