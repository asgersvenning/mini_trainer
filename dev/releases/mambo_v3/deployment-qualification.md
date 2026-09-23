# Portable deployment qualification — 2026-09-23

This increment supplies a local bundle builder, an independent `mambo_deploy`
wheel, and the `mini_trainer.deploy.Predictor` compatibility entry point. The
consumer guide is [deployment/README.md](../../../deployment/README.md).
Shared training, loading and classifier modules are unchanged.

## Reproduce

From the repository root, using the existing environment without syncing:

```sh
.venv/bin/python -m dev.releases.mambo_v3.build_bundle local-evidence/mambo-v3 local-evidence/mambo-bundle-final
uv build --project deployment --wheel --out-dir local-evidence/deployment-wheels
uv build --wheel --out-dir local-evidence/deployment-wheels
```

The source directory must contain the production directory named in
`inventory.toml`; the builder verifies the pinned hashes and refuses to overwrite
an existing destination. Model files stay outside Git. The bundle includes both
ONNX graphs and their external data, native weights, one ordered vocabulary,
parent mappings, preprocessing, preset lists/scopes, provenance and hashes.

In a disposable environment, install the deployment wheel with `[onnx]`.
For native qualification also install the matching training wheel with the
chosen CPU/CUDA dependencies. For GPU ONNX use `onnxruntime-gpu` instead of the
CPU runtime, and matching CUDA/cuDNN libraries. Then run:

```sh
python dev/releases/mambo_v3/qualify_bundle.py /path/to/bundle /path/to/flemming --device cpu --output cpu.json
python dev/releases/mambo_v3/qualify_bundle.py /path/to/bundle /path/to/flemming --device cuda:0 --output gpu.json
```

The runner selects the first JPEG in each of the first four sorted species
directories, records image hashes, and checks full/Europe/custom lists. It checks
prediction/embedding mode agreement, finite unit-length 1280-dimensional vectors,
and that class filtering does not change embeddings. Across backends it reports
label agreement, without imposing micro-numerical equivalence.

In a separate ONNX-only environment with neither torch nor mini_trainer installed:

```sh
python -I dev/releases/mambo_v3/check_portable_install.py /path/to/bundle /path/to/image.jpg --output portable.json
```

This copies the bundle to a temporary location, removes write permissions, changes
working directory, blocks Python socket connections, runs both API modes and the
CLI, and verifies bundle contents remain unchanged. This is not an OS-level
network isolation test; platform-native runtime networking is outside that guard.

## Initial adapter increment

| Check | Result |
|---|---|
| CPU PyTorch / ONNX, four images | 4/4 identical top-1 species/genus/family tuples for full, Europe and shared custom lists |
| RTX 3080 Ti Laptop GPU, same images | 4/4 identical tuples for those three list modes |
| Predictions with/without embeddings | Same top-1 tuples within each backend on CPU and GPU |
| Embeddings | `[4,1280]`, finite, unit length; unaffected by class mask |
| ONNX CUDA provider | CUDA first in both sessions; individual CPU operators remain allowed |
| Clean ONNX installation | No torch/training package; relocated read-only bundle, API and CLI passed |
| Minimal training wheel | Imports, CLI help without deployment extra, training, reload and prediction passed |
| Focused release suite | 28 tests passed, including eight deployment contracts |
| Static checks | Ruff, formatting and both import contracts passed; standalone deployment package checked separately |

The optional broader `dev/check.sh all` run was interrupted while still in
unrelated benchmark tests after approximately five minutes; it did not establish
a complete full-suite result. No failure had been reported before interruption.

CPU comparison used PyTorch 2.12.0, ONNX Runtime 1.29.0, NumPy 2.4.6 and
Pillow 12.2.0. GPU qualification used the CUDA 13.0 PyTorch build and ONNX
Runtime GPU 1.30.0. Its temporary environment reused existing training dependencies;
it was not a clean GPU dependency-resolution test. The independent CPU-only install
used ONNX Runtime 1.30.0, NumPy 2.5.3 and Pillow 12.3.0, Python 3.13.7 on Linux.
Local JSON evidence is under `local-evidence/mambo-v3/*deployment-qualification.json`
and `portable-install-qualification.json` (ignored).

The shared NumPy preprocessing implements the recorded campaign recipe. A sampled
comparison to the original torchvision path differed by at most one uint8 level
at resize rounding boundaries; it is not a byte-exact preprocessing claim.

## Subsequent evaluation and remaining work

The [measured release report](../../../docs/mambo-v3-evaluation.md) supersedes the
initial subset-only evidence above with full Flemming metrics, CPU/GPU timings
and a completed broad test suite. The [evaluation workflow](evaluation.md) preserves
unknown truth and documents the UCloud commands. In-domain inference still needs
to run on UCloud with the verified original test split.

Four deterministic images establish execution contracts, not representative
accuracy, embedding quality or speed. Windows/macOS, clean CUDA installations,
additional architectures, training revision/best-epoch provenance and redistribution
notices remain unqualified. Nothing has been uploaded, tagged or promoted.
