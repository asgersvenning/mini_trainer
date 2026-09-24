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

## Enabled-TTA promotion — 2026-09-24

The deployment default when TTA is requested is now `rotation30_pad25_3`: original,
−30° with 25% edge padding, +30° with 25% edge padding. Rotation expands the canvas
and uses bilinear interpolation and RGB (124,116,104) corner fill, exactly as in
the full-data study. TTA stays off when omitted. Explicit `padded_scale` is retained;
`wide_rotation_mixed_padding_5` is also available as an opt-in named profile.

The [current deployment comparison](../../../deployment/README.md#release-comparison)
uses the full-study predictions, with macro metrics computed by pinned mini_metrics.
Support intersections are recomputed across the five displayed pipelines; do not
copy the eleven-pipeline exploratory tail scores into that table. Both threshold
settings use the same 52,788 reporting images, with 5,852 calibration images.

Reproduce the current quality tables and figure:

```sh
/tmp/mambo-release-metrics/bin/python -m dev.releases.mambo_v3.promoted_report \
  --quality docs/assets/mambo-composed-tta.json --output /tmp/promoted-report
.venv/bin/python -m dev.releases.mambo_v3.tail_charts --paired \
  --data /tmp/promoted-report/mambo-promoted-tail.json --output /tmp/promoted-report
```

The figure command writes `mambo-threshold-tail.svg`; publish it under the distinct
name `mambo-promoted-quality.svg` to preserve earlier studies. JSON/CSV outputs use
`mambo-promoted-*`. `quality-tables.md` supplies the README metric and support rows.
The JSON threshold artifact retains full macro/micro scores, coverage, recipe,
source hashes and exact reporting/calibration identities; thresholds are not
silently installed as runtime defaults.

Measure the selected recipe independently, on the same bank as earlier timings:

```sh
CUDA_VISIBLE_DEVICES=0 /tmp/mambo-deploy-qualification-gpu/bin/python \
  -m dev.releases.mambo_v3.benchmark_acceleration \
  --python /tmp/mambo-deploy-qualification-gpu/bin/python \
  --bundle local-evidence/mambo-bundle-presets-v2 \
  --manifest local-evidence/mambo-v3/flemming-manifest.json \
  --root /home/asger/data/flemming --output /tmp/promoted-speed \
  --presets north_europe --tta rotation30_pad25_3
.venv/bin/python -m dev.releases.mambo_v3.promoted_report \
  --performance /tmp/promoted-speed --output /tmp/promoted-report
.venv/bin/python -m dev.releases.mambo_v3.promoted_report \
  --render-speed /tmp/promoted-report/mambo-promoted-speed.json --output /tmp/promoted-report
```

These commands use three fresh processes per backend/device, seven observations,
CPU batches 1/8 and GPU batches 1/8/32, without concurrent model workloads. They
reuse earlier V2 and single-view V3 timings; temperature/power differences between
campaigns remain a limitation. Timing does not reuse full-evaluation wall time.
Resource records retain host peak RSS. New runs cover northern Europe only;
earlier resource sweeps also covered other presets, so RSS is descriptive.

Promotion checks: 53 focused release checks passed (including the two pinned-metric
checks run separately), static/import checks passed, and preset transforms match
the full-study transforms byte-for-byte. The standalone wheel builds offline and
its enabled default imports without torch or mini_trainer in a clean Python 3.13
environment. Python 3.14 installation was not qualified: its wheels were absent
from the offline cache. Older regional/frequency/threshold studies are preserved
and labelled historical rather than relabelled as new-recipe evidence.

The installed ONNX-only wheel also passed prediction-only/embedding API and CLI
inference with the selected recipe against a relocated read-only bundle, with
Python socket connections blocked and model hashes unchanged. Evidence:
`local-evidence/mambo-promoted-portable.json`. The twelve timing trials completed;
GPU batch-32 throughput was 50.89 images/s native and 38.27 ONNX. CPU batch-1 was
2.04 and 3.39 images/s. Trial ranges and input hashes are in
`docs/assets/mambo-promoted-speed.json`.

## CUDA optimization compatibility probe — 2026-09-24

Each CUDA session now executes a synthetic batch-one input before its first user
prediction. Kernel-image/device-function incompatibility retries with ORT graph
optimizations disabled; no CPU-only fallback or retry of unrelated errors occurs.
The selected profile and initialization/probe timings are retained in reports.
A batch-one check does not establish every batch-dependent execution path.

On the RTX 3080 Ti laptop with ORT GPU 1.30.0, both optimized graphs executed,
including default TTA and embedding output. Injecting the initial compatibility
error exercised recovery into real unoptimized CUDA execution for both graphs.
This validates local recovery mechanics, not resolution of the reported B200
failure. The installed ONNX-only wheel also passed CPU prediction, embeddings and
TTA without importing torch. Focused deployment/download/evaluation checks passed
(64 tests); nine UCloud harness tests passed separately. Static/import checks and
the standalone deployment lint/format checks passed. No full suite was run.
