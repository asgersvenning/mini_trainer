# Deployment qualification and report reproduction

Current installed-package evidence is in [final qualification](final-qualification.md).
This page retains the build/check commands, TTA report reproduction and the scope
of earlier adapter checks. Consumer instructions are in
[deployment/README.md](../../../deployment/README.md).

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

## What the early checks established

The September 23 adapter check used four deterministic images on CPU and an
RTX 3080 Ti Laptop. Both runtimes agreed on species/genus/family top-1 for
full/Europe/custom lists, with and without embeddings. Vectors were finite unit
1280-dimensional embeddings, unchanged by class masks. This qualifies execution
contracts, not representative accuracy or downstream embedding quality.

The independent ONNX-only CPU install used Python 3.13.7, ORT 1.30.0, NumPy 2.5.3
and Pillow 12.3.0. CPU backend comparison used PyTorch 2.12.0, ORT 1.29.0,
NumPy 2.4.6 and Pillow 12.2.0; CUDA used PyTorch cu130 and ORT GPU 1.30.0,
reusing existing NVIDIA libraries. It was not a clean GPU dependency-resolution
test. Local reports remain under
`local-evidence/mambo-v3/*deployment-qualification.json` and
`portable-install-qualification.json`.

The original torchvision/NumPy resize comparison differed by at most one uint8
level at rounding boundaries. Later full-dataset, installed-artifact and TTA
checks supersede this small initial qualification; see
[local evaluation](../../../docs/mambo-v3-evaluation.md),
[in-domain evidence](../../../docs/mambo-indomain-evidence.md) and
[final qualification](final-qualification.md).
The latter owns current package identities, notices and remaining limitations;
early checks do not certify the final wheels or untested operating systems.

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

The promoted transforms matched the full-study transforms byte-for-byte.
The installed ONNX-only wheel passed prediction/embedding API and CLI with TTA
against a relocated read-only bundle, Python socket calls blocked and unchanged
model hashes (`local-evidence/mambo-promoted-portable.json`). Python 3.14 was not
qualified in that offline environment. Timing observations and input hashes remain
in `docs/assets/mambo-promoted-speed.json`; current results are linked from the
deployment README rather than repeated here.

## CUDA optimization compatibility probe — 2026-09-24

Each CUDA session now executes a synthetic batch-one input before its first user
prediction. Kernel-image/device-function incompatibility retries with ORT graph
optimizations disabled; no CPU-only fallback or retry of unrelated errors occurs.
The selected profile and initialization/probe timings are retained in reports.
A batch-one check does not establish every batch-dependent execution path.

On the RTX 3080 Ti Laptop with ORT GPU 1.30.0, both optimized graphs ran,
including TTA and embeddings. Injecting an initial compatibility error exercised
real unoptimized CUDA recovery for both graphs. This qualified recovery mechanics,
not the B200 failure: there, the same wheel also failed an unfused standalone
Sigmoid. The [UCloud runbook](ucloud-release.md) records the separately qualified
ORT build used for that environment. Disabling graph optimizations is not a
general runtime/device compatibility fix.
