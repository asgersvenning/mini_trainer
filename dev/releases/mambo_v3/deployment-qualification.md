# Deployment qualification and report reproduction

Current installed-package evidence is in [final qualification](final-qualification.md).
Use this page to reproduce runtime contracts and the promoted Flemming reports.
Consumer installation and configuration belong in
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

Use disposable environments following the [runtime installation guide](../../../docs/mambo-integration.md#runtime-installation).
The default check below runs both backends, so install the matching training wheel
and chosen PyTorch backend too. Use `--backends onnx` for an ONNX-only check.
Do not install CPU and GPU ONNX Runtime packages together.

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

These four-image checks establish execution contracts, not representative accuracy,
embedding quality or broad platform support. Add `--tta rotation30_pad25_3` to
both runners to check the released enabled-TTA default; omission checks TTA off.
[Final qualification](final-qualification.md) owns installed artifact identities,
environments and results, including the promoted TTA checks.

## Reproduce the promoted Flemming report

Use the pinned [metric environment](evaluation.md#metrics) for quality processing
and a separately prepared runtime for timing. The commands below show the paths
used in the retained campaign; substitute your environment, dataset and bundle
paths. Run from the repository root, with fresh output directories.

The five displayed pipelines need their own shared support intersection; do not
copy the eleven-pipeline exploratory tail scores. Both confidence settings use
the same 52,788 reporting images, with 5,852 separate calibration images. Full
metric semantics and results are in [deployment evidence](../../../docs/mambo-deployment-evidence.md).

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

The timing runner uses three fresh processes per backend/device, seven observations,
CPU batches 1/8 and GPU batches 1/8/32. Do not overlap model workloads. The report
combines these northern-Europe TTA timings with retained V2 and single-view V3
measurements; campaign conditions can differ. RSS is a process high-water mark,
not per-request memory. [Evaluation](evaluation.md#timing-and-summary) defines the
measurement boundaries; [HPC evidence](../../../docs/mambo-hpc-evidence.md) owns
the later B200 measurements.

## CUDA compatibility boundary

The [integration guide](../../../docs/mambo-integration.md#runtime-installation)
describes the synthetic first-use probe and checked retry without graph optimizations.
On the RTX 3080 Ti Laptop with ORT GPU 1.30.0, both optimized graphs ran;
injecting a compatibility error exercised real unoptimized CUDA recovery for both.
That verifies recovery mechanics, not every batch-dependent path or GPU runtime.
On B200, that wheel also failed an unfused standalone Sigmoid: disabling graph
optimizations cannot repair missing device kernels. The [UCloud runbook](ucloud-release.md)
records the separately qualified runtime build.
