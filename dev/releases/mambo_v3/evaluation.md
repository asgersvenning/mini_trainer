# Release evaluation and UCloud handoff

The runner compares the pinned native PyTorch and standard ONNX artifacts through
the release preprocessing and hierarchy reducer. No quantization, retraining,
threshold optimization or new dataset split is involved. All work here is release
tooling; the shared core remains unchanged.

## Local workflow

Use the existing checkout environment without syncing, or install the candidate
wheels and prepare a separate CUDA environment. The local qualification uses
Python 3.13.7, PyTorch 2.12.0+cu130, ONNX Runtime GPU 1.30.0, NumPy 2.4.6 and
Pillow 12.2.0. ONNX CPU execution does not import PyTorch. CUDA library dependencies
must be provisioned separately (CUDA 13 and cuDNN 9 for this tested ORT build); a CPU-only ONNX wheel cannot execute CUDA.

From the repository root:

```sh
python -m dev.releases.mambo_v3.evaluate prepare \
  --root /path/to/flemming \
  --reference /path/to/production/evaluation/expert/predictions/mini_metric.csv \
  --output /path/to/flemming-manifest.json

python -m dev.releases.mambo_v3.run_local qualification \
  --python /path/to/runtime-env/bin/python \
  --bundle /path/to/bundle --manifest /path/to/flemming-manifest.json \
  --root /path/to/flemming --output /path/to/new-subset-run
```

The evaluation and timing commands retain `--precision fp32` by default to
preserve the original reference protocol. Pass `--precision auto` to measure the
current deployment defaults (native CUDA FP16 backbone, ONNX CUDA TF32, CPU FP32).
The [accelerated comparison workflow](../../../docs/mambo-accelerated-deployment.md#reproduce)
provides the full qualification and three-trial timing commands. The public
`Predictor` and deployment CLI default to `auto`; every report records the resolved
precision.

Preparation joins all three archived truth ranks by original species/image identity
and hashes local image bytes. It rejects missing/extra images and duplicate or
incomplete truth. The seeded 256-image subset comes from the existing benchmark
selector; it is a qualification subset, not a replacement for the full expert set.
The manifest preserves every truth label, including labels outside the model.

After reviewing qualification, replace `qualification` with `full` for the two full
GPU prediction runs, or `benchmark` for isolated timing trials. Always choose a
new output directory. Failed outputs are preserved; a report must say `complete`
before using its predictions. Jobs run sequentially; do not overlap the timing
phase with collection, metrics or other heavy work.

The collector prepares at most one model batch with four ordered preprocessing
workers (`collect --decode-workers 0` selects serial preparation), checks image
hashes and executes the
backbone once. It then applies the same reducer to `full`, legacy `europe` and
`north_europe`, and updated `europe_v3` and `north_europe_v3`. No leaf-score archive
is needed for these predetermined lists. Qualification also verifies that supplying
the updated Europe preset as a custom class list preserves mask membership/order.
Embeddings are written incrementally into a memory-mapped NPY array on subset runs.

Each variant retains ordered sample identities, artifact/list hashes, precision,
runtime versions, timing components, canonical CSVs and a completion/failure report.
Use `compare_quality` for paired identity/ground-truth checks and per-rank top-1
agreement/accuracy deltas. Small score differences are not a release failure.

```sh
python -m dev.releases.mambo_v3.compare_quality /path/to/left /path/to/right \
  --output /path/to/comparison.json
```

## Fixed metric policy

Prepare a separate environment at the campaign's mini_metrics commit:

```sh
uv venv --python 3.13 /path/to/metrics-env
uv pip install --python /path/to/metrics-env/bin/python \
  'mini_metrics @ git+https://github.com/GuillaumeMougeot/mini_metrics.git@70cc69adc05362863439277048e06386c1f885e1'
/path/to/metrics-env/bin/python -m dev.releases.mambo_v3.metrics \
  --source /path/to/variant/europe_v3/mini_metric.csv \
  --output /path/to/variant/europe_v3/metrics.json
```

For a completed full phase, use `--collection /path/to/full-run` instead of
`--source`/`--output` to generate every variant/list report, skipping only existing
reports whose input hash and metric revision match. The helper enforces that revision and
reports per-rank micro accuracy, the pinned implementation's macro-F1, macro-recall,
macro-precision, coverage and Theil's U, including known-only and per-class results.
Undefined values remain null. List coverage (truth in the active vocabulary) and
abstention coverage are distinct; the latter is 100% at the fixed zero threshold.
Do not choose thresholds or geographic filters from test results.

## Timing protocol

The benchmark runs three fresh-process trials in alternating variant order:
PyTorch/ONNX × CPU/CUDA × predictions/embeddings. Four-thread CPU trials use batches
1 and 8, and GPU trials also include 32; additional one-thread CPU trials measure
batch-1 latency. CPU/GPU comparisons use the common batch sizes; this bounded
sweep does not establish the maximum possible CPU throughput. Both `full`
and `europe_v3` are measured. Every configuration uses the same seeded 32-image bank (the appropriate prefix
for each batch). Each cell uses two warmups and seven observations;
retain raw observations and show between-trial variation, not only one best time.

Runtime import/configuration, lightweight predictor construction and first-call
time are recorded separately from warmed work. These are application boundaries,
not cold-boot or complete Python-interpreter startup measurements.
The cold first call includes lazy loading; instrumentation records checkpoint
loading/model construction or ONNX session construction inside that call. Prepared
runtime timings include CPU-to-device and device-to-CPU transfers, but omit image
decoding and hierarchy reduction. End-to-end timings include those operations and
embedding copies. GPU calls return completed CPU arrays, so timings include completed
work. FP32 is used with TF32 disabled for both backends; no autocast is enabled.

RSS is the Linux process high-water mark across a trial's batch sweep. Native CUDA
allocator peaks and nvidia-smi device/process snapshots are retained. Snapshots are
observations, not continuous per-process ONNX peak measurements. Record AC state,
clocks, temperatures, power and any unavailable power-policy fields. ONNX CUDA may
place some operators on CPU; provider placement requires the separate profiler.
These are measurements of this laptop/environment, not universal hardware claims.

## UCloud: preparation only

The 632,913 original test identities and all three truth ranks have been checked
locally against the pinned Parquet (`set == "0"`), archived staging map and archived
prediction CSV. Staged numeric filenames are never used as original identities.
This does not verify image availability or content on UCloud.

Use the [UCloud release workflow](ucloud-release.md) for isolated `uv` setup,
automatic artifact downloads, five-pipeline qualification, full collection and
CPU/GPU benchmarks. Only the original Parquet location is required as dataset
input when images retain their original layout. No in-domain inference has been
run locally.

## Publication preparation

Before staging, attach the measured results to the short deployment README,
freeze wheel/runtime versions and bundle revision, and resolve training-source
revision, best-epoch provenance and model/data redistribution notices. Add the
in-domain results when UCloud runs finish. Keep MAMBO_v2 model-quality comparison
separate from this same-model backend comparison; the archived September native
predictions are historical context, not MAMBO_v2 evidence. Cross-OS and clean CUDA
installation claims require their own checks. No publishing, tagging or uploading
is performed by these tools.

After metrics and all timing trials complete, generate the combined tables:

```sh
python -m dev.releases.mambo_v3.summarize \
  --quality /path/to/full-run --benchmarks /path/to/benchmark-run \
  --output /path/to/new-summary
```

Both human-readable and machine-readable summaries preserve evidence scope. The
summary refuses an incomplete timing phase. Keep source CSVs, raw timings and
reports alongside it when staging release evidence.

The local isolated GPU environment reuses the already-installed CUDA libraries.
Its first independent ONNX placement attempt could not locate those libraries;
exposing the existing NVIDIA dependency directory in that temporary environment
resolved it without importing PyTorch. A fresh supported deployment should install
matching NVIDIA dependencies into its own environment or configure library search
paths explicitly. Both profiled graphs then ran all 170 convolution operations on
CUDA; one `Acos` and four `Concat` operations remained on CPU. The profiler's
incidental timings overlapped collection and are not included in speed results.

Native startup instrumentation also records spherical classifier initialization.
That measurement is nested inside model construction, so the loading components
must not all be added together. The existing core constructs a normalized head
with 100 initialization iterations before loading checkpoint weights; any change
to that shared loading behavior belongs on a separate feature/fix branch with
checkpoint validation, not directly on the release branch.

Metric extraction uses mini_metrics for all predictive scores, explicitly selecting
`micro_accuracy` (the bare `accuracy` field is macro), `accuracy`, `f1`, `recall`,
`precision`, `coverage` and `theilU`. Both `known_only=False` and `True` are retained;
rank summary fields reference those outputs. Schema `mini-metrics-quality-v2`
rejects cached results from the older direct-accuracy extractor. Archive old metric
JSON before recomputing from unchanged prediction CSVs. Pairwise comparison reports
only prediction agreement; predictive accuracy comes from the metric files.
