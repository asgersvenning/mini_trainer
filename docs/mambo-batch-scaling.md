# Why MAMBO v3 batch throughput plateaus

This records the pre-optimization diagnosis at commit `a99b855`. See the
[accelerated deployment qualification](mambo-accelerated-deployment.md) for the
implemented fixes and their new measurements. Historical preprocessing probes
should be replayed from that commit, since the current adapter is optimized.

The plateau comes from **serial, allocation-heavy CPU preprocessing plus a strict
FP32 convolutional backend that gains little throughput beyond batch 8**. It is
not a batch-size parameter being ignored. Forward hooks observed exactly
`[1,3,384,384]`, `[8,3,384,384]` and `[32,3,384,384]` at the native model boundary.
The released speed charts remain unchanged; the following interventions explain
them and are not new qualified release variants.

## CPU cause: the release image adapter

The NumPy preprocessor executes one image at a time before calling the GPU. Its
advanced indexing produces non-contiguous arrays, and `float32 coordinates -
int64 indices` promotes interpolation weights and intermediate images to float64.
It interpolates the entire 438×438 image before discarding its border for the
384×384 crop. The row/column interpolation and normalization dominate the CPU
profile; decoding accounts for only about 30 ms of a 602 ms batch-32 profile.
Increasing the batch size cannot amortize this per-image work.

A controlled, single-thread intervention retained the arithmetic and verified
byte-identical prepared pixels for all 32 benchmark images. Three sweeps with
reversed middle ordering and seven observations per condition gave:

| Preparation of 32 images | Median time |
|---|---:|
| Current implementation | 562 ms |
| Make intermediate arrays contiguous | 344 ms |
| Compute only the retained crop | 526 ms |
| Both interventions | 300 ms |

A separate diagnostic with the **original** pixel function and eight preparation
workers reduced measured end-to-end native batch-32 time from 753 to 361 ms
(42.5 → 88.7 images/s). Four workers reached 398 ms. Those single-process,
seven-observation interventions demonstrate causality, not a replacement for the
three-fresh-process release benchmark. Results are hardware-dependent. Preprocessing
and inference still do not overlap in this probe.

## GPU cause: FP32 backbone work, not hierarchy or silent CPU execution

With inputs already resident on the GPU, the same model at the same resolution
was tested with TF32 disabled. These are medians across three sweeps × seven
observations, with the middle order reversed:

| Diagnostic native mode | Batch 1, images/s | Batch 8, images/s | Batch 32, images/s |
|---|---:|---:|---:|
| Release FP32 / NCHW | 53.4 | 162.7 | 177.6 |
| FP16 autocast / NCHW | 39.9 | 310.2 | 371.4 |
| FP32 / channels-last | 51.3 | 134.0 | 137.6 |
| FP16 autocast / channels-last | 39.5 | 293.4 | 418.4 |

Batch 8 already captures most of the FP32 throughput benefit. At batch 32, CUDA
kernel durations are approximately **63.6% convolution, 14.2% batch normalization
and 12.3% SiLU**. The final classifier matrix multiplication is about 0.15% of
kernel time. Reducing hierarchy work or changing class lists cannot explain away
the backbone plateau. Memory-layout changes alone do not fix it. The controlled
precision change more than doubles large-batch throughput; FP16 can still be
slower at batch 1 because launch/cast overhead remains.

ONNX placement traces confirm convolution runs on CUDA at batches 1, 8 and 32.
Small `Acos` and `Concat` nodes run on CPU; this is not a silent CPU backbone
fallback. Their host-side node durations are not GPU kernel durations and should
not be read as a GPU utilization breakdown.

V2 uses a different backbone at 224 pixels with CUDA autocast; v3 uses 384 pixels
and strict FP32. These released choices, plus the adapter's CPU work, explain why
v3 does not inherit v2's batching curve. The evidence localizes the bottleneck to
feature-map operations and demonstrates a precision effect; it does **not**
establish a hardware-counter distinction between arithmetic and memory bandwidth
limits. Profiler overhead and laptop clock variation are why unprofiled timings
and reversed-order interventions are reported separately.

## Next implementation step

Prioritize pixel-preserving contiguous/crop preparation, then bounded workers and
CPU/GPU overlap. Qualify exact inputs on the larger retained image subset and
array-input edge cases, then rerun fresh-process end-to-end timings. These adapter
changes can stay on the release branch. Any shared-core optimization belongs on a
feature/fix branch and must be merged through the established release workflow.

FP16 is a separate numerical/runtime variant, not quantization, but is diagnostic
only here. It needs task-level quality and deployment qualification before becoming
a supported option. The current PyTorch/ONNX FP32 baseline remains available.

## Reproduce

[Compact diagnostic evidence](assets/mambo-batch-diagnosis.json) records timings,
shapes, array layouts, kernel attribution and raw evidence hashes. Large traces and
private sample paths stay under `local-evidence/mambo-batch-root-cause/`.
Run these sequentially, with no competing benchmark workload:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m dev.releases.mambo_v3.profile_batch_scaling \
  --bundle /path/to/bundle --manifest /path/to/flemming-manifest.json \
  --root /path/to/flemming --output /path/to/new-diagnosis

python -m dev.releases.mambo_v3.profile_preprocessing \
  --evidence /path/to/new-diagnosis --root /path/to/flemming
python -m dev.releases.mambo_v3.probe_preprocessing \
  --evidence /path/to/new-diagnosis --root /path/to/flemming
# Use the qualified ONNX CUDA environment for this command:
python -m dev.releases.mambo_v3.profile_onnx_batch \
  --evidence /path/to/new-diagnosis --root /path/to/flemming --bundle /path/to/bundle
python -m dev.releases.mambo_v3.summarize_batch_scaling \
  --evidence /path/to/new-diagnosis --output /path/to/diagnosis.json
```
