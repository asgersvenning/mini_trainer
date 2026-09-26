# MAMBO_v3 local evaluation — 23 September 2026

Historical **FP32, single-view** reference. Current adoption comparisons are in
[the deployment README](../deployment/README.md#release-comparison); these timings
precede AMP and pipeline improvements.

The native PyTorch and standard ONNX candidates produced identical top-1 labels
on all **58,640 Flemming images**, for species, genus and family, with all five
lists below. This establishes same-model prediction agreement on this dataset;
it is not a claim of numerical identity or improvement over MAMBO_v2.

## Quality and geographic filtering

| Preset | Species micro accuracy, all | Species micro accuracy, known | Species macro-F1, all | Genus micro accuracy, all | Family micro accuracy, all |
|---|---:|---:|---:|---:|---:|
| Full | 58.43% | 67.72% | 0.0971 | 70.58% | 90.31% |
| Europe, legacy | 68.95% | 79.91% | 0.1993 | 77.94% | 92.86% |
| Northern Europe, legacy | 70.79% | 82.04% | 0.2545 | 79.37% | 92.98% |
| Europe, updated | 68.81% | 79.74% | 0.1975 | 77.79% | 92.55% |
| Northern Europe, updated | 70.32% | 81.50% | 0.2367 | 78.89% | 92.89% |

Both backends have the same values. There are 522 truth species; 16 species,
covering 8,042 images, are absent from all five vocabularies. All-image accuracy
keeps those examples; known-truth accuracy uses the remaining 50,598 images
(86.29% coverage). Family truth is fully covered; genus truth is fully covered
except for one image under legacy northern Europe. Predictions use a fixed zero
threshold, without tuning or abstention.

Updated Europe adds 72 candidate species and updated northern Europe adds 222,
with no removals. Those broader choices slightly reduce accuracy on Flemming;
choose a preset for its documented geographic scope, not its test-set score.
The [preset catalogue](model-presets.md) describes the occurrence filters and
documented minimum metadata-row counts. This European dataset does not qualify the usefulness
of every other geographic preset.

Metrics use `mini_metrics` at the revision below. Macro-F1 includes predicted-only
classes. [Retained comparison data](assets/mambo-release-comparison.json) includes
macro and micro metrics at all ranks; the
[metric definitions](mambo-release-comparison.md#prediction-quality) explain their
denominators. These results have no support truncation or threshold optimization.

A separate seeded **256-image / 112-species** qualification found identical
rank labels across PyTorch/ONNX × CPU/CUDA × predictions/embeddings, for all five
lists. Embeddings were finite, normalized 1280-dimensional vectors. Supplying
updated Europe as a custom class list preserved selection and order. This does
not establish downstream embedding quality or full-dataset CPU/embedding accuracy.

## Measurement environment

Hardware: Intel Core i7-12800H and RTX 3080 Ti Laptop GPU (16 GB), Linux/WSL2,
on AC power. Python 3.13.7, PyTorch 2.12.0+cu130, ONNX Runtime GPU 1.30.0,
NumPy 2.4.6, Pillow 12.2.0; FP32, TF32 disabled, no autocast.
Thirty-six fresh processes cover three trials of each runtime/device/output/thread
configuration; each reported cell pools 21 warmed observations. No evaluation or
test jobs overlapped timing. Power-policy fields were unavailable under WSL;
raw reports retain observed GPU temperatures, power and clocks.

## End-to-end latency and throughput

Updated Europe, four CPU threads. Batch latency includes decoding, preparation,
transfers, hierarchy reduction and optional embeddings. Selected median timings:

| Backend | Device / batch | Predictions, ms | With embeddings, ms |
|---|---|---:|---:|
| ONNX | CPU / 8 | 829.4 | 812.2 |
| PyTorch | CPU / 8 | 1013.9 | 1035.1 |
| ONNX | CUDA / 32 | 774.5 | 775.9 |
| PyTorch | CUDA / 32 | 724.5 | 737.7 |

Embeddings added little batched cost in this experiment; small differences were
within trial variation. The [original tables](https://github.com/asgersvenning/mini_trainer/blob/595582c212e7f248b400f7a782857bc5bdfe5111/docs/mambo-v3-evaluation.md#end-to-end-latency-and-throughput)
retain every batch size, p95 and trial range. The subsequent
[FP32 release comparison](mambo-release-comparison.md) provides V2/V3 charts.

## Startup and process memory

Cold first image includes lazy load and first execution. RSS is the process
high-water mark across the batch sweep; these selected rows exclude embeddings.

| Backend | Device | Median cold first image, s | Peak host RSS range, MiB |
|---|---|---:|---:|
| ONNX | CPU | 0.37 | 573–591 |
| ONNX | CUDA | 1.64 | 1461–1477 |
| PyTorch | CPU | 40.46 | 1569–1636 |
| PyTorch | CUDA | 41.78 | 1896–1899 |

Single-thread CPU batch-1 medians were 214 ms (ONNX) and 269 ms (PyTorch),
or 225/268 ms with embeddings. Four threads improve this workload; the bounded
batch sweep does not establish maximum CPU throughput. Full-vocabulary timings
and all raw observations are retained in the machine-readable evidence.

At GPU batch 32 without embeddings, prepared-runtime median latency was about
180–182 ms, compared with 725–775 ms end-to-end. Image preparation is a substantial
cost. Prepared-runtime timing bypasses the public image path; the public API still
applies preprocessing to array inputs.

First-call timings exclude runtime import/configuration (median approximately
0.86 s for PyTorch and 0.03 s for ONNX), lightweight predictor construction and
Python interpreter startup. They are not cold-boot measurements. Native loading
spent 36.9–39.4 s in spherical classifier initialization before restoring weights.

Native CUDA allocator peaks were approximately 865 MiB allocated / 1,348 MiB
reserved across the batch sweep. ONNX has device-memory snapshots, not a matching
continuous per-process peak measurement; do not compare those quantities directly.
Both ONNX graphs placed all 170 convolution operations on CUDA; one `Acos` and
four `Concat` operations ran on CPU. Their separate profiler timings are excluded
from the benchmark. All ONNX benchmark processes ran without importing PyTorch.

## Reproducibility and limits

The [evaluation workflow](../dev/releases/mambo_v3/evaluation.md) provides the
commands, timing boundaries, pinned metric environment and UCloud handoff.
The original 632,913 in-domain test identities and taxonomy were checked locally;
image verification, inference and reporting subsequently completed on
[UCloud](mambo-indomain-evidence.md), preserving the original split.

Local evidence is retained outside Git under `local-evidence/mambo-v3/`:
`quality-subset-256`, `quality-full`, `performance` and `combined-results`.
Reports preserve image identities/hashes, class-list hashes, artifact hashes,
runtime versions and completion status. Full-dataset collection used serial image
preparation for native and four ordered workers for ONNX; its elapsed time is not
a backend speed comparison. The dedicated benchmark uses matched boundaries.

- Bundle manifest SHA-256: `c576f53f404575bb9301de16427dd4feee9b03e48292180e184e0385052ee49b` (artifact revision 2).
- Flemming manifest SHA-256: `04e56e9189933b6758b192f587e296e49012fd935baa00c04ca91e0a3ce3795f`.
- Metric revision: `70cc69adc05362863439277048e06386c1f885e1`.
- Runner implementation: `0de3e0d`, with startup instrumentation in `922a45f`.

The rebuilt deployment wheel passed a fresh installed ONNX-only check with a
relocated read-only bundle, API/CLI execution and Python network calls blocked.
Installed CUDA prediction/embedding modes also passed independently of PyTorch;
the temporary CUDA environment reused existing NVIDIA libraries and is not a clean
dependency-installation qualification. Those JSON reports are retained as
`portable-install-final.json` and `installed-gpu-final.json`.

For current provenance, notices and platform limits see
[final qualification](../dev/releases/mambo_v3/final-qualification.md).
These early checks are not final-wheel certification. The archived September
native predictions are historical context, not a MAMBO_v2 quality baseline.
