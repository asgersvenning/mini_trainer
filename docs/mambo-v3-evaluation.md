# MAMBO_v3 local evaluation — 23 September 2026

The native PyTorch and standard ONNX candidates produced identical top-1 labels
on all **58,640 Flemming images**, for species, genus and family, with all five
lists below. This establishes same-model prediction agreement on this dataset;
it is not a claim of numerical identity or improvement over MAMBO_v2.

For the subsequently measured MAMBO_v2 baseline and readable release-to-release
charts, see the [real-world comparison](mambo-release-comparison.md).

## Quality and geographic filtering

| Preset | Species accuracy, all images | Species accuracy, known truth | Macro-F1, all | Genus accuracy | Family accuracy |
|---|---:|---:|---:|---:|---:|
| Full | 58.43% | 67.72% | 0.0971 | 70.58% | 90.31% |
| Europe, legacy | 68.95% | 79.91% | 0.1993 | 77.94% | 92.86% |
| Northern Europe, legacy | 70.79% | 82.04% | 0.2545 | 79.37% | 92.98% |
| Europe, updated | 68.81% | 79.74% | 0.1975 | 77.79% | 92.55% |
| Northern Europe, updated | 70.32% | 81.50% | 0.2367 | 78.89% | 92.89% |

Both backends have the same values. There are 522 truth species; 16 species,
covering 8,042 images, are absent from all five vocabularies. All-image accuracy
keeps those examples; known-truth accuracy uses the remaining 50,598 images
(86.29% coverage). Genus and family truth are fully covered. Predictions use a
fixed zero threshold, without tuning or abstention.

Updated Europe adds 72 candidate species and updated northern Europe adds 222,
with no removals. Those broader choices slightly reduce accuracy on Flemming;
choose a preset for its documented geographic scope, not its test-set score.
The [preset catalogue](model-presets.md) describes the occurrence filters and
documented minimum metadata-row counts. This European dataset does not qualify the usefulness
of every other geographic preset.

Macro-F1 follows the pinned `mini_metrics` implementation, including predicted-only
classes; it must not be read as accuracy. The retained JSON reports also contain
macro precision/recall, Theil's U, known-only and per-class results at every rank.
No metric-policy or threshold optimization was performed.

A separate seeded **256-image / 112-species** qualification found identical
rank labels across PyTorch/ONNX × CPU/CUDA × predictions/embeddings, for all five
lists. Embeddings were finite, normalized 1280-dimensional vectors. Supplying
updated Europe as a custom class list preserved selection and order. This does
not establish downstream embedding quality or full-dataset CPU/embedding accuracy.

## Choosing a runtime

ONNX is the practical default for new integrations here: it needs no training
package, starts much faster, and uses less CPU process memory. Native PyTorch
remains compatible with existing callers and gives competitive warmed GPU
throughput. Embedding extraction adds little batched cost in this measurement;
small differences and single-image differences should be read alongside the
trial ranges, not as universal speed claims.

Hardware: Intel Core i7-12800H and RTX 3080 Ti Laptop GPU (16 GB), Linux/WSL2,
on AC power. Python 3.13.7, PyTorch 2.12.0+cu130, ONNX Runtime GPU 1.30.0,
NumPy 2.4.6, Pillow 12.2.0; FP32, TF32 disabled, no autocast.
Thirty-six fresh processes cover three trials of each runtime/device/output/thread
configuration; each reported cell pools 21 warmed observations. No evaluation or
test jobs overlapped timing. Power-policy fields were unavailable under WSL;
raw reports retain observed GPU temperatures, power and clocks.

## End-to-end latency and throughput

Laptop measurements; updated Europe, four CPU threads, three alternating-order trials. Batch latency includes image decoding, preprocessing, transfers, hierarchy reduction and optional embeddings. p95 is descriptive of the retained observations, not a service-level guarantee.

| Backend | Device | Embeddings | Batch | Median ms | p95 ms | Images/s | Trial median range ms |
|---|---|---|---:|---:|---:|---:|---:|
| onnx | cpu | No | 1 | 114.8 | 134.6 | 8.7 | 105.1–124.6 |
| onnx | cpu | No | 8 | 829.4 | 904.7 | 9.6 | 772.6–864.6 |
| onnx | cpu | Yes | 1 | 108.0 | 119.1 | 9.3 | 99.4–113.2 |
| onnx | cpu | Yes | 8 | 812.2 | 891.5 | 9.8 | 759.7–851.0 |
| onnx | cuda:0 | No | 1 | 33.4 | 38.4 | 29.9 | 30.6–35.8 |
| onnx | cuda:0 | No | 8 | 194.5 | 211.8 | 41.1 | 181.0–202.4 |
| onnx | cuda:0 | No | 32 | 774.5 | 949.1 | 41.3 | 764.2–784.3 |
| onnx | cuda:0 | Yes | 1 | 33.9 | 35.5 | 29.5 | 33.3–34.2 |
| onnx | cuda:0 | Yes | 8 | 199.2 | 211.7 | 40.2 | 190.6–201.8 |
| onnx | cuda:0 | Yes | 32 | 775.9 | 1019.4 | 41.2 | 748.1–784.9 |
| torch | cpu | No | 1 | 156.1 | 174.7 | 6.4 | 155.8–160.8 |
| torch | cpu | No | 8 | 1013.9 | 1098.4 | 7.9 | 994.5–1078.9 |
| torch | cpu | Yes | 1 | 151.7 | 164.4 | 6.6 | 148.7–153.2 |
| torch | cpu | Yes | 8 | 1035.1 | 1120.3 | 7.7 | 1033.5–1036.3 |
| torch | cuda:0 | No | 1 | 35.3 | 48.1 | 28.3 | 33.2–41.3 |
| torch | cuda:0 | No | 8 | 178.3 | 194.9 | 44.9 | 170.8–192.5 |
| torch | cuda:0 | No | 32 | 724.5 | 921.8 | 44.2 | 717.8–746.3 |
| torch | cuda:0 | Yes | 1 | 38.7 | 54.0 | 25.8 | 37.5–50.9 |
| torch | cuda:0 | Yes | 8 | 190.4 | 202.6 | 42.0 | 184.6–198.5 |
| torch | cuda:0 | Yes | 32 | 737.7 | 905.9 | 43.4 | 722.2–744.8 |

## Startup and process memory

Cold first image includes lazy load and first execution; RSS is the process high-water mark across the batch sweep.

| Backend | Device | Embeddings | Median cold first image s | Peak RSS range MiB |
|---|---|---|---:|---:|
| onnx | cpu | No | 0.37 | 573–591 |
| onnx | cpu | Yes | 0.38 | 562–582 |
| onnx | cuda:0 | No | 1.64 | 1461–1477 |
| onnx | cuda:0 | Yes | 1.69 | 1460–1463 |
| torch | cpu | No | 40.46 | 1569–1636 |
| torch | cpu | Yes | 40.10 | 1518–1571 |
| torch | cuda:0 | No | 41.78 | 1896–1899 |
| torch | cuda:0 | Yes | 41.80 | 1898–1900 |

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
A loading optimization belongs on a separate core feature/fix branch, followed by
checkpoint regression checks and merge into the release branch.

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
image verification and inference remain for UCloud. No new split is generated.

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

Tests: **718 passed, 161 skipped, 1 expected failure** in the full repository
suite; static checks passed. The subsequent startup instrumentation was covered by
**39 passing focused release tests** and static checks. Skipped GPU/slow tests are
not implied to pass by that suite; actual GPU evidence is described separately above.

The rebuilt deployment wheel passed a fresh installed ONNX-only check with a
relocated read-only bundle, API/CLI execution and Python network calls blocked.
Installed CUDA prediction/embedding modes also passed independently of PyTorch;
the temporary CUDA environment reused existing NVIDIA libraries and is not a clean
dependency-installation qualification. Those JSON reports are retained as
`portable-install-final.json` and `installed-gpu-final.json`.

In-domain results, cross-OS support, a clean CUDA dependency installation,
training-source/best-epoch provenance and redistribution notices remain open.
The archived September native predictions are historical context, not a
MAMBO_v2 quality baseline. Nothing has been published or tagged.
