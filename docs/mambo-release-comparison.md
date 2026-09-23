# MAMBO_v2 → v3: real-world deployment comparison

This compares the models and inference pipelines used by the two releases on the
same **58,640 Flemming images** and the same laptop. **Northern Europe is the lead
preset for Flemming**; Europe and global show how the result changes with a broader
candidate vocabulary. V3 is measured through both its native PyTorch and standard
ONNX deployment paths.

## Prediction quality

![Species, genus and family accuracy, plus species macro-F1 for all three lists](assets/mambo-release-quality.svg)

With the northern-Europe list, v3 gains **2.08 percentage points** in species
accuracy (68.71% → 70.79%). Updated northern Europe retains a **1.61-point gain**
over v2. Europe also improves by 1.99 points, while global species accuracy falls
by 0.93 points. Family accuracy falls across all three lists, and regional species
macro-F1 is slightly lower. This is a useful species-level gain in the relevant
regional setting, accompanied by clear trade-offs elsewhere.

| Preset | v2 species accuracy | v3 species accuracy | Change |
|---|---:|---:|---:|
| Northern Europe | 68.71% | 70.79% | +2.08 pp |
| Europe | 66.96% | 68.95% | +1.99 pp |
| Global | 59.36% | 58.43% | −0.93 pp |

The primary comparison uses identical legacy lists in both releases. All-image
accuracy includes 8,042 images from 16 species absent from the model vocabulary;
known-truth accuracy uses the remaining 50,598 images (86.29% coverage). Macro-F1
follows the pinned metric implementation, including predicted-only classes.
Predictions are unthresholded, and the lists were not tuned to these results.

## Inference speed

The unadapted v2 CPU API failed here with `expected scalar type BFloat16 but found
Float`. Its CPU bars therefore show an explicitly labelled caller-side adapter:
`predictor.preproc = lambda x: original_preproc(x).float()`. It preserves the
preprocessed values while matching the model's input dtype. V2 GPU and all quality
measurements use the original published path unchanged.

![CPU latency and GPU throughput by model pipeline and region](assets/mambo-release-speed.svg)

For northern Europe, the measured medians are:

| Pipeline | CPU, one image | GPU, one image | GPU, batch 8 | GPU, batch 32 |
|---|---:|---:|---:|---:|
| v2 PyTorch (CPU input cast) | 796 ms | 22 ms | 44.8 images/s | 83.5 images/s |
| v3 PyTorch | 148 ms | 37 ms | 46.6 images/s | 44.5 images/s |
| v3 ONNX | 104 ms | 33 ms | 42.8 images/s | 42.4 images/s |

V3 substantially improves CPU inference. V2 retains lower single-image GPU latency
and higher batch-32 throughput; batch-8 throughput is much closer. The released
resolution, backbone and precision choices contribute to these practical trade-offs.

Measurements include image decoding, preprocessing, classification and completed
CPU results. Each configuration has three fresh-process trials on the same seeded
image bank. Whiskers show the range of trial medians. The compact source data also
includes CPU batch-8 results and timings for the updated v3 lists.

## Memory and startup

![Host process memory and loading time for the released pipelines](assets/mambo-release-resources.svg)

Median peak host memory during CPU execution falls from **4,123 MiB** for adapted
v2 to **1,573 MiB** for v3 PyTorch and **587 MiB** for v3 ONNX. Loading plus the first
CPU prediction falls from **209 s** to **40.5 s** and **0.38 s**, respectively.

Native GPU allocator peaks during the batch sweep are:

| Pipeline | Peak allocated GPU memory | Peak reserved GPU memory |
|---|---:|---:|
| v2 PyTorch | 1,936 MiB | 2,072 MiB |
| v3 PyTorch | 865 MiB | 1,348 MiB |
| v3 ONNX | Not measured comparably | Not measured comparably |

These native allocator counters exclude CUDA allocations outside PyTorch. ONNX
device snapshots are not comparable allocator peaks and are not presented as such.

Host RSS includes model loading and the full batch sweep. Startup uses cached local
files and includes predictor construction, classifier initialization and the first
completed prediction; process launch and explicit runtime setup are excluded.
Keep a predictor alive across requests to amortize loading. CPU sweeps use batches
1/8; GPU sweeps add 32. RSS is host memory, not GPU VRAM.

## Updated v3 European lists

![Accuracy changes from legacy to updated European presets](assets/mambo-release-preset-delta.svg)

Updated northern Europe adds 222 candidate species, and updated Europe adds 72,
with no removals. Their Flemming species accuracies are 70.32% and 68.81%, compared
with 70.79% and 68.95% for the legacy lists. These are small costs for broader
occurrence coverage. Species macro-F1 also changes from **0.2545 to 0.2367** for
northern Europe, and **0.1993 to 0.1975** for Europe under the pinned policy.
The [preset catalogue](model-presets.md) explains geographic
scope and construction; select a preset for where it will be used.

## Reproduce and interpret

The [comparison workflow](../dev/releases/mambo_v3/release-comparison.md) pins the
historical source, head weights, external BioCLIP-2 backbone and metric revision.
[Chart source data](assets/mambo-release-comparison.json) contains compact metrics,
timing summaries and evidence hashes, and can regenerate the figures without
image data. Raw predictions and timing observations remain under `local-evidence/`.

V2 uses its original release API, including a 512-pixel initial resize, BioCLIP
preprocessing to 224 pixels and CUDA float16 autocast. V3 uses its 384-pixel release
recipe and FP32. These choices are part of the real-world pipeline comparison.
The v2 historical code is unchanged, running with the documented contemporary
comparison environment. Both releases use the same PyTorch/CUDA installation;
ONNX Runtime is 1.30.0. Hardware is an i7-12800H / RTX 3080 Ti Laptop GPU (16 GB),
on AC power under Linux/WSL2, with four CPU threads and TF32 disabled.

The earlier [v3 qualification report](mambo-v3-evaluation.md) covers embedding-mode
consistency, additional timing detail and installed-package checks. In-domain
comparison remains UCloud work with the original test split. Provenance/licensing
and broader OS qualification still precede publication. Nothing is published by
this comparison.
