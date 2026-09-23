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

The primary comparison uses identical legacy lists in both releases. Every
predictive metric is computed by pinned `mini_metrics` at commit
`70cc69adc05362863439277048e06386c1f885e1`; the chart extracts these fields:

| Display | `metrics.json` source | Averaging and population |
|---|---|---|
| Species/genus/family accuracy | `all.micro_accuracy["0"/"1"/"2"]` | Micro: each of the 58,640 images has equal weight |
| Species macro-F1 | `all.f1["0"]` | Equal weight over the union of true and predicted species |
| Known-truth accuracy | `known.micro_accuracy[rank]` | Micro, restricted to truth in the active preset vocabulary |

All calls use `threshold=0`, `optimal=False`, `simple=True`,
`hierarchical=False`; the main chart uses `known_only=False`. Every image receives
a prediction; no threshold is fitted. The class list limits predictions, **not the
evaluation population**. All 522 Flemming species remain in the main results,
including 8,042 images from 16 species outside the vocabulary. The known-only
species denominator is 50,598 images from 506 species for every compared list.
Known membership is determined separately at each taxonomic rank.

The plain `accuracy` field in this mini_metrics revision is **macro** accuracy;
we explicitly extract `micro_accuracy`. Both macro accuracy and known-only metrics
are retained in the metric files. The original direct CSV accuracy calculation
has been replaced by mini_metrics; recomputation leaves the reported values unchanged.

Macro-F1 includes species predicted despite having no ground-truth images, but
excludes species with neither truth nor predictions. Thus its class denominator can
change between pipelines: northern Europe has 1,221 active classes for v2 and 1,308
for v3, versus 522 ground-truth species. It is not a mean over just the 506 known
truth species, nor over every species in the preset. These results use the pinned
metric policy; lists and thresholds were not tuned to Flemming.

## Inference speed

The unadapted v2 CPU API failed here with `expected scalar type BFloat16 but found
Float`. Its CPU bars therefore show an explicitly labelled caller-side adapter:
`predictor.preproc = lambda x: original_preproc(x).float()`. It preserves the
preprocessed values while matching the model's input dtype. V2 GPU and all quality
measurements use the original published path unchanged.

![CPU and GPU throughput by model pipeline and region](assets/mambo-release-speed.svg)

For northern Europe, the measured medians are:

| Pipeline | CPU, batch 1 | GPU, batch 1 | GPU, batch 8 | GPU, batch 32 |
|---|---:|---:|---:|---:|
| v2 PyTorch (CPU input cast) | 1.26 | 45.2 | 44.8 | 83.5 |
| v3 PyTorch | 6.74 | 27.1 | 46.6 | 44.5 |
| v3 ONNX | 9.64 | 30.4 | 42.8 | 42.4 |

All speed columns and panels use **images per second; higher is better**.

V3 substantially improves CPU inference. V2 retains lower single-image GPU latency
and higher batch-32 throughput; batch-8 throughput is much closer. The released
resolution, backbone and precision choices contribute to these practical trade-offs.

Measurements include image decoding, preprocessing, classification and completed
CPU results. Each configuration has three fresh-process trials on the same seeded
image bank. Whiskers show the range of trial medians. The compact source data also
includes CPU batch-8 results and timings for the updated v3 lists.

### Why v3 throughput plateaus

The adapter sends each batch as one NCHW tensor to the backend; the benchmark sets
its batch limit to 32. It does not silently split batches into single-image calls.
However, it decodes and preprocesses each image serially on CPU, then runs the
model, with no CPU/GPU overlap. The existing three-trial northern-Europe timings
separate these boundaries:

| Backend | Batch | CPU preparation, ms/batch | Prepared backend, ms/batch | End-to-end, images/s |
|---|---:|---:|---:|---:|
| PyTorch | 1 | 14.9 | 18.5 | 27.1 |
| PyTorch | 8 | 126.3 | 46.7 | 46.6 |
| PyTorch | 32 | 519.5 | 181.4 | 44.5 |
| ONNX | 1 | 16.6 | 13.4 | 30.4 |
| ONNX | 8 | 134.5 | 50.6 | 42.8 |
| ONNX | 32 | 559.4 | 180.6 | 42.4 |

These are separately timed medians, not additive profiler spans. The prepared
boundary includes transfers and completed CPU leaf scores, excluding decoding and
hierarchy reduction. CPU preparation accounts for roughly 72–74% of end-to-end
batch-32 time. The prepared backend also shows little throughput gain beyond batch
8, so preprocessing alone does not explain the entire plateau. Kernel profiling
would be needed to explain that remaining hardware/runtime behavior; these data do
not establish a specific GPU bottleneck.

A useful next optimization is to qualify parallel image preparation and overlap
with inference while preserving exact input values. The current end-to-end figures
remain the measured release behavior; they are not GPU-only throughput claims.

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
