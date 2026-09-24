# MAMBO deployment defaults and release comparison

Ordinary inference stays single-view. Enabling TTA with `tta=True` or bare `--tta`
selects **padded scale**: the original image plus views with 8% and 15% edge padding.
Each view uses unchanged preprocessing; FP32 leaf logits are averaged before
class-list filtering and hierarchy reduction. Use `tta="padded_scale"` to pin the
recipe explicitly, or `tta=False` / `--tta none` to disable it.

This is the best measured **accuracy/cost compromise** among the explored policies,
not a claim of optimality for every metric. It had the highest subset macro
accuracy, used only three views and was faster than D4 or padded rotations.
Padded rotations had higher subset macro-F1. The [candidate study](mambo-tta.md)
retains all 13 recipes, exact parameters, sources and their trade-offs.

## Full Flemming comparison

Both V3 backends were evaluated on all **58,640 images / 522 truth species**, with
and without TTA at automatic GPU precision. Main comparisons use identical legacy
northern-Europe, Europe and global vocabularies across V2/V3. Updated European lists are included in the
complete tables. All predictive metrics come from `mini_metrics` commit
`70cc69adc05362863439277048e06386c1f885e1`, using `threshold=0`, `optimal=False`,
`simple=True`, `hierarchical=False`.

The primary charts include **all truth** at each rank, including out-of-vocabulary
taxa. Macro accuracy gives equal weight to ground-truth taxa at that rank; macro-F1/precision
also reflect predicted-only classes under the package's metric policy. See the
[metric definitions](mambo-release-comparison.md#prediction-quality). The TTA
recipe was selected using a 1,024-image subset of this same dataset: the full
results are descriptive comparisons, **not independent validation**.

### Species

![Full-data species quality](assets/mambo-defaults-quality-all.svg)

| Preset | Pipeline | Macro accuracy | Macro-F1 | Micro accuracy |
|---|---|---:|---:|---:|
| Northern Europe | MAMBO v2 | 68.52% | 0.2575 | 68.71% |
| Northern Europe | V3 PyTorch | 71.25% | 0.2543 | 70.78% |
| Northern Europe | V3 ONNX | 71.24% | 0.2543 | 70.78% |
| Northern Europe | V3 PyTorch + TTA | 73.95% | 0.2935 | 73.19% |
| Northern Europe | V3 ONNX + TTA | 73.98% | 0.2944 | 73.19% |
| Europe | MAMBO v2 | 66.04% | 0.2000 | 66.96% |
| Europe | V3 PyTorch | 69.05% | 0.1997 | 68.94% |
| Europe | V3 ONNX | 69.06% | 0.1997 | 68.95% |
| Europe | V3 PyTorch + TTA | 71.81% | 0.2276 | 71.57% |
| Europe | V3 ONNX + TTA | 71.85% | 0.2283 | 71.56% |
| Global | MAMBO v2 | 57.21% | 0.0899 | 59.36% |
| Global | V3 PyTorch | 58.01% | 0.0973 | 58.42% |
| Global | V3 ONNX | 58.03% | 0.0973 | 58.43% |
| Global | V3 PyTorch + TTA | 61.73% | 0.1110 | 62.63% |
| Global | V3 ONNX + TTA | 61.74% | 0.1112 | 62.61% |

Padded-scale TTA raises northern-Europe macro accuracy by **2.70 / 2.74 percentage
points** for PyTorch / ONNX, with macro-F1 rising to **0.2935 / 0.2944**. Species
macro accuracy, macro-F1 and micro accuracy exceed V2 for all three primary lists.
Genus and family results follow below; ordinary V3 loses family macro accuracy
against V2, while TTA recovers it.

Updated European presets (the same inference, different candidate lists):

| Preset | Backend | Ordinary macro accuracy / F1 | TTA macro accuracy / F1 |
|---|---|---:|---:|
| `north_europe_v3` | torch | 70.46% / 0.2368 | 73.10% / 0.2713 |
| `north_europe_v3` | onnx | 70.49% / 0.2365 | 73.12% / 0.2721 |
| `europe_v3` | torch | 68.86% / 0.1979 | 71.66% / 0.2253 |
| `europe_v3` | onnx | 68.88% / 0.1978 | 71.70% / 0.2261 |

![Species quality restricted to known truth](assets/mambo-defaults-quality-known.svg)

### Genus

![Full-data genus quality](assets/mambo-defaults-quality-genus-all.svg)

| Preset | Pipeline | Macro accuracy | Macro-F1 | Micro accuracy |
|---|---|---:|---:|---:|
| Northern Europe | MAMBO v2 | 78.90% | 0.3169 | 79.23% |
| Northern Europe | V3 PyTorch | 80.53% | 0.3204 | 79.37% |
| Northern Europe | V3 ONNX | 80.50% | 0.3212 | 79.36% |
| Northern Europe | V3 PyTorch + TTA | 83.04% | 0.3532 | 81.59% |
| Northern Europe | V3 ONNX + TTA | 83.04% | 0.3536 | 81.59% |
| Europe | MAMBO v2 | 77.29% | 0.2553 | 77.64% |
| Europe | V3 PyTorch | 79.28% | 0.2552 | 77.94% |
| Europe | V3 ONNX | 79.28% | 0.2557 | 77.94% |
| Europe | V3 PyTorch + TTA | 82.93% | 0.2820 | 80.40% |
| Europe | V3 ONNX + TTA | 82.97% | 0.2828 | 80.40% |
| Global | MAMBO v2 | 71.68% | 0.1221 | 72.85% |
| Global | V3 PyTorch | 71.30% | 0.1268 | 70.57% |
| Global | V3 ONNX | 71.33% | 0.1269 | 70.58% |
| Global | V3 PyTorch + TTA | 75.10% | 0.1457 | 74.52% |
| Global | V3 ONNX + TTA | 75.10% | 0.1461 | 74.52% |
| `north_europe_v3` | V3 PyTorch | 80.01% | 0.3041 | 78.88% |
| `north_europe_v3` | V3 ONNX | 79.98% | 0.3049 | 78.89% |
| `north_europe_v3` | V3 PyTorch + TTA | 82.78% | 0.3337 | 81.16% |
| `north_europe_v3` | V3 ONNX + TTA | 82.78% | 0.3341 | 81.16% |
| `europe_v3` | V3 PyTorch | 79.08% | 0.2528 | 77.79% |
| `europe_v3` | V3 ONNX | 79.08% | 0.2530 | 77.80% |
| `europe_v3` | V3 PyTorch + TTA | 82.68% | 0.2803 | 80.29% |
| `europe_v3` | V3 ONNX + TTA | 82.72% | 0.2809 | 80.29% |

Northern-Europe genus macro accuracy rises from **78.90% (V2)** to
**80.53% / 80.50% (ordinary V3)** and **83.04% / 83.04% (TTA)** for PyTorch / ONNX.
Known-genus results contain 58,639 or 58,640 images depending on the preset.

<details>
<summary>Known-truth genus metrics</summary>

![Known-truth genus quality](assets/mambo-defaults-quality-genus-known.svg)

</details>

### Family

![Full-data family quality](assets/mambo-defaults-quality-family-all.svg)

| Preset | Pipeline | Macro accuracy | Macro-F1 | Micro accuracy |
|---|---|---:|---:|---:|
| Northern Europe | MAMBO v2 | 84.40% | 0.2691 | 94.49% |
| Northern Europe | V3 PyTorch | 81.05% | 0.2804 | 92.97% |
| Northern Europe | V3 ONNX | 81.06% | 0.2808 | 92.99% |
| Northern Europe | V3 PyTorch + TTA | 85.72% | 0.2967 | 94.77% |
| Northern Europe | V3 ONNX + TTA | 85.73% | 0.2967 | 94.77% |
| Europe | MAMBO v2 | 83.54% | 0.2513 | 94.23% |
| Europe | V3 PyTorch | 80.62% | 0.2556 | 92.85% |
| Europe | V3 ONNX | 80.63% | 0.2556 | 92.86% |
| Europe | V3 PyTorch + TTA | 85.63% | 0.2753 | 94.72% |
| Europe | V3 ONNX + TTA | 85.64% | 0.2755 | 94.73% |
| Global | MAMBO v2 | 80.70% | 0.2052 | 92.65% |
| Global | V3 PyTorch | 78.42% | 0.1936 | 90.30% |
| Global | V3 ONNX | 78.42% | 0.1938 | 90.31% |
| Global | V3 PyTorch + TTA | 81.46% | 0.2183 | 92.75% |
| Global | V3 ONNX + TTA | 81.45% | 0.2183 | 92.75% |
| `north_europe_v3` | V3 PyTorch | 80.66% | 0.2732 | 92.89% |
| `north_europe_v3` | V3 ONNX | 80.67% | 0.2733 | 92.89% |
| `north_europe_v3` | V3 PyTorch + TTA | 83.47% | 0.2778 | 94.69% |
| `north_europe_v3` | V3 ONNX + TTA | 83.47% | 0.2778 | 94.70% |
| `europe_v3` | V3 PyTorch | 80.48% | 0.2539 | 92.53% |
| `europe_v3` | V3 ONNX | 80.52% | 0.2541 | 92.55% |
| `europe_v3` | V3 PyTorch + TTA | 83.35% | 0.2729 | 94.50% |
| `europe_v3` | V3 ONNX + TTA | 83.37% | 0.2730 | 94.52% |

Family macro accuracy exposes a regression that species-only reporting missed:
northern Europe falls from **84.40% (V2)** to **81.05% / 81.06% (ordinary V3)**.
TTA recovers it to **85.72% / 85.73%**, while macro-F1 reaches **0.2967** for both
backends, versus **0.2691** for V2. All 58,640 images have known family truth.

<details>
<summary>Known-truth family metrics</summary>

![Known-truth family quality](assets/mambo-defaults-quality-family-known.svg)

</details>

The [complete metric table](assets/mambo-defaults-metrics.csv) retains macro
accuracy, precision, recall and F1, micro accuracy, Theil U and coverage, at all
three ranks and for both all/known truth. Known-only species results contain
50,598 images. Known genus contains 58,639–58,640 images by preset; known family
contains all 58,640. These are already-computed `mini_metrics` results; the added
rank views do not change the model runs, score extraction or threshold policy.

## Inference speed and memory

![CPU and GPU throughput by batch size](assets/mambo-defaults-speed.svg)

Northern Europe, images/second:

| Pipeline | CPU B1 | CPU B8 | GPU B1 | GPU B8 | GPU B32 |
|---|---:|---:|---:|---:|---:|
| MAMBO v2 | 1.26 | 1.39 | 45.2 | 44.8 | 83.5 |
| V3 PyTorch | 5.70 | 7.84 | 29.8 | 126.7 | 136.2 |
| V3 ONNX | 10.08 | 10.81 | 46.7 | 114.1 | 111.3 |
| V3 PyTorch + TTA | 2.29 | 2.63 | 8.5 | 42.7 | 50.4 |
| V3 ONNX + TTA | 3.56 | 3.66 | 16.8 | 41.7 | 39.4 |

At GPU batch 32, TTA takes about **2.70× the native time / 2.82× the ONNX time**
per image versus ordinary V3. It is slower than V2 on GPU at that batch size,
while still faster than V2 in these CPU measurements. These are measured pipeline
trade-offs, not a uniform speed or quality ranking across every setting.

These are complete-pipeline **images per second**, including decoding, preparation,
inference and completed CPU results. TTA runs all three views inside that boundary.
The [acceleration study](mambo-accelerated-deployment.md) retains the earlier FP32
comparison and the AMP/preparation improvements. Both backends here use automatic
precision: FP16 backbone / FP32 head on native CUDA,
TF32 execution of the standard FP32 ONNX graph on CUDA, and FP32 on CPU.

TTA timings use three fresh-process trials, two warmups and seven observations
per cell, the same seeded 32-image bank and four preparation/runtime CPU threads
as the retained V2 and ordinary V3 measurements. CPU batches are 1/8; GPU batches
are 1/8/32. Throughput uses the median of 21 observations; error bars show the
range of trial medians. The hardware is an i7-12800H / RTX 3080 Ti Laptop (16 GB),
Linux/WSL2 on AC power. Campaigns were run separately, so laptop conditions can
vary. V2 CPU uses the documented caller-side float32 input cast.

![Peak host memory](assets/mambo-defaults-memory.svg)

| Pipeline | CPU host MiB | GPU-run host MiB | CPU first-use s | GPU first-use s | Native GPU allocated MiB |
|---|---:|---:|---:|---:|---:|
| MAMBO v2 | 4123 | 4123 | 209.40 | 209.60 | 1936 |
| V3 PyTorch | 1618 | 2512 | 42.04 | 42.96 | 598 |
| V3 ONNX | 689 | 1619 | 0.37 | 1.46 | — |
| V3 PyTorch + TTA | 1639 | 2442 | 43.27 | 44.05 | 597 |
| V3 ONNX + TTA | 671 | 1629 | 0.53 | 1.72 | — |

First use includes construction, lazy loading and the first prediction, excluding
interpreter launch and explicit runtime setup. Native classifier initialization
still dominates startup; reusing a predictor avoids repeating it. Loading and
allocator variation influence process peaks.

Peak host RSS includes model loading and the entire batch sweep. Native allocated
GPU memory is a PyTorch allocator counter, not total VRAM; an equivalent ONNX peak
is unavailable. TTA holds the decoded source batch plus the current prepared view;
it does not multiply the model batch size by the number of views. Host memory
therefore depends on source image dimensions as well as batch size.

Use ordinary V3 for throughput-sensitive workflows and enable TTA when the measured
accuracy/cost trade-off fits the application. Reuse a loaded predictor. Tune
`preprocess_workers` independently of ONNX `threads`; the [loading study](mambo-loading-scaling.md)
explains why higher batches or more workers are not automatically faster.
In-domain UCloud evaluation, other operating systems and downstream embedding
quality remain separate qualification work.

## Reproduce

```sh
python -m dev.releases.mambo_v3.run_local full --precision auto --tta padded_scale \
  --python /path/to/gpu-env/bin/python --bundle /path/to/bundle \
  --manifest /path/to/flemming-manifest.json --root /path/to/flemming \
  --output /path/to/new-tta-quality
/path/to/pinned-metrics-env/bin/python -m dev.releases.mambo_v3.metrics \
  --collection /path/to/new-tta-quality
python -m dev.releases.mambo_v3.benchmark_acceleration --tta padded_scale \
  --python /path/to/gpu-env/bin/python --bundle /path/to/bundle \
  --manifest /path/to/flemming-manifest.json --root /path/to/flemming \
  --output /path/to/new-tta-timings
python -m dev.releases.mambo_v3.defaults_report \
  --baseline docs/assets/mambo-accelerated-comparison.json \
  --reference docs/assets/mambo-release-comparison.json \
  --quality /path/to/new-tta-quality --performance /path/to/new-tta-timings \
  --output /path/to/charts
```

Run timing processes sequentially without other CPU/GPU jobs. All reported speed
is end-to-end; the benchmark's separately labelled prepared-input diagnostic is
single-view even when TTA is enabled, and is not used in these comparisons.
The [compact evidence](assets/mambo-defaults-comparison.json) includes source hashes
and regenerates the figures with `defaults_report --data FILE --output DIRECTORY`.
