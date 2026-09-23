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

The primary chart includes **all truth**, including out-of-vocabulary species.
Macro accuracy gives equal weight to ground-truth species; macro-F1/precision
also reflect predicted-only classes under the package's metric policy. See the
[metric definitions](mambo-release-comparison.md#prediction-quality). The TTA
recipe was selected using a 1,024-image subset of this same dataset: the full
results are descriptive comparisons, **not independent validation**.

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
The full export retains other ranks; this is not a claim that V3 wins every metric.

Updated European presets (the same inference, different candidate lists):

| Preset | Backend | Ordinary macro accuracy / F1 | TTA macro accuracy / F1 |
|---|---|---:|---:|
| `north_europe_v3` | torch | 70.46% / 0.2368 | 73.10% / 0.2713 |
| `north_europe_v3` | onnx | 70.49% / 0.2365 | 73.12% / 0.2721 |
| `europe_v3` | torch | 68.86% / 0.1979 | 71.66% / 0.2253 |
| `europe_v3` | onnx | 68.88% / 0.1978 | 71.70% / 0.2261 |

![Species quality restricted to known truth](assets/mambo-defaults-quality-known.svg)

The [complete metric table](assets/mambo-defaults-metrics.csv) retains macro
accuracy, precision, recall and F1, micro accuracy, Theil U and coverage, at all
three ranks and for both all/known truth. Known-only species results contain
50,598 images; membership is checked separately at genus and family level.

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
