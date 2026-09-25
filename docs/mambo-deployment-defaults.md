# MAMBO regional choice and historical padded-scale evidence

The [deployment README](../deployment/README.md#release-comparison) contains the
current model, timing and rotation-and-padding TTA comparisons. This page retains
the evidence for the northern-Europe preset choice and aggregate regional effect.

These historical results use `padded_scale`: original plus 8% and 15% edge-padded
views, with FP32 leaf logits averaged before filtering and hierarchy reduction.
It was the earlier accuracy/cost choice; it is **not the current enabled-TTA
default**. The [candidate study](mambo-tta.md) records its selection.

## Northern-Europe preset choice

Legacy `north_europe` retains the 1,977-species V2 vocabulary. The explicit
`north_europe_v3` option adds 222 species without removals, using the same
geographic filter and a minimum of 3 regional records instead of 26, plus a global
minimum of 25. Both cover the same 50,598 species-labelled Flemming images (86.29%).

Holding PyTorch inference fixed, macro accuracy was:

| Rank | Legacy | Updated | Legacy + padded-scale TTA | Updated + padded-scale TTA |
|---|---:|---:|---:|---:|
| Species | 71.25% | 70.46% | 73.95% | 73.10% |
| Genus | 80.53% | 80.01% | 83.04% | 82.78% |
| Family | 81.05% | 80.66% | 85.72% | 83.47% |

ONNX, macro-F1 and micro accuracy showed the same preference for the legacy list.
The additions permit more plausible regional species, but Flemming contains no
examples of them: their recognition benefit is unmeasured. Legacy therefore
combines better measured discrimination with V2 comparability. The updated list
remains an explicit broader option. The V3 API/CLI default is global (`full`);
see [geographic definitions](model-presets.md).

### Regional filtering effect

![Paired regional gains across pipelines](assets/mambo-defaults-regional-effect.svg)

Each point compares two presets **within the same pipeline**. Black marks are the
median of five changes (V2, V3 PyTorch/ONNX, and each V3 backend with padded-scale
TTA); grey lines show their min–max range. These related pipelines are not
independent replicates; no confidence intervals or significance are implied.
All comparisons retain the same 58,640 images, including out-of-vocabulary truth.
This supports regional filtering for northern-European images, not use of that
vocabulary elsewhere.

## Population, metrics and retained results

The [complete CSV](assets/mambo-defaults-metrics.csv) retains all presets,
species/genus/family, all/known truth, macro accuracy/precision/recall/F1, micro
accuracy, Theil U and coverage. The [source JSON](assets/mambo-defaults-comparison.json)
also retains historical timings, memory measurements and provenance hashes.

Predictive metrics used `mini_metrics` commit
`70cc69adc05362863439277048e06386c1f885e1` with `threshold=0`, `optimal=False`,
`simple=True` and `hierarchical=False`. The full population is 58,640 images /
522 truth species. Legacy northern-Europe known-truth counts are 50,598 species,
58,639 genus and 58,640 family images. See [metric definitions](mambo-release-comparison.md#prediction-quality)
for macro denominators. TTA selection used 1,024 images from this same dataset,
so these are descriptive comparisons, not independent validation.

[Threshold](mambo-confidence-thresholds.md) and [tail](mambo-tail-metrics.md)
analyses instead use 5,852 calibration images and 52,788 reporting images.
Do not mix those populations with the full-data results here. The
[family audit](mambo-family-precision.md) explains the rare predicted-only classes
that affect macro precision and F1. Current deployment figures retain both
thresholded/unthresholded results, coverage and support >5 comparisons.

Historical laptop timing used an i7-12800H / RTX 3080 Ti Laptop, four CPU threads,
a seeded 32-image bank, three fresh processes, two warmups and seven observations
per cell (CPU batches 1/8; GPU 1/8/32). Reported throughput includes decoding
through completed CPU results, all three TTA views; it excludes the separate
single-view prepared-input diagnostic. Host RSS covers loading and the batch
sweep; PyTorch allocated GPU bytes are not total VRAM or an ONNX measurement.
First use excludes interpreter launch and explicit runtime setup. V2 CPU used
the documented float32 input cast. Use the current README for adoption decisions.

## Reproduce the historical presentation

All retired per-rank/regional, speed and memory plots can be regenerated from the
retained JSON without inference or recalculating metrics:

```sh
python -m dev.releases.mambo_v3.defaults_report \
  --data docs/assets/mambo-defaults-comparison.json \
  --output /tmp/mambo-historical-defaults
```

The [release comparison runbook](../dev/releases/mambo_v3/release-comparison.md)
describes collecting new evidence. Pin `padded_scale` explicitly when reproducing
this historical recipe; bare `tta=True` selects the current default.
