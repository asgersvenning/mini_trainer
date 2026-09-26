# Deployment comparison evidence

Complete results supporting the [integration guide](../deployment/README.md).
The complementary [in-domain/HPC evidence](mambo-indomain-evidence.md) uses the same
calibration and support policy on a separate image domain; the tables below remain Flemming-only.
This reference retains both confidence settings, full and truncated support,
all three ranks, coverage, timing ranges and historical-study boundaries.

Results use the **same 52,788 Flemming reporting images** for both confidence
settings, including out-of-vocabulary truth. Calibrated thresholds were fitted on
5,852 separate images using pinned `mini_metrics` Macro-F1, independently for each
pipeline and rank. No-threshold results use threshold zero. All comparisons use
the shared legacy `north_europe` preset; TTA uses `rotation30_pad25_3`.
V3 uses automatic GPU precision. Deployment defaults remain threshold zero.

In each metric cell, values are **full support / support >5**. Full support retains
each model’s complete class domain, including predicted-only classes. Support >5
retains classes with more than five truth instances and **accepted predictions in
every pipeline**, separately for each confidence setting. These class sets can differ
between settings; truncation is a change in averaging domain, not improved predictions.
Coverage is the percentage of reporting images accepted, and is identical for both
averaging domains. No evaluation rows are dropped; per-class FP/FN remain intact.

### Species

| Pipeline | Confidence | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |
|---|---|---:|---:|---:|
| MAMBO v2 | None | 68.56% / 79.01% | 0.2620 / 0.7836 | 100.00% |
| MAMBO v2 | Calibrated | 84.65% / 95.23% | 0.4467 / 0.7800 | 69.73% |
| V3 PyTorch | None | 71.36% / 81.59% | 0.2593 / 0.8066 | 100.00% |
| V3 PyTorch | Calibrated | 86.69% / 96.36% | 0.5081 / 0.8016 | 70.81% |
| V3 ONNX | None | 71.34% / 81.64% | 0.2594 / 0.8068 | 100.00% |
| V3 ONNX | Calibrated | 86.95% / 96.44% | 0.5100 / 0.8001 | 70.45% |
| V3 PyTorch + TTA | None | 74.94% / 86.53% | 0.3201 / 0.8574 | 100.00% |
| V3 PyTorch + TTA | Calibrated | 86.34% / 96.39% | 0.5661 / 0.8619 | 81.16% |
| V3 ONNX + TTA | None | 74.93% / 86.55% | 0.3197 / 0.8575 | 100.00% |
| V3 ONNX + TTA | Calibrated | 86.35% / 96.37% | 0.5676 / 0.8627 | 81.33% |

### Genus

| Pipeline | Confidence | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |
|---|---|---:|---:|---:|
| MAMBO v2 | None | 78.94% / 82.07% | 0.3213 / 0.7942 | 100.00% |
| MAMBO v2 | Calibrated | 95.34% / 97.55% | 0.5869 / 0.7875 | 69.81% |
| V3 PyTorch | None | 80.53% / 83.83% | 0.3230 / 0.8057 | 100.00% |
| V3 PyTorch | Calibrated | 95.48% / 97.31% | 0.6019 / 0.8311 | 75.75% |
| V3 ONNX | None | 80.54% / 83.84% | 0.3245 / 0.8055 | 100.00% |
| V3 ONNX | Calibrated | 95.52% / 97.39% | 0.6043 / 0.8290 | 75.37% |
| V3 PyTorch + TTA | None | 84.73% / 88.13% | 0.3873 / 0.8552 | 100.00% |
| V3 PyTorch + TTA | Calibrated | 95.77% / 97.55% | 0.6776 / 0.8858 | 84.32% |
| V3 ONNX + TTA | None | 84.73% / 88.12% | 0.3877 / 0.8549 | 100.00% |
| V3 ONNX + TTA | Calibrated | 95.79% / 97.56% | 0.6760 / 0.8857 | 84.31% |

### Family

| Pipeline | Confidence | Macro accuracy (full / >5) | Macro-F1 (full / >5) | Coverage |
|---|---|---:|---:|---:|
| MAMBO v2 | None | 84.35% / 87.00% | 0.2697 / 0.8238 | 100.00% |
| MAMBO v2 | Calibrated | 99.64% / 99.58% | 0.6545 / 0.8021 | 77.44% |
| V3 PyTorch | None | 81.05% / 85.70% | 0.2805 / 0.7831 | 100.00% |
| V3 PyTorch | Calibrated | 99.33% / 99.22% | 0.5807 / 0.8161 | 73.06% |
| V3 ONNX | None | 81.06% / 85.72% | 0.2809 / 0.7843 | 100.00% |
| V3 ONNX | Calibrated | 99.33% / 99.22% | 0.5816 / 0.8174 | 73.26% |
| V3 PyTorch + TTA | None | 86.37% / 89.32% | 0.3592 / 0.8506 | 100.00% |
| V3 PyTorch + TTA | Calibrated | 98.92% / 98.75% | 0.7074 / 0.8802 | 82.09% |
| V3 ONNX + TTA | None | 86.37% / 89.32% | 0.3591 / 0.8502 | 100.00% |
| V3 ONNX + TTA | Calibrated | 98.92% / 98.75% | 0.7077 / 0.8806 | 82.13% |

### Support retained and comparison figure

Counts below describe truth classes outside the truncated average, not rejected images.
The prediction range counts accepted predictions into excluded classes, divided by all
52,788 reporting images. Truth and prediction counts must not be added.

| Confidence | Rank | Shared classes | Truth outside: images / % | Accepted predictions outside: images / % (pipeline range) |
|---|---|---:|---:|---:|
| None | Species | 311 | 8,079 / 15.30% | 11,060–13,504 / 20.95%–25.58% |
| None | Genus | 242 | 221 / 0.42% | 6,814–8,015 / 12.91%–15.18% |
| None | Family | 20 | 5 / 0.01% | 385–897 / 0.73%–1.70% |
| Calibrated | Species | 273 | 10,001 / 18.95% | 5,780–7,678 / 10.95%–14.54% |
| Calibrated | Genus | 215 | 5,920 / 11.21% | 2,693–4,684 / 5.10%–8.87% |
| Calibrated | Family | 19 | 18 / 0.03% | 11–36 / 0.02%–0.07% |

![Both confidence settings, full and truncated macro metrics, and coverage](assets/mambo-promoted-quality.svg)

The [metric export](assets/mambo-promoted-tail.csv) retains macro precision,
recall and support cutoffs −1/5/10/20; the [JSON](assets/mambo-promoted-tail.json)
records exact class sets. The [threshold evidence](assets/mambo-promoted-thresholds.json)
contains per-rank calibrated thresholds, all full-support metrics and prediction hashes.
These are optional study operating points, not automatic deployment thresholds;
the CLI's `--threshold` applies one scalar to all ranks. The
[recipe comparison](mambo-composed-tta.md) shows the prior and new recipes
at matched coverage. Thresholding and truncation can change rankings; neither
should be confused with an improvement in underlying predictions.

Regional filtering improves results on Flemming. The [single regional-effect figure](mambo-deployment-defaults.md#regional-filtering-effect)
summarizes global → Europe → northern Europe across pipelines and ranks using
the earlier padded-scale TTA; the new recipe has been fully evaluated only for northern Europe.
We recommend legacy `north_europe` here: the updated list adds 222 species but
no Flemming species coverage, and lowers measured accuracy/F1. It remains available
as `north_europe_v3` for broader eligibility; the V3 API/CLI default is now global (`full`).

Speed remains **images per second**, measured end to end on an i7-12800H / RTX
3080 Ti Laptop, with four preparation/runtime CPU threads. CPU uses FP32.
New TTA timings use three fresh processes per backend/device,
with seven observations per cell. V2 and single-view V3 reuse the same earlier
image-bank measurements; laptop conditions can vary between campaigns.

| Pipeline | CPU B1 | GPU B1 | GPU B8 | GPU B32 |
|---|---:|---:|---:|---:|
| MAMBO v2 | 1.26 | 45.19 | 44.85 | 83.49 |
| V3 PyTorch | 5.70 | 29.84 | 126.73 | 136.24 |
| V3 ONNX | 10.08 | 46.71 | 114.11 | 111.27 |
| V3 PyTorch + TTA | 2.04 | 10.10 | 41.74 | 50.89 |
| V3 ONNX + TTA | 3.39 | 16.50 | 39.40 | 38.27 |

![CPU and GPU throughput](assets/mambo-promoted-speed.svg)

The [timing evidence](assets/mambo-promoted-speed.json) retains trial ranges
and process-memory measurements. The [earlier comparison](mambo-deployment-defaults.md)
uses the previous padded-scale TTA; the [frequency curves](mambo-frequency-comparison.md)
compare single-view models only. Neither measures the new recipe.
The [loading study](mambo-loading-scaling.md) explains scheduling limits;
`preprocess_workers` / `--preprocess-workers` tunes preparation separately from
ONNX runtime `threads` and defaults to it.

Use ordinary V3 for throughput and enable TTA when its accuracy/cost trade-off fits.
Recipe exploration used this Flemming dataset, so these results are descriptive,
not independent validation. The complementary [in-domain evaluation](mambo-indomain-evidence.md) is complete.
[Installed qualification](../dev/releases/mambo_v3/final-qualification.md) records
runtime/platform limits and the selected license; publication remains separate.
See the [evaluation workflow](../dev/releases/mambo_v3/evaluation.md).
