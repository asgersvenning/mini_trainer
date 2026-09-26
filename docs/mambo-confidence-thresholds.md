# Confidence thresholds and the MAMBO comparison

**Historical padded-scale TTA evidence.** The [deployment README](../deployment/README.md#release-comparison)
contains the current rotation-and-padding default comparison.

Thresholding substantially changes the quality comparison, at the cost of rejecting
images. These results use **legacy northern Europe** for V2, V3 PyTorch/ONNX and
both V3 backends with padded-scale TTA. Deployment defaults remain threshold zero.

## Method and interpretation

Metrics, splitting and calibration use `mini_metrics` revision
`70cc69adc05362863439277048e06386c1f885e1`:

- `MetricDF.split((0.9, 0.1), strata=("label",), seed=42)` groups ranks by image ID,
  yielding **5,852 calibration / 52,788 reporting images**. Identity hashes verify
  matching truth/partitions across pipelines. Both operating points below use the
  reporting partition, including unknown truth—not the full 58,640-image population.
- `OptimalConfidenceThreshold(crit=MacroF1, eps=0.01, use_quantiles=True,
  n_bootstraps=0)` fits each pipeline/rank on calibration only. Its exact F1 curve
  and connected near-optimal plateau selector need not select the exact maximum.
  Results match `evaluate_file(optimal=True, seed=42)`; full-data threshold-zero
  checks separately reproduce the earlier comparison.
- Coverage is the fraction accepted at each rank (`confidence >= threshold`),
  distinct from vocabulary coverage. Independent thresholds and
  `hierarchical=False` do not implement species-to-family fallback.

Macro accuracy averages accepted-prediction accuracy within truth taxa, excluding
those with no accepted predictions; micro accuracy averages all accepted images.
Recall counts rejection as misses. Macro-F1 penalizes rejection and includes
truth-supported and predicted-only taxa; precision averages predicted-class groups.
Interpret accuracy/precision together with recall and coverage.

This single image-level split is not site/observation-level or stability validation.
Prior TTA selection used Flemming, so it is not independent model/TTA validation
either. Thresholds are specific to these pipelines, scores and preset; deployment
still defaults to zero.

## Effect on the model comparison

Calibration raises Macro-F1 at every rank while lowering coverage and recall.
V3 overtakes V2 at species level; TTA further improves species/genus F1. At family
level the ranking reverses: V2 leads calibrated F1, while V3 + TTA retains more
recall. The [family audit](mambo-family-precision.md) explains how rare predicted-only
groups cause that reversal. Compare coverage alongside scores.

![Macro-F1 and coverage before and after calibration](assets/mambo-threshold-comparison.svg)

Coverage at threshold zero is 100%; all other columns except the before/after F1 pair use calibrated thresholds.

### Species

| Pipeline | Threshold | Macro-F1 zero → calibrated | Coverage | Macro accuracy | Micro accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAMBO v2 | 0.7484 | 0.2620 → 0.4467 | 69.73% | 84.65% | 83.40% | 0.6381 | 0.5418 |
| V3 PyTorch | 0.8100 | 0.2593 → 0.5081 | 70.81% | 86.69% | 83.38% | 0.7028 | 0.5775 |
| V3 ONNX | 0.8162 | 0.2594 → 0.5100 | 70.45% | 86.95% | 83.45% | 0.7089 | 0.5753 |
| V3 PyTorch + TTA | 0.7393 | 0.3001 → 0.5239 | 78.32% | 86.59% | 82.57% | 0.6695 | 0.6393 |
| V3 ONNX + TTA | 0.8280 | 0.3011 → 0.5431 | 74.05% | 87.55% | 83.56% | 0.7363 | 0.6076 |

### Genus

| Pipeline | Threshold | Macro-F1 zero → calibrated | Coverage | Macro accuracy | Micro accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAMBO v2 | 0.8760 | 0.3213 → 0.5869 | 69.81% | 95.34% | 92.58% | 0.7848 | 0.6177 |
| V3 PyTorch | 0.8211 | 0.3230 → 0.6019 | 75.75% | 95.48% | 90.77% | 0.7454 | 0.6784 |
| V3 ONNX | 0.8302 | 0.3245 → 0.6043 | 75.37% | 95.52% | 90.86% | 0.7523 | 0.6746 |
| V3 PyTorch + TTA | 0.8699 | 0.3601 → 0.6655 | 77.73% | 96.28% | 91.16% | 0.8167 | 0.7031 |
| V3 ONNX + TTA | 0.8709 | 0.3605 → 0.6655 | 77.69% | 96.30% | 91.18% | 0.8169 | 0.7032 |

### Family

| Pipeline | Threshold | Macro-F1 zero → calibrated | Coverage | Macro accuracy | Micro accuracy | Macro precision | Macro recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAMBO v2 | 0.9553 | 0.2697 → 0.6545 | 77.44% | 99.64% | 99.86% | 0.8413 | 0.6467 |
| V3 PyTorch | 0.9708 | 0.2805 → 0.5807 | 73.06% | 99.33% | 99.80% | 0.7406 | 0.6535 |
| V3 ONNX | 0.9699 | 0.2809 → 0.5816 | 73.26% | 99.33% | 99.80% | 0.7406 | 0.6549 |
| V3 PyTorch + TTA | 0.9648 | 0.2970 → 0.6073 | 78.74% | 99.56% | 99.82% | 0.7394 | 0.7002 |
| V3 ONNX + TTA | 0.9665 | 0.2971 → 0.6065 | 78.47% | 99.56% | 99.83% | 0.7395 | 0.6987 |

## TTA backend threshold sensitivity

The selected species threshold differs substantially between PyTorch (0.7393)
and ONNX (0.8280). Applying **each threshold to both backends** isolates the effect:

| Shared species threshold | PyTorch TTA Macro-F1 / coverage | ONNX TTA Macro-F1 / coverage |
|---|---:|---:|
| 0.7393 | 0.5239 / 78.32% | 0.5238 / 78.30% |
| 0.8280 | 0.5433 / 74.06% | 0.5431 / 74.05% |

Both thresholds are within 0.01 of each backend's calibration maximum F1
(about 0.6347). Shared-threshold results nearly coincide: the apparent ONNX
advantage chiefly reflects near-optimal threshold selection. Keep the selected
values rather than choosing a new winner on reporting data; validate threshold
stability before adopting defaults.

## Precision–recall and accuracy–coverage curves

![Five-pipeline precision–recall and accuracy–coverage curves](assets/mambo-threshold-curves.svg)

Left: macro precision/recall; right: accepted-image micro accuracy/coverage.
Dashed lines are ONNX; circles mark threshold zero, stars the calibrated points.
Similar coverage helps distinguish discrimination from stronger rejection;
calibration optimizes Macro-F1, not a shared coverage target.

These are **top-prediction rejection curves**, not one-vs-rest PR curves or AP/AUC
estimates. Each point uses `evaluate_file` on reporting data. The grid combines
51 uniform thresholds, 21 calibration-confidence quantiles per rank and selected
thresholds, deduplicated. Lines follow threshold order without smoothing or a
monotonic envelope; changing class domains can make macro precision irregular.
Calibration uses the exact F1 curve, not this plotting grid. Undefined results stay null.

The [tail comparison](mambo-tail-metrics.md) supplements full-support results with
common-class support >5/10/20 averages. Excluding rare and predicted-only families
changes the question; retain the untruncated comparison.

## Evidence and reproduction

The [CSV](assets/mambo-threshold-metrics.csv) retains all/known-truth reporting
scores, calibration scores, thresholds, Theil U and coverage. Known-only results
reuse all-truth calibration. The [JSON](assets/mambo-threshold-comparison.json)
retains curve points, source/partition hashes and full-data checks.

Collect from the retained prediction directories in
[`SOURCES`](../dev/releases/mambo_v3/threshold_report.py), or regenerate charts
directly from committed evidence without predictions or inference:

```sh
/path/to/pinned-metrics-env/bin/python -m dev.releases.mambo_v3.threshold_report \
  --evidence local-evidence --output local-evidence/mambo-threshold-study
.venv/bin/python -m dev.releases.mambo_v3.threshold_report \
  --data docs/assets/mambo-threshold-comparison.json \
  --output /tmp/mambo-threshold-charts
```

Before adoption, validate coverage/recall in the target workflow and define abstention
or fallback. The deployment CLI accepts one scalar threshold, not these three
independently calibrated rank thresholds.
