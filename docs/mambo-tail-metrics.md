# Tail-truncated release metrics

These supplementary metrics summarize classes with **more than 0, 5, 10 or 20**
truth instances **and accepted predictions**, at each taxonomic rank. Main results
use the intersection of qualifying classes across all five pipelines, so each
pipeline is averaged over the same classes. Exact-boundary counts do not qualify.
Support >0 is the baseline for classes represented in both domains; it still
excludes predicted-only and never-predicted truth classes, unlike full-support metrics.

The dataset, legacy northern-Europe preset, 52,788-image reporting partition,
calibrated thresholds and pinned `mini_metrics` revision are unchanged from the
[threshold study](mambo-confidence-thresholds.md). Classes are selected from
reporting support; this is descriptive analysis, not independent validation.

Per-class accuracy, precision, recall and F1 are computed on the **complete reporting
partition** using `mini_metrics`. Its own aggregator then averages the retained
class groups. No image rows are dropped: mistakes from excluded truth classes into
retained predictions still contribute false positives, and mistakes from retained
truth classes into excluded predictions still contribute false negatives.

Truncation excludes predicted-only classes and rare supported classes from the
average. It therefore intentionally hides the rare-family failure mode studied
[in the family audit](mambo-family-precision.md). Keep these results alongside the
full-support metrics. The class sets may differ between threshold zero and calibrated
thresholds; comparisons across those sections are not on a fixed class domain.

## Calibrated thresholds · common classes

Macro-F1 on each shared class set:

| Rank | Support > | Classes retained | V2 | V3 PyTorch | V3 ONNX | PyTorch + TTA | ONNX + TTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Species | 0 | 413 | 0.7346 | 0.7625 | 0.7611 | 0.8021 | 0.7858 |
| Species | 5 | 272 | 0.7799 | 0.8016 | 0.8001 | 0.8413 | 0.8224 |
| Species | 10 | 237 | 0.8009 | 0.8139 | 0.8124 | 0.8538 | 0.8356 |
| Species | 20 | 180 | 0.8110 | 0.8221 | 0.8202 | 0.8597 | 0.8420 |
| Genus | 0 | 290 | 0.7589 | 0.8055 | 0.8030 | 0.8276 | 0.8276 |
| Genus | 5 | 215 | 0.7875 | 0.8311 | 0.8290 | 0.8477 | 0.8477 |
| Genus | 10 | 192 | 0.8055 | 0.8364 | 0.8345 | 0.8523 | 0.8525 |
| Genus | 20 | 151 | 0.8215 | 0.8533 | 0.8514 | 0.8692 | 0.8693 |
| Family | 0 | 22 | 0.7735 | 0.7919 | 0.7930 | 0.8281 | 0.8271 |
| Family | 5 | 19 | 0.8021 | 0.8161 | 0.8174 | 0.8536 | 0.8524 |
| Family | 10 | 19 | 0.8021 | 0.8161 | 0.8174 | 0.8536 | 0.8524 |
| Family | 20 | 15 | 0.8074 | 0.8138 | 0.8154 | 0.8548 | 0.8533 |

Within these commonly represented classes, V3 improves family Macro-F1 over V2,
and TTA improves it further. With support >5, all five pipelines are averaged over
19 families: F1 is 0.8021 for V2, 0.8161–0.8174 for ordinary V3 and 0.8524–0.8536
with TTA. This is compatible with V2 leading the **full-support** family Macro-F1;
the averaging domains answer different questions.

The TTA species backend ordering also changes in this view. Their separately
selected thresholds have different coverage; shared-threshold testing in the
[threshold study](mambo-confidence-thresholds.md#tta-backend-threshold-sensitivity)
shows closely aligned backend predictions.

## Threshold zero · common classes

| Rank | Support > | Classes retained | V2 | V3 PyTorch | V3 ONNX | PyTorch + TTA | ONNX + TTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Species | 0 | 479 | 0.6579 | 0.6922 | 0.6924 | 0.7300 | 0.7300 |
| Species | 5 | 313 | 0.7815 | 0.8013 | 0.8016 | 0.8288 | 0.8287 |
| Species | 10 | 271 | 0.8062 | 0.8232 | 0.8235 | 0.8484 | 0.8483 |
| Species | 20 | 208 | 0.8209 | 0.8454 | 0.8455 | 0.8656 | 0.8656 |
| Genus | 0 | 317 | 0.7003 | 0.7266 | 0.7267 | 0.7656 | 0.7653 |
| Genus | 5 | 242 | 0.7942 | 0.8057 | 0.8055 | 0.8342 | 0.8340 |
| Genus | 10 | 218 | 0.8087 | 0.8223 | 0.8220 | 0.8520 | 0.8517 |
| Genus | 20 | 174 | 0.8425 | 0.8551 | 0.8549 | 0.8765 | 0.8764 |
| Family | 0 | 23 | 0.7504 | 0.7317 | 0.7328 | 0.7749 | 0.7750 |
| Family | 5 | 20 | 0.8238 | 0.7831 | 0.7843 | 0.8211 | 0.8212 |
| Family | 10 | 20 | 0.8238 | 0.7831 | 0.7843 | 0.8211 | 0.8212 |
| Family | 20 | 17 | 0.8635 | 0.8503 | 0.8505 | 0.8813 | 0.8814 |

## Coverage, complete results and reproduction

Truncating the averaging domain does not change which images the pipeline accepts.
The [CSV](assets/mambo-tail-metrics.csv) retains overall acceptance coverage, the
number of truth images and accepted predictions belonging to retained classes,
and macro accuracy/precision/recall/F1. These support counts are not interchangeable
with acceptance coverage. Both common-class and per-model class sets are included;
use the common sets for direct model comparisons. The [JSON](assets/mambo-tail-metrics.json)
additionally records every retained class ID. Empty class domains produce null metrics.

```sh
/path/to/pinned-metrics-env/bin/python -m dev.releases.mambo_v3.tail_report \
  --study docs/assets/mambo-threshold-comparison.json \
  --output /tmp/mambo-tail-metrics
```

[The collector](../dev/releases/mambo_v3/tail_report.py) verifies source hashes and
reporting-partition identity, uses public per-class calls and the pinned package's
`_aggregate_groups` implementation, and retains the original thresholds. No deployment
default or existing headline metric is changed.
