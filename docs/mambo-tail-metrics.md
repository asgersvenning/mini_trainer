# Tail-truncated release metrics

**Historical padded-scale TTA evidence.** The [deployment README](../deployment/README.md#release-comparison)
contains the current rotation-and-padding default comparison.

These metrics change the **class averaging domain**, not the evaluation images:

- Support **>5, >10 or >20** requires strictly more than that many truth instances
  **and accepted predictions**. Main tables intersect qualifying classes across all
  five pipelines; exact-boundary counts do not qualify.
- Support **>−1** is the untruncated baseline: each model's union of truth and
  accepted-prediction classes, including zero support in either domain. It does not
  intersect models, which would hide model-specific predicted-only families.
  Baseline class counts follow model-column order; undefined groups keep the
  package's own treatment.

The legacy northern-Europe preset, 52,788 reporting images, calibration and pinned
`mini_metrics` revision match the [threshold study](mambo-confidence-thresholds.md).
Per-class accuracy/precision/recall/F1 use that **complete partition**; the package's
aggregator then averages selected groups. Cross-domain mistakes still contribute
false positives/negatives. **No image rows are dropped or acceptance decisions changed.**

Classes are selected using reporting support, so this is descriptive analysis.
Domains can differ between threshold zero and calibration. Positive cutoffs hide
the [rare-family failure mode](mambo-family-precision.md); retain full-support scores.

## Calibrated thresholds

Macro-F1: full-support baseline, followed by the shared truncated class sets.

| Rank | Support > | Classes retained | V2 | V3 PyTorch | V3 ONNX | PyTorch + TTA | ONNX + TTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Species | −1 | 690 / 654 / 650 / 681 / 636 | 0.4467 | 0.5081 | 0.5100 | 0.5239 | 0.5431 |
| Species | 5 | 272 | 0.7799 | 0.8016 | 0.8001 | 0.8413 | 0.8224 |
| Species | 10 | 237 | 0.8009 | 0.8139 | 0.8124 | 0.8538 | 0.8356 |
| Species | 20 | 180 | 0.8110 | 0.8221 | 0.8202 | 0.8597 | 0.8420 |
| Genus | −1 | 379 / 398 / 395 / 372 / 372 | 0.5869 | 0.6019 | 0.6043 | 0.6655 | 0.6655 |
| Genus | 5 | 215 | 0.7875 | 0.8311 | 0.8290 | 0.8477 | 0.8477 |
| Genus | 10 | 192 | 0.8055 | 0.8364 | 0.8345 | 0.8523 | 0.8525 |
| Genus | 20 | 151 | 0.8215 | 0.8533 | 0.8514 | 0.8692 | 0.8693 |
| Family | −1 | 26 / 30 / 30 / 30 / 30 | 0.6545 | 0.5807 | 0.5816 | 0.6073 | 0.6065 |
| Family | 5 | 19 | 0.8021 | 0.8161 | 0.8174 | 0.8536 | 0.8524 |
| Family | 10 | 19 | 0.8021 | 0.8161 | 0.8174 | 0.8536 | 0.8524 |
| Family | 20 | 15 | 0.8074 | 0.8138 | 0.8154 | 0.8548 | 0.8533 |

V3 leads family F1 on these commonly represented classes, despite V2 leading full
support: the domains answer different questions. Separately calibrated TTA backends
also have different coverage; [shared-threshold results](mambo-confidence-thresholds.md#tta-backend-threshold-sensitivity)
show closely aligned predictions.

## Threshold zero

| Rank | Support > | Classes retained | V2 | V3 PyTorch | V3 ONNX | PyTorch + TTA | ONNX + TTA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Species | −1 | 1205 / 1289 / 1289 / 1177 / 1173 | 0.2620 | 0.2593 | 0.2594 | 0.3001 | 0.3011 |
| Species | 5 | 313 | 0.7815 | 0.8013 | 0.8016 | 0.8288 | 0.8287 |
| Species | 10 | 271 | 0.8062 | 0.8232 | 0.8235 | 0.8484 | 0.8483 |
| Species | 20 | 208 | 0.8209 | 0.8454 | 0.8455 | 0.8656 | 0.8656 |
| Genus | −1 | 691 / 713 / 710 / 674 / 673 | 0.3213 | 0.3230 | 0.3245 | 0.3601 | 0.3605 |
| Genus | 5 | 242 | 0.7942 | 0.8057 | 0.8055 | 0.8342 | 0.8340 |
| Genus | 10 | 218 | 0.8087 | 0.8223 | 0.8220 | 0.8520 | 0.8517 |
| Genus | 20 | 174 | 0.8425 | 0.8551 | 0.8549 | 0.8765 | 0.8764 |
| Family | −1 | 64 / 60 / 60 / 60 / 60 | 0.2697 | 0.2805 | 0.2809 | 0.2970 | 0.2971 |
| Family | 5 | 20 | 0.8238 | 0.7831 | 0.7843 | 0.8211 | 0.8212 |
| Family | 10 | 20 | 0.8238 | 0.7831 | 0.7843 | 0.8211 | 0.8212 |
| Family | 20 | 17 | 0.8635 | 0.8503 | 0.8505 | 0.8813 | 0.8814 |

## Coverage, complete results and reproduction

The [CSV](assets/mambo-tail-metrics.csv) retains macro accuracy/precision/recall/F1,
overall acceptance coverage, and truth-image/accepted-prediction counts within
retained classes. Those support counts are not acceptance coverage. Positive
cutoffs include common and per-model domains; these tables use common domains.
The baseline remains per-model. The [JSON](assets/mambo-tail-metrics.json) also
records every class ID; empty domains yield null metrics.

```sh
/path/to/pinned-metrics-env/bin/python -m dev.releases.mambo_v3.tail_report \
  --study docs/assets/mambo-threshold-comparison.json \
  --output /tmp/mambo-tail-metrics
```

[The collector](../dev/releases/mambo_v3/tail_report.py) verifies source hashes and
reporting-partition identity, uses public per-class calls and the pinned package's
`_aggregate_groups` implementation, and retains the original thresholds.
