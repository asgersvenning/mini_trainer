# TTA recipe selection

When TTA is enabled, use `rotation30_pad25_3`: original plus ±30° rotations,
each with 25% edge padding. It improved the Flemming comparison on both PyTorch
and ONNX at the same three-view cost as the previous `padded_scale` recipe.
TTA remains off by default; explicit `padded_scale` preserves the previous option.
See [the API and spatial policy](mambo-tta.md) and
[current release comparisons](../deployment/README.md#release-comparison).

## Evidence for the choice

Native PyTorch, legacy northern Europe, all truth, full-support macro-F1:

| Operating point | Recipe | Species | Genus | Family |
|---|---|---:|---:|---:|
| No threshold | Previous padded scale | 0.3001 | 0.3601 | 0.2970 |
| No threshold | Selected three views | 0.3201 | 0.3873 | 0.3592 |
| Approximately 80% coverage | Previous padded scale | 0.5140 | 0.6404 | 0.6105 |
| Approximately 80% coverage | Selected three views | 0.5729 | 0.6990 | 0.7560 |

The five-view mixed-padding candidate added modest species/genus gains at
80% coverage (0.5770/0.7126), but lower family F1 (0.6969) and two extra passes.
It remains an explicit alternative, not the default. Improvements were not
universal across metrics: recipe-specific calibration increased coverage/F1
while reducing family macro accuracy relative to the old recipe.

Metrics use pinned `mini_metrics`, 52,788 reporting images and 5,852 separate
calibration images. Recipe exploration used some reporting images, so this is
not independent validation. These out-of-domain results should be read alongside
the in-domain comparison; neither establishes a universal TTA benefit.

Matched-coverage thresholds use reporting confidences without labels and retain
ties. They are diagnostic operating points, distinct from thresholds optimized
on the calibration split. Support >5 results in this study intersect eleven
pipelines and therefore differ from the current five-pipeline comparison.
No rows are removed when truncating the class macro average.

## Provenance

The [complete evidence](assets/mambo-composed-tta.json) retains metrics, thresholds
and source identities and remains an input to
[release promotion reporting](../dev/releases/mambo_v3/promoted_report.py).
Collection timings are not speed benchmarks: the native and ONNX jobs overlapped.

For replay, use [composed_full.py](../dev/releases/mambo_v3/composed_full.py),
[composed_metrics.py](../dev/releases/mambo_v3/composed_metrics.py) and
[composed_report.py](../dev/releases/mambo_v3/composed_report.py).
The [historical report](https://github.com/asgersvenning/mini_trainer/blob/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/docs/mambo-composed-tta.md)
records exact commands and the native run's older nested precision-metadata caveat;
its top-level `effective_precision=fp16` describes actual execution.
