# Rare-family sensitivity in the historical comparison

**Historical padded-scale TTA evidence.** See the [deployment README](../deployment/README.md#release-comparison)
for the current rotation-and-padding comparison.

V3's lower calibrated family macro precision and F1 came from **more predicted-only
families surviving rejection**, despite better recall. V2/V3 have identical parent
mappings for all 12,632 species; their shared northern-Europe list spans 67 families.
The 52,788-image Flemming reporting partition contains 23 truth families.

## Effect on the averages

The [threshold study](mambo-confidence-thresholds.md) defines the shared all-truth
reporting partition, separate per-pipeline calibration and pinned `mini_metrics`.
At the calibrated thresholds, all five pipelines predict the same 22 truth-present
families; Gelechiidae has truth support but no accepted predictions. Each additional
predicted-only family contributes a zero-precision, zero-F1 group—even a singleton.

Precision and F1 cells show **official / truth-group diagnostic** scores. The diagnostic uses
`mini_metrics` to reaggregate its per-class outputs over truth-present families;
it changes the averaging domain, not predictions, thresholds or image rows.
It is explanatory, not a replacement benchmark.

| Pipeline | Predicted-only families / images | Macro precision | Macro-F1 | Macro recall | Coverage |
|---|---:|---:|---:|---:|---:|
| MAMBO v2 | 3 / 4 | 0.8413 / 0.9561 | 0.6545 / 0.7399 | 0.6467 | 77.44% |
| V3 PyTorch | 7 / 31 | 0.7406 / 0.9762 | 0.5807 / 0.7575 | 0.6535 | 73.06% |
| V3 ONNX | 7 / 31 | 0.7406 / 0.9762 | 0.5816 / 0.7586 | 0.6549 | 73.26% |
| V3 PyTorch + TTA | 7 / 24 | 0.7394 / 0.9747 | 0.6073 / 0.7921 | 0.7002 | 78.74% |
| V3 ONNX + TTA | 7 / 23 | 0.7395 / 0.9748 | 0.6065 / 0.7911 | 0.6987 | 78.47% |

The package weights give the precision decomposition:

- V2: `0.956078 × 22 / (22 + 3) = 0.841349`.
- V3 PyTorch: `0.976235 × 22 / (22 + 7) = 0.740592`.

Macro-F1 averages per-family F1, not the harmonic mean of macro precision/recall.
Its domain includes all 23 truth families plus predicted-only families: 26 groups
for V2, 30 for V3. Recall is unaffected by removing predicted-only groups because
they have no truth support.

Bedelliidae accounts for 21 of V3's 31 accepted predicted-only cases (15 labelled
Erebidae, six Geometridae); the other six families have one or two cases each.
These are label-based errors, not expert-confirmed misidentifications or evidence
that a family cannot occur locally. The [per-family CSV](assets/mambo-family-precision.csv)
retains all five pipelines' counts and metric weights.

## Why calibration changes the ranking

At threshold zero, V2 predicts **41 absent families / 408 images**, versus
**37 / 756** for ordinary V3 (535 images for either TTA backend). Calibration
removes more of V2's low-confidence groups, leaving 3 versus 7. A shared family
threshold of **0.96** still leaves 3 groups for V2, 8 for ordinary V3 and 7 for TTA:
separate optimized thresholds are not the sole cause.

Keep full-support metrics, recall and coverage alongside [tail-truncated results](mambo-tail-metrics.md).
Truncation intentionally hides this rare-family failure mode. Removing absent
families from deployment based on Flemming would tailor the vocabulary to the test.

## Evidence and reproduction

The [JSON](assets/mambo-family-precision.json) retains official scores, diagnostic
aggregations, confusion counts, source hashes and the verified name/taxonomy audit.
The CSV covers threshold zero, calibrated thresholds and 0.96. A blank metric with
zero weight means the package did not emit that group.

The [audit script](../dev/releases/mambo_v3/family_precision_report.py) uses public
per-class calls (`aggregate=False`) and the pinned package's `_aggregate_groups`.
Prepare names/taxonomy with PyArrow/PyTorch, then use the existing pinned metrics
environment; no inference is needed:

```python
from pathlib import Path
from dev.releases.mambo_v3.family_precision_report import prepare_taxonomy
from dev.releases.mambo_v3.evaluation_data import write_json

write_json(Path("/tmp/family-taxonomy.json"), prepare_taxonomy(
    Path("examples/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"),
    Path("local-evidence/mambo-v3/MAMBO/hierarchical_bioclip2_ft_neu_v1.pt"),
    Path("local-evidence/mambo-bundle-final/classes.json"),
))
```

```sh
/path/to/pinned-metrics-env/bin/python -m dev.releases.mambo_v3.family_precision_report \
  --taxonomy /tmp/family-taxonomy.json --output /tmp/family-precision
```
