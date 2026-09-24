# Why calibrated family Macro-F1 favours V2

**The difference is driven by additional predicted-only families surviving V3's
confidence thresholds.** Their image counts are small, but each family receives
equal weight in macro precision and Macro-F1. It is not a larger V3 taxonomy:
V2 and V3 have identical genus/family mappings for all 12,632 model species, and
the shared northern-Europe species list spans 67 eligible families in both.
Flemming reporting truth contains 23 families.

This audit uses the same 52,788 reporting images, all truth, and independently
calibrated family thresholds as the [threshold study](mambo-confidence-thresholds.md).
All precision, recall and F1 values and class weights come from pinned
`mini_metrics` revision `70cc69adc05362863439277048e06386c1f885e1`.
Counts below describe retained predictions, without reimplementing metrics.

## Which families enter the average?

At the calibrated thresholds, every pipeline predicts the **same 22 truth-present
families**. Gelechiidae is present in truth but has no accepted predictions in any
pipeline. In addition, V2 predicts 3 families absent from reporting truth; V3 predicts
7. Those extra families have zero precision and zero F1.

| Pipeline | Predicted-only families | Images assigned to them | Official macro precision | Precision over truth-present groups only |
|---|---:|---:|---:|---:|
| MAMBO v2 | 3 | 4 | 0.8413 | 0.9561 |
| V3 PyTorch | 7 | 31 | 0.7406 | 0.9762 |
| V3 ONNX | 7 | 31 | 0.7406 | 0.9762 |
| V3 PyTorch + TTA | 7 | 24 | 0.7394 | 0.9747 |
| V3 ONNX + TTA | 7 | 23 | 0.7395 | 0.9748 |

The last column is a **diagnostic change to the averaging domain**, not a corrected
benchmark score. It reaggregates the package's existing per-class outputs over
truth-present families; it does not delete images, rerun predictions, or replace
the official metrics. At these operating points, all pipelines have the same 22
active precision groups in that diagnostic.

The package's weights give the exact decomposition:

- V2: `0.956078 × 22 / (22 + 3) = 0.841349` macro precision.
- V3 PyTorch: `0.976235 × 22 / (22 + 7) = 0.740592` macro precision.

V3 has higher average precision within those shared truth-present groups. Its
lower official macro precision arises from the four additional zero-precision
groups. Even one accepted prediction can activate such a group; these are errors
relative to this dataset's labels, not evidence that the family cannot occur locally.

## Which extra families survive?

Accepted predictions into families absent from reporting truth:

| Family | V2 | V3 PyTorch | V3 ONNX | PyTorch + TTA | ONNX + TTA |
|---|---:|---:|---:|---:|---:|
| Bedelliidae | 0 | 21 | 21 | 17 | 16 |
| Choreutidae | 0 | 2 | 2 | 2 | 2 |
| Coleophoridae | 0 | 2 | 2 | 1 | 1 |
| Cossidae | 0 | 1 | 1 | 1 | 1 |
| Opostegidae | 0 | 2 | 2 | 1 | 1 |
| Pieridae | 2 | 0 | 0 | 0 | 0 |
| Psychidae | 0 | 1 | 1 | 1 | 1 |
| Thyrididae | 1 | 0 | 0 | 0 | 0 |
| Tineidae | 1 | 2 | 2 | 1 | 1 |

Bedelliidae contributes most V3 cases: without TTA, 15 are labelled Erebidae and
6 Geometridae. With PyTorch TTA, those counts fall to 13 and 4; ONNX TTA has 12 and
4. Each of the other six predicted-only V3 families has only one or two accepted
images. V2's four cases are two Nolidae → Pieridae, one Erebidae → Tineidae and
one Geometridae → Thyrididae. These are label-based confusions; image-level expert
review has not established whether every ground-truth annotation is correct.

## How this affects F1 and the crossing

Macro-F1 is an average of per-family F1 values, **not** the harmonic mean of the
reported macro precision and macro recall. Its active domain includes all 23
truth families, even the one without accepted predictions, plus predicted-only
families: 26 groups for V2 and 30 for V3.

| Pipeline | Official family Macro-F1 | F1 over the same 23 truth families only | Macro recall |
|---|---:|---:|---:|
| MAMBO v2 | 0.6545 | 0.7399 | 0.6467 |
| V3 PyTorch | 0.5807 | 0.7575 | 0.6535 |
| V3 ONNX | 0.5816 | 0.7586 | 0.6549 |
| V3 PyTorch + TTA | 0.6073 | 0.7921 | 0.7002 |
| V3 ONNX + TTA | 0.6065 | 0.7911 | 0.6987 |

Again, the middle column is explanatory, not a substitute benchmark. Its ranking
favours V3, especially TTA; adding the zero-F1 predicted-only groups yields the
official ranking favouring V2. Recall is unchanged by this diagnostic because
predicted-only families have no true support.

At threshold zero, the pattern is different: **V2 predicts 41 absent families,
V3 37**, with 408 versus 756 accepted images assigned to them (535 for either TTA
backend). Thresholding removes more of V2's low-confidence predicted-only groups,
leaving 3 versus 7. This explains why the ordering changes after calibration.
A common family threshold of **0.96** still leaves 3 such groups for V2, 8 for
ordinary V3 and 7 for TTA; the effect is not solely the choice of different
optimized threshold values. Within-truth averages also vary with operating point.

At the calibrated operating points, V3 improves recall of represented families
while retaining a wider set of rare, confident false-family predictions.
The official macro metrics expose that weakness, with considerable sensitivity
to singleton predictions. Keep both official metrics and coverage; excluding
absent families from deployment or evaluation based on Flemming would hide the
failure mode and artificially tailor the system to this benchmark.

## Evidence and reproduction

The [per-family CSV](assets/mambo-family-precision.csv) contains family names,
truth counts, accepted prediction counts, and the package's P/R/F1 values and
weights for threshold zero, calibrated thresholds and the common 0.96 threshold.
A blank metric with zero weight means the package did not emit that class group.
The [JSON evidence](assets/mambo-family-precision.json) retains official scores,
diagnostic aggregations, false-positive confusion counts, input hashes, and the
name/taxonomy audit. Family names come from the pinned local metadata parquet.

The [audit script](../dev/releases/mambo_v3/family_precision_report.py) uses public
per-class metric calls (`aggregate=False`) and the pinned package's own
`_aggregate_groups` implementation for diagnostic reaggregation. No data rows or
thresholds are chosen using the diagnostic to improve the official results.

Prepare names and verify taxonomy using `.venv` (PyArrow/PyTorch), then collect
metrics using the existing pinned metrics environment:

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
