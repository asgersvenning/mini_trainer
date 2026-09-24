# Compact padding-and-rotation TTA qualification

The completed [full composed-TTA comparison](mambo-composed-tta.md) evaluates the
three shortlisted recipes on both backends, with calibration and matched coverage.

**Composing stronger padding into existing rotation views improves the preliminary
five-view recipe without adding passes.** The leading five-view candidate uses
original, ±10° with 15% padding, and ±30° with 25% padding. Three-view candidates
remain useful cost/quality alternatives. This is exploratory qualification, not
a release-default change or a replacement for full-data evaluation.

Each transformed view rotates the decoded image once with bilinear interpolation,
expands the canvas to retain the image extent, fills rotation corners with RGB
(124,116,104), then edge-pads **each side** by the specified fraction of its axis.
Ordinary deployment preprocessing follows. Padding produces a wider framing;
it is combined with rotation in each view, rather than added as separate model
passes. There is no additional crop transform. Mean FP32 leaf logits feed the
existing regional filter and hierarchy. The earlier `wide_rotation_5` already
pads every rotated view by 8%; its five views are original and ±10°/±30°.

## Evidence population and controls

Native CUDA automatic precision (FP16 backbone, FP32 head), legacy northern Europe,
batch 32, four preparation/runtime threads. Thirteen cached individual views form
ten initial recipes through the existing public callable TTA interface. Two
five-view compositions were then added using the same cached views; these are
explicitly post-hoc candidates, not independent confirmation.

- The same 38 flagged family-error images and 128 controls correct/accepted at
  family level in both V2 and ordinary V3.
- A disjoint uniform random sample of 1,024 reporting images, seed 20260925,
  selected without reference to prediction correctness. It is not an independent
  validation set: it comes from the already-studied Flemming reporting partition.
- All predictive metrics use pinned `mini_metrics`; both threshold zero and the
  **same ordinary-V3 thresholds** are reported. No recipe-specific optimization.
  Thresholds at species/genus/family are 0.8099595 / 0.8210953 / 0.9708009.
- Raw cached aggregation agrees with the ordinary TTA API at atol=1e-6 for identical
  batch shape. Regression checks show the composed 8% transform exactly reproduces
  the earlier rotation transform.

## Random sample: no confidence threshold

All truth, no class truncation. These small-sample macro metrics are not comparable
to the headline full-data or 52,788-image tables.

| Recipe | Views | Species macro accuracy | Macro-F1: species / genus / family |
|---|---:|---:|---:|
| `none` | 1 | 72.88% | 0.4849 / 0.5679 / 0.5016 |
| `padded_scale` | 3 | 74.55% | 0.5195 / 0.6213 / 0.5717 |
| `wide_rotation_5` | 5 | 74.54% | 0.5410 / 0.6491 / 0.6261 |
| `wide_rotation_pad15_5` | 5 | 74.73% | 0.5377 / 0.6503 / 0.6120 |
| `wide_rotation_mixed_padding_5` | 5 | 76.28% | 0.5546 / 0.6426 / 0.6307 |
| `rotation30_3` | 3 | 74.53% | 0.5388 / 0.6454 / 0.6658 |
| `rotation30_pad15_3` | 3 | 74.69% | 0.5358 / 0.6492 / 0.6360 |
| `rotation30_pad25_3` | 3 | 76.30% | 0.5523 / 0.6423 / 0.6521 |
| `mixed_3` | 3 | 74.84% | 0.5435 / 0.6436 / 0.6272 |
| `mixed_mirrored_3` | 3 | 74.92% | 0.5419 / 0.6467 / 0.6236 |

## Random sample: fixed confidence thresholds

Each cell is **macro-F1 / coverage**. These thresholds were calibrated for ordinary
V3, so the candidates will need their own calibration before selecting deployment
operating points.

| Recipe | Species | Genus | Family |
|---|---:|---:|---:|
| `none` | 0.5979 / 71.48% | 0.6982 / 75.10% | 0.8095 / 72.36% |
| `padded_scale` | 0.6044 / 74.71% | 0.7153 / 78.61% | 0.8334 / 77.93% |
| `wide_rotation_5` | 0.6398 / 79.20% | 0.7450 / 82.91% | 0.8452 / 82.42% |
| `wide_rotation_pad15_5` | 0.6500 / 79.88% | 0.7580 / 83.79% | 0.8627 / 83.69% |
| `wide_rotation_mixed_padding_5` | 0.6570 / 79.88% | 0.7631 / 83.89% | 0.8637 / 84.08% |
| `rotation30_3` | 0.6392 / 78.91% | 0.7419 / 82.91% | 0.8522 / 82.52% |
| `rotation30_pad15_3` | 0.6580 / 79.00% | 0.7671 / 83.11% | 0.8596 / 83.01% |
| `rotation30_pad25_3` | 0.6515 / 79.49% | 0.7688 / 83.11% | 0.8643 / 83.50% |
| `mixed_3` | 0.6288 / 78.52% | 0.7397 / 81.93% | 0.8513 / 81.84% |
| `mixed_mirrored_3` | 0.6425 / 78.81% | 0.7538 / 82.81% | 0.8409 / 81.54% |

The 25% recipe has higher species macro accuracy and F1 without thresholding than
the default and five-view recipe. At fixed thresholds, 15% has slightly higher
species F1; 25% has slightly higher genus/family F1 and coverage. Neither dominates
all metrics. Three-view ±30° with 8% padding has the highest unthresholded family
F1 among these candidates, further showing that there is no single universal winner.

## Targeted errors and possible harm

Family-level diagnostic counts at the same fixed threshold:

| Recipe | Correct flagged cases /38 | Accepted wrong flagged cases /38 | Previously accepted errors now rejected | Correct/accepted controls becoming rejected /128 |
|---|---:|---:|---:|---:|
| `none` | 2 | 31 | 0 | 0 |
| `padded_scale` | 2 | 21 | 11 | 0 |
| `wide_rotation_5` | 4 | 13 | 19 | 0 |
| `wide_rotation_pad15_5` | 4 | 9 | 22 | 0 |
| `wide_rotation_mixed_padding_5` | 5 | 8 | 24 | 0 |
| `rotation30_3` | 4 | 9 | 22 | 0 |
| `rotation30_pad15_3` | 4 | 8 | 23 | 0 |
| `rotation30_pad25_3` | 5 | 4 | 27 | 0 |
| `mixed_3` | 3 | 12 | 19 | 1 |
| `mixed_mirrored_3` | 3 | 13 | 18 | 0 |

With the three-view 25% recipe, only three initially wrong family predictions become correct;
27 previously accepted errors become rejected. Confidence suppression is the main
benefit on the flagged cases. All 128 selected controls remain family-correct;
the 15% and 25% recipes also keep them all accepted. Some flagged images are visually
similar, and 16 have truth species absent from the model vocabulary, so these 38
cases are not independent or representative.

The random sample exposes trade-offs that the easy family controls cannot:
the three-view 25% recipe has 127 accepted species errors versus 118 with the default, while
species coverage rises from 74.71% to 79.49%. Accepted family errors rise from two
to three. Relative to single-view inference, it corrects 51 species predictions
but breaks 12; the default corrects 25 and breaks three. Better aggregate metrics
therefore do not mean every image improves or every wrong confidence decreases.

## Five-view composition: equal-budget comparison

At the same five-view budget, replacing 8% padding with 15% on the ±10° views and
25% on the ±30° views improves fixed-threshold macro-F1 at every rank:
0.6398 → 0.6570 species, 0.7450 → 0.7631 genus, 0.8452 → 0.8637 family.
Coverage also increases (79.20% → 79.88%, 82.91% → 83.89%, 82.42% → 84.08%).
Flagged accepted family errors fall from 13 to eight, while all 128 family controls
stay correct and accepted. On the random sample, correct family predictions rise
from 974 to 979; accepted family errors remain three.

This is direct support for composing transformations rather than adding standalone
views. The mixed-padding five-view candidate has better genus accuracy and family
micro accuracy than the three-view 25% candidate, while the latter rejects more
flagged errors and costs fewer passes. Both deserve broader qualification. There
is no blanket preference for three views or prohibition on additional views when
they supply useful diversity.

## Inference cost

Exploratory warm native-GPU timings on the same 32-image bank, three interleaved
rounds of three observations per recipe, after one warmup per recipe. Includes
image decoding through completed CPU results. These single-process timings are
not the fresh-process release benchmark and have no CPU or new ONNX counterpart.

| Recipe | Views | Images/s |
|---|---:|---:|
| `none` | 1 | 146.7 |
| `padded_scale` | 3 | 53.3 |
| `wide_rotation_5` | 5 | 31.0 |
| `rotation30_3` | 3 | 51.4 |
| `rotation30_pad15_3` | 3 | 51.5 |
| `rotation30_pad25_3` | 3 | 50.7 |
| `mixed_3` | 3 | 49.6 |

The three-view 25% recipe is about 1.64× as fast as five-view wide rotation, and about 5%
slower than the current three-view default in this warm diagnostic. Composition
therefore preserves the three-pass budget with only modest transform overhead.
The two added five-view compositions have not yet been timed.

## Next qualification and reproduction

Advance mixed-padding five-view and the 15%/25% three-view recipes to full-data native/ONNX evaluation,
with separate calibration/reporting, full and support >5 metrics, and coverage.
Then compare at matched coverage as well as each recipe’s optimized thresholds,
and benchmark CPU/GPU end-to-end cost before changing the enabled-TTA default.
This study does not establish in-domain behavior or independent generalization.

The [compact evidence](assets/mambo-compact-tta.json) includes all twelve recipes,
all ranks, precision/recall, both confidence settings, sample IDs and transitions.
Raw view logits, canonical predictions and input records remain in ignored local evidence.
The [inference collector](../dev/releases/mambo_v3/compact_tta.py) and
[mini_metrics collector](../dev/releases/mambo_v3/compact_tta_metrics.py) reproduce it:

```sh
CUDA_VISIBLE_DEVICES=0 /path/to/gpu-env/bin/python -m dev.releases.mambo_v3.compact_tta \
  --bundle /path/to/bundle --manifest /path/to/flemming-manifest.json \
  --root /path/to/flemming --samples /path/to/original-error-study/samples.json \
  --reporting-ids /path/to/reporting-ids.json --output /path/to/fresh-output
/path/to/pinned-metrics-env/bin/python -m dev.releases.mambo_v3.compact_tta_metrics \
  --root /path/to/fresh-output
```
