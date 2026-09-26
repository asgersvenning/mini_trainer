# Reusable MAMBO release evidence

This release establishes a retained baseline, not an immutable evaluation policy.
Future improvements must receive an evaluation revision and a short explanation in
that release's README/changelog. Never relabel old measurements as newly collected.

## What each comparison must identify

Record the trained checkpoint/graph hashes; adapter version/source commit; complete
bundle hash; ordered vocabulary and selected-list hash; preprocessing, score and
TTA recipe; effective precision/backend/provider; embeddings on/off; dataset and
ordered sample identities; reporting/calibration split identities; metric package
revision and options; rank and all/known-truth policy; support threshold and class
intersection rule. For timing also record hardware, runtime/dependency versions,
thread/worker counts, batch, image bank, warmup, repetition method, measurement
boundary and memory method. Keep failures and unqualified configurations visible.

## Retention and reuse

Retain per-image truth, prediction, confidence and sample identity at all ranks,
per-class metrics/support, calibrated thresholds, aggregate metrics, raw timing
observations and environment/provenance records. Keep restricted data and original
photographs out of public release assets. Hashes identify retained inputs; they do
not imply that another developer has permission or access to those inputs.

Existing prediction/confidence files permit many metric, calibration and support
policy changes without running the old model again. A different class-list mask,
TTA recipe or preprocessing cannot generally be reconstructed from top-1 outputs;
those changes need retained richer scores or another inference run. Do not promise
that all future comparisons are possible from the current compact predictions.

Maintain a fixed historical comparison alongside any improved policy, or recompute
old and new metrics from retained predictions under one new revision. Do not pool
incompatible reporting populations, class-support averages or timing protocols.
On a new hardware environment, one reference-model measurement can anchor a new
comparison without requiring every previous model to run again. Historical speed
numbers remain explicitly tied to their original environment.

## Current baseline

- `docs/assets/mambo-promoted-tail.json` and associated quality/threshold evidence:
  Flemming, legacy northern Europe, reporting/calibration separation, both confidence
  settings, all ranks and full/support >5 macro metrics.
- `docs/assets/mambo-indomain-support.json` and threshold/tail artifacts:
  original global-lepi test population, global vocabulary, analogous metric policy.
- `docs/assets/mambo-promoted-speed.json`: laptop timing and environment evidence.
- `docs/assets/mambo-indomain-speed.csv` and campaign metadata: retained EPYC and
  earlier B200 request comparisons, including V2.
- `docs/assets/mambo-hpc-current-speed.csv` and provenance: latest B200 request and
  streaming observations. These replace only corresponding measured points.

The public evidence pages link complete tables and reproducible report commands.
The publication preparation manifest inventories the linked asset bytes; restricted
per-image inputs remain in the retained local/UCloud archives and are not bundled.
