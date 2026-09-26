# Training feature comparison plan

This page covers **unmeasured feature comparisons**, not implementation status.
The [repository roadmap](roadmap.md) sets priorities; the
[quantization roadmap](quantization-roadmap.md), [validation contract](quantized-training-validation.md)
and [measured findings](benchmarks.md) own native QT/PTQ/QAT work. Those paths are
implemented but have different numerical and resource contracts; do not repeat
their implementation plans here. EMA remains unsupported and excluded.

Select a bounded comparison only when it answers the next decision. Existing
CPU/GPU baselines do not establish benefits for the features below. New precision
formats or backends require their own operator-placement, checkpoint and
quality/resource evidence; AMP checks do not establish deeper quantization.

## Augmentation defaults

The actual training order is loader output → augmentation → model preprocessing.
The default loader supplies RGB uint8 batches. Test shape, range, dtype and class
semantics at this boundary before changing defaults.

Compare the current recipe, no augmentation, and candidate task-aware recipes.
Vertical flips and arbitrary rotations are not universal label-preserving choices
for digits; color changes can invalidate the synthetic color oracle. Biological
imagery has different invariances. Use validation data to select a candidate,
retain a named legacy recipe, and evaluate the chosen default on held-out data once.
Do not describe a replacement as better without quality evidence.

## Feature comparison matrix

| Feature | Controlled comparison | Interpretation and required controls |
| --- | --- | --- |
| MuonAuxAdamW | AdamW and SGD with momentum | Optimizer compatibility and skipped-AMP-step gating are now tested. Give each optimizer equal validation-tuning budget for learning rate and weight decay; identical hyperparameters alone are not a fair best-performance comparison. |
| `Classifier(normalized=...)` | `False` versus `True` | This switches normalization, prototype parametrization/initialization and score scaling, not one isolated operation. Preserve matching backbone initialization and report the full treatment. |
| EMLACrossEntropy | Ordinary cross entropy with identical smoothing | Use training-only class counts. Include controlled long-tail training splits; balanced synthetic counts produce zero logit adjustment and cannot demonstrate an imbalance benefit. |
| `class_weight_distribution_regularization` | Zero strength versus explicitly recorded strengths | Match loss, head and optimization settings. Report interactions with normalized prototypes and effects on rare classes. |
| `label_smoothing=None` | Explicit zero, automatic, and a fixed nonzero value | In the current flat builder, `None` resolves to `1 / num_classes`; the hierarchical builder uses the leaf-class count. Record resolved numeric values for every level. `None` does not disable smoothing. |
| Hierarchical heads | Flat `Classifier` on the identical Blair images and leaf classes | Use the same backbone and normalized setting. Report leaf and parent quality; aggregate flat probabilities through the same hierarchy for parent comparisons. Record auxiliary loss weights and parameter counts. |

Run the synthetic oracle first, adding controlled class imbalance and label noise
with independently recorded seeds where relevant. Then use MNIST and finally
Blair with the same frozen taxonomy and split manifests across variants. The
current synthetic oracle reaches ceiling accuracy: it is a correctness gate, not
a sensitive measure of every feature's benefit. Birds and iNaturalist 2021 remain
larger follow-ups after the smaller experiments are reliable.

## Measurement and reporting

Use at least three paired seeds initially. Separate fixed-hyperparameter ablations
from equally budgeted, validation-tuned comparisons. Freeze the test split and
classification mapping; record selected settings before computing final test results.
Use actual matching backbone tensors when initialization must be held constant:
a common global seed alone does not ensure identical initialization after different
constructors consume random numbers. Use separate sampler and augmentation random
generators when their sequences must remain matched across variants.

Report accuracy, macro/per-class recall, balanced accuracy, probabilistic quality
such as NLL/Brier score, and calibration alongside training/inference cost. Verify
mini_metrics APIs and score semantics before routing these metrics through that
optional integration. Publish individual runs and paired differences with uncertainty;
distinguish variation across training seeds from uncertainty due to finite test data.

Use the existing versioned report/provenance artifacts and visible repository run
summaries. Retain failures and negative/null effects. The [reporting guide](../dev/benchmarks/reporting.md) distinguishes existing
artifact retention from the optional publisher's still-unverified live activation. Do not combine different hardware,
precision, dataset versions or tuning budgets into one apparent improvement trend.
