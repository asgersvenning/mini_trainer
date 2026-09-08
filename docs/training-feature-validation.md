# Training features: implementation and comparison plan

This is planned work. Existing CPU/GPU benchmark results establish a baseline;
they do not measure the benefit of the features below. EMA is temporarily
nonfunctional and excluded from these experiments; repair is deferred.

## Actual quantized training is the primary implementation target

The priority is now QT that lowers training memory and increases speed, together
with faster data loading for floating-point and quantized workloads. PTQ/QAT do
not satisfy that objective. See the [QT and loader probes](../dev/benchmarks/README.md#quantized-training-and-loader-performance);
optimizer integration, checkpoint/resume, convergence and end-to-end measurement
remain requirements, not optional follow-ups.

The initial [INT8 PTQ/QAT Python backend](quantization.md) is implemented on the
`quant` branch. Its CPU tests establish a training-to-integer-inference path;
user-facing checkpoint integration, other backends and quality studies remain open.

Deliver two distinct paths through the existing builders, checkpoint and export
interfaces, with optional dependencies:

- Quantization-aware training: specify simulated weight/activation bit widths,
  scales and observer behavior, then convert and evaluate the resulting inference
  artifact. Fake quantization during training is not itself proof of faster or
  smaller inference.
- Post-training quantization: compare calibrated integer/low-bit artifacts against
  the same floating-point checkpoint. Calibration uses a recorded subset of
  training data, never held-out test images.

Start with an explicit, validated precision/backend combination before expanding
to lower bit widths. Record weight-only versus weight-and-activation quantization,
which operators remain floating point, model size, peak memory, latency/throughput,
and per-class/per-level quality. Validate normalized/parametrized heads, functional
linear operations, masks and hierarchical outputs instead of limiting export to
one backbone. Verify save/reload, optimizer/scheduler/AMP state during training,
and quantized artifact loading in the intended standalone runtime.

Hardware-specific reduced-precision compute, including FP8 where supported, is a
separate profile. Existing float16/bfloat16 autocast tests do not establish deeper
quantization. Declare unsupported backends and models; prohibit silent fallback
that mislabels ordinary floating-point execution as quantized.

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
summaries. Retain failures and negative/null effects. Add durable historical hosting
before the current artifact retention expires. Do not combine different hardware,
precision, dataset versions or tuning budgets into one apparent improvement trend.
