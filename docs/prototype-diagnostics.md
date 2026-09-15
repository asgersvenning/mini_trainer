# Prototype diagnostics in log space

Prototype diagnostics use the classifier's effective last-layer weights and the
existing dimension-dependent cosine-to-z transform. The log-domain path changes
how the resulting Gaussian quantities are evaluated and displayed; it does not
fit a new distribution or impose a different organization on the learned space.

```python
from mini_trainer.modeling import class_log_similarity
from mini_trainer.visualization import plot_class_distance_matrix

log_cdf = class_log_similarity(model)  # log Phi(z), one matrix per level
log_tail = class_log_similarity(model, complement=True)  # log Phi(-z)
figures = plot_class_distance_matrix(model, log_domain=True)
```

Numerical range and colour range are independent. For example, add
`log_range=(-8 * math.log(10), 0.0)` (with `import math`) to focus colours on
probabilities from `1e-8` to `1`. These are natural-log display bounds. Values
outside them receive endpoint colours; the computed matrix is not clipped.
Omit `log_range` for the full computed colour range. Small outliers can compress
the visually interesting region in that full-range view.

`class_log_similarity` returns float32 natural logs on the weights' device. It
uses the existing z-score computation, disables any surrounding autocast for the
diagnostic, and evaluates `torch.special.log_ndtr` directly. The self-similarity
diagonal is exactly 0 in log-CDF and `-inf` in log-tail. No gradients or probability
floor are used. Quantized-training weights follow the existing dequantization
handling before the diagnostic, not a separate prototype interpretation.

The complementary representation is useful for strongly aligned prototypes:
`log Phi(-z)` remains representable when `Phi(z)` rounds to 1. In contrast, even
directly evaluating `-log Phi(z)` can yield a distance too small for float32. Keep
log-tail values for analysis/rendering rather than exponentiating them back into
probabilities. Higher precision is useful as an independent reference, not required
by this path. See the [PyTorch definition of log_ndtr](https://docs.pytorch.org/docs/stable/special.html#torch.special.log_ndtr).

## Rendering and training-time evaluation

`plot_heatmap(..., log_input=True)` accepts natural-log values. It performs maximum
aggregation and colour normalization directly on logs, pads with `-inf`, and
formats colourbar ticks in exponent notation without creating tiny linear tick
values. `min_val_display`, if supplied, remains a nonnegative linear threshold.
Zeros (`-inf`), NaNs and infinities are masked. Ordinary `plot_heatmap` defaults
and its legacy pixel mapping remain unchanged.

Evaluation logging during training now calls the log-domain renderer with
`log_range=(-8 * math.log(10), 0.0)` under the existing
`Class distance matrix/lvl...` figure tags. The represented quantity is still the
complementary probability, with percentage labels; near-neighbour distinctions
are no longer lost by subtracting a rounded CDF from 1. This uses float32, including
the log-probability matrix, and no matrix-wide float64 promotion.
The initial `1e-8` to `1` colour window follows the familiar legacy display range,
so extremely tiny tails do not consume most of the palette. Endpoints show
clipping with ≤/≥ labels. Distinctions below the display floor remain in the
log scores and can be explored with a different range; this is not a statistical
cutoff or a universal choice of interesting scale.

The existing `class_similarity`, `class_distance`, and default
`plot_class_distance_matrix(model)` remain available as the known baseline.
The dendrogram still uses its existing Ward linkage and distance. Do not feed
log-tail or log-distance values directly into Ward and call that the same tree:
nonlinear distance transformations can change its merges. A future stable
clustering implementation needs its own numerical and behavioural validation.

These diagnostics run during evaluation and do not alter training logits, loss,
optimizer updates or saved checkpoint formats. They apply to runs using this
feature branch; an already-running training process does not pick them up.

## Evidence and exploration

The epoch-4 example has 12,632 unit-length prototypes in 1,280 dimensions, with
zero biases. Full evaluation found 249,324 unordered off-diagonal pairs whose
legacy distance is zero, involving every class. Direct float64 log-CDF evaluation
gives a positive distance for all of those pairs, confirming numerical saturation
rather than exact equality of their vectors. Float32 log-tail is the primary
representation for the improved views; the higher-precision evaluation is only
a comparison reference.

Tests compare log-tail values against independent `math.erfc` evaluations, reject
CDF calls in the log path, check autocast/device/dtype and multiple prototype levels,
preserve legacy heatmap pixels, and exercise padding and tails beyond linear
probability range. CPU evidence does not establish CUDA execution or GPU cost.

For a runnable report using the real checkpoint and labelled synthetic probes,
see [the prototype explorer](../dev/prototype_space/README.md).
