# Optional outer test-time augmentation

The completed [full composed-TTA comparison](mambo-composed-tta.md) evaluates the
three shortlisted recipes on both backends, with calibration and matched coverage.

TTA is an opt-in deployment feature shared by PyTorch and ONNX. View generation
operates on decoded images **before** the ordinary, unchanged preprocessing recipe.
It does not depend on backbone internals, ONNX graph changes, an intermediate crop
size or the selected class list.

| Profile | Views | Spatial policy |
|---|---:|---|
| `none` | 1 | Ordinary single-view path; default |
| `rotation30_pad25_3` | 3 | Original plus ±30° rotations, each with 25% edge padding; default when TTA is enabled |
| `wide_rotation_mixed_padding_5` | 5 | Original, ±10° with 15% padding, ±30° with 25% padding |
| `padded_scale` | 3 | Original plus 8% / 15% edge padding; previous default, available explicitly |
| `hflip` | 2 | Original and horizontal reflection |
| `five_crop` | 5 | Original and four corner crops, each 90% of original height/width |
| `ten_crop` | 10 | Five-crop views and their horizontal reflections |
| `d4` | 8 | Rotations of 0/90/180/270 degrees and their horizontal reflections |
| `light_noise` | 3 | Original plus two independently seeded 1% salt-and-pepper views |

Enable the recommended recipe with `Predictor(..., tta=True)` or bare `--tta`.
Both resolve to `rotation30_pad25_3`, also available explicitly. Omitting TTA
keeps single-view inference; `tta=False` and `--tta none` explicitly disable it.
Expanded-canvas rotations preserve the source extent, fill corners with RGB
(124,116,104), then edge-pad each axis by 25% per side before ordinary preprocessing.
The full-data comparison supports this promotion on both backends. Explicit
`padded_scale` retains the former behavior. `RotatePad(degrees, padding)` and
`EdgePad(fraction)` expose these transforms for custom policies.

`SaltAndPepper(proportion=0.01, seed=0)` is also available as a public transform.
It uses one RGB-shared black/white pixel mask and an image-keyed seed, so built-in
noise is reproducible across batch sizes and preparation worker counts. It operates
on the decoded source before normal preprocessing; this is not a claim of exact
training-pipeline RNG or noise-placement equivalence.

These are convenience profiles, not restrictions on the interface. `TTA` accepts
an ordered finite sequence of arbitrary callables. `View` implements fractional
crops, quarter-turn rotations and reflection; arbitrary rotations, scales, color
transforms or other policies can be supplied by a caller. Each callable receives
its own uint8 RGB CHW copy of the decoded image. It can return a CHW array or PIL
image accepted by the existing preprocessing function. Random custom transforms
are the caller's responsibility; built-ins are deterministic. Give custom policies
a descriptive name for output provenance.

For each image batch, the outer layer decodes once, prepares one view at a time,
and invokes the ordinary runtime. Runtime batches never grow by the view count.
It retains the decoded batch and the current prepared view, not all prepared views.
Host memory also depends on original image dimensions; use smaller batches for
large source images. Species logits are averaged in FP32, then the ordinary class mask, hierarchy and
confidence normalization are applied. This is **logit averaging**, not voting or
averaging already-normalized probabilities. Preset and custom-list semantics stay
aligned. Output metadata records the policy name and view count.

When requested, each view uses the normal embedding path. Its embeddings are
averaged in FP32 and normalized to unit length; a nonfinite or near-zero mean
raises an error. Averaged embeddings have not been qualified for downstream
retrieval/clustering. The default single-view representation is unchanged.

## Evidence and selection

The [selection record](mambo-composed-tta.md) explains why the three-view
rotation-and-padding recipe replaced `padded_scale`. Current quality, coverage
and timings are in the [deployment comparison](../deployment/README.md#release-comparison).

TTA was selected using Flemming, including part of its reporting partition.
Its benefit is domain-dependent; it is not a universal improvement or an
independently validated recipe choice. Earlier crop/noise/reflection sweeps
remain in [historical study results](https://github.com/asgersvenning/mini_trainer/blob/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/docs/mambo-tta.md).

The early sweep runners and chart generator are retired; replay that completed
study from [its pinned source](https://github.com/asgersvenning/mini_trainer/tree/b631c52c74b5bae2bd1d3addc517af2095ef7f7b/dev/releases/mambo_v3).
The composed-study collectors and current release report generators remain maintained.
