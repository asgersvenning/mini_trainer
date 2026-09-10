# Visualizing prototype packing on a unit hypersphere

Research review, 2026-09-10. Scope: mathematical methods, Python implementations,
and browser rendering for the prototype explorer. This is a research deliverable;
methods described as candidates have not been added to training or the explorer.

The unit sphere is the geometry in which we interpret the observed prototypes.
It does not prescribe their distribution, require clusters, imply uniformity, or
establish a lower-dimensional latent surface. Evaluate a display against the
actual weights and the existing mini_trainer diagnostics. Synthetic arrangements
are controlled tests of what a method reveals or distorts.

## 1. Separate geometry, dimensional reduction, and rendering

There are three separate decisions:

| Layer | Question | Examples |
| --- | --- | --- |
| Source geometry | What relationships do the weights define? | Angular separation, directional neighbours, spherical caps |
| Reduction or slice | Which relationships will the display retain? | Anchor distances, angular t-SNE, spherical stress, a great-sphere slice |
| Rendering | How are the resulting coordinates shown? | Plane, rotatable globe, cartographic map, culled photo labels |

An angular input metric does not imply spherical output. Conversely, placing a
layout on a globe does not establish that it preserves the original angles.
Both planar and spherical layouts can be useful when their losses are measured.

For effective weight rows, let

\[
u_i=w_i/\|w_i\|,\quad c_{ij}=u_i^\top u_j,\quad
\theta_{ij}=\arccos(c_{ij}).
\]

Cosine dissimilarity, chord distance, and angular distance are related by

\[
1-c_{ij}=2\sin^2(\theta_{ij}/2),\qquad
\|u_i-u_j\|=2\sin(\theta_{ij}/2).
\]

These give identical exact neighbour orderings, including ties. They do **not**
give identical distance weights, kernels, stress objectives, or fitted layouts.
Cosine dissimilarity is also not generally a metric satisfying the triangle
inequality. Changing it to angular distance is more than changing axis labels.

### Repository baseline

The explorer loads effective weights through the existing classifier
parametrization and calls the repository distance functions. See
[the numerical contract](../dev/prototype_space/README.md#numerical-contract),
[distance.py](../mini_trainer/modeling/distance.py), and
[the transform](../mini_trainer/utils/_core/math.py).

The current pre-CDF transform is, algebraically,

\[
z_{ij}=\sqrt{D-2}\,(\pi/2-\theta_{ij}^{\rm clamped}).
\]

Thus `pi/2 - z/sqrt(D-2)` recovers the baseline's clamped angles without
inverting a saturated CDF. It does not recover distinctions already lost to the
cosine clamp. Self-distances are explicitly zeroed. Baseline class distance,
pre-CDF z, and log-tail displays remain separately identified.

For probability diagnostics, retain direct log-domain evaluation and separate
colour clipping from numerical values. A projection cannot repair CDF saturation,
and a better colour window cannot repair projection distortion.

### What the real checkpoint establishes

Checkpoint: `best_global-lepi-production-w32-1_epoch4.pt`.
SHA256: `42d50f56cb0c6e3ee17ae64fbbcc4eb2ae24335ffe7080197c479c130fff5a81`.
It contains 12,632 prototype directions in 1,280 coordinates, hence on
\(S^{1279}\). The observed effective norms are within about \(2.4\times10^{-7}\)
of one; biases are zero.

| Measurement | Observed value | Interpretation |
| --- | --- | --- |
| PC1–2 directional variance | 0.6784% | Two linear coordinates retain little total variation |
| PC1–2 mean top-12 neighbour recall | 0.6551% | This particular plane loses most original neighbours |
| Angular t-SNE mean top-12 recall | 48.6212% | A useful measured improvement for the tested settings |
| Linear components for 50%, 80%, 90%, 95%, 99% variance | 354, 794, 1,003, 1,128, 1,246 | Quantifies linear compression loss |
| Centred spectral participation ratio | 901.76 | Descriptive spectrum summary, not intrinsic dimension |
| Mean resultant length | 0.01822 | Describes first-moment cancellation, not uniformity |

Across classes, original-space angular separations have these quantiles:

| Neighbour rank | 10th percentile | Median | 90th percentile |
| --- | --- | --- | --- |
| 1 | 58.32° | 66.29° | 73.21° |
| 5 | 70.88° | 74.83° | 77.65° |
| 12 | 75.64° | 77.93° | 79.85° |
| 32 | 79.53° | 80.89° | 82.21° |

Nearest neighbours are therefore not automatically a small-angle patch. This
matters when assessing tangent approximations; it does not invalidate their
neighbourhood relationships or imply that a different packing was intended.
The spectrum does not rule out every nonlinear representation.

The t-SNE result uses angular inputs, seed 42, perplexity 30, 1,000 iterations,
random initialization, automatic learning rate and Barnes–Hut angle 0.5. It is
one fitted comparison, not a method ranking across seeds or configurations.
Existing payloads are in ignored `tmp/prototype-report/`; additional descriptive
measurements are in `tmp/hypersphere-research/geometry.json`.

## 2. Views that preserve specific geometric facts

### Anchor-centred angular atlas

Select a class direction \(a\). Place every displayed prototype at planar radius
\(r_i=\theta(a,u_i)\), with rings labelled in degrees. Radius then has an exact
meaning under the chosen angular convention. Bearings are a separate display
choice: a projected tangent direction or an optimized arrangement.

For \(0<\theta<\pi\), the spherical logarithm is

\[
\log_a(u)=\frac{\theta}{\sin\theta}(u-\cos\theta\,a),
\qquad \|\log_a(u)\|=\theta.
\]

With a two-column orthonormal tangent basis \(B\), let
\(q_i=B^\top\log_a(u_i)\). Ordinary tangent projection uses \(q_i\) and generally
shrinks radii. An exact-radius display instead uses
\(y_i=\theta_i q_i/\|q_i\|\). This preserves distances **to the anchor only**;
bearings and distances between other points remain lossy. If \(q_i=0\), its
bearing is unresolved and must be handled explicitly. The anchor maps to zero;
an antipode has no unique logarithm/bearing.

This is especially suitable for photo browsing: selected class at the centre,
angular rings, true neighbours highlighted, thumbnails culled without moving
their coordinates. Switching anchors produces an atlas of partial views. It
does not require selecting a global mean. The formulas can use NumPy/PyTorch;
[Geomstats supplies hypersphere log/exp operations](https://geomstats.github.io/_modules/geomstats/geometry/hypersphere.html).

### Two-anchor distance coordinates

Plot \((\theta(a,u_i),\theta(b,u_i))\). Both coordinates are directly interpretable,
although many different directions can coincide. Triangle inequalities restrict
the feasible region, providing a useful diagnostic check. This offers a concrete
way to inspect a close pair, a cross-tree connection, or two competing directions.
It is a distance plot, not a complete spatial embedding.

### Great-sphere slices of directional decision regions

Choose orthonormal columns \(Q\in\mathbb R^{D\times3}\). Directions
\(x=Qs\), \(s\in S^2\), form a genuine great two-sphere in the original space.
On that slice,

\[
u_i^\top x=(Q^\top u_i)^\top s.
\]

Precompute three coefficients per class, then colour each sampled slice direction
by the largest dot product. This evaluates the nearest-prototype directional
partition exactly on the slice, up to numerical and sampling resolution.
**Do not normalize \(Q^\top u_i\)**: doing so changes the competition.

An anchor and two chosen tangent directions define a useful navigable slice.
Two-dimensional subspaces instead give great-circle sweeps. Adjacent cells,
winning-score margins and transitions can be inspected with GBIF image labels.
Some classes have no winning region in a chosen slice; that is a property of
the slice and must not be presented as global absence.

Slice areas are not high-dimensional cell volumes. These are geometric
nearest-direction regions; reproducing the complete hierarchical classifier's
decision rule would require its actual scoring and inference path. Evaluate the
cost of `number_of_classes × number_of_pixels` before interactive full-resolution
rendering; tiles, coarse previews and batched matrix products are natural options.

### Packing profiles alongside any map

Compute original-space cap counts
\(N_i(r)=\#\{j\ne i:\theta_{ij}\le r\}\), neighbour radii \(r_{k,i}\), and
mutual-neighbour links. These expose local packing without treating 2D point
density as spherical density. Display distributions, selected-class curves and
linked outliers. A random/uniform reference can be an explicitly selected
comparison; it is not needed to define these measurements.

## 3. Dimensional-reduction families

| Method | What it fits or preserves | Useful role here | Main limitation |
| --- | --- | --- | --- |
| Centred PCA | Ambient squared reconstruction error | Measured linear reference and dominant directional variation | Optimizes neither angular error nor neighbour recall |
| Great-subspace projection | Directions projected into a low-rank linear subspace, optionally renormalized | Simple spherical reference with residuals | Renormalizing a tiny projection can amplify poorly retained directions |
| Tangent PCA / principal geodesic analysis | Variation expressed around a base point / fitted geodesic subspaces | Anchor or regional analysis | Linearized and exact geodesic objectives differ; base point and angular extent matter |
| Principal nested spheres | Successive fitted lower-dimensional great or small subspheres | Probe curved modes that great subspheres miss | Model flexibility, sequential fitting cost and residual loss need assessment |
| Angular t-SNE | Neighbour-affinity agreement in a plane | Already useful class navigation | Map areas and long-range gaps are not angular measurements |
| Angular-input UMAP | Fuzzy neighbour graph fitted to a chosen output geometry | Compare graph structure at several neighbourhood scales | Local rescaling changes what visible density means |
| Spherical stress / MDS | Explicit pairwise-distance objective on \(S^2\) | A globe with a declared angular fidelity target | Global/local weighting tradeoffs and optimization cost |
| Spherical-output UMAP | Neighbour graph embedded with a spherical output distance | A practical first optimized-globe experiment | Spherical output alone does not preserve source packing |
| DOSNES | Neighbour embedding after doubly stochastic affinity normalization | An informative alternative normalization experiment | Equalizes affinity mass, potentially hiding packing differences of interest |

### PCA, tangent PCA and principal geodesic analysis

PCA is mathematically well defined for unit vectors in ambient Euclidean space.
Its objective simply answers a different question from angular fidelity. Retain
it as a baseline with variance and neighbour measurements.

For a great-subspace projection, inspect both the retained projection norm and
the angular residual to the subspace. Renormalization onto a display sphere
should not conceal the lost component. Centring first changes the geometry:
centred PCA coordinates are not automatically a great-sphere projection.

Tangent PCA takes logarithms at a base point and performs linear PCA there.
Principal geodesic analysis more generally seeks geodesic subspaces; its exact
projection objective is not identical to this linearization. A fitted intrinsic
mean can be useful, but its uniqueness and stability cannot be assumed from
unit norms. An extrinsic mean direction is also different from an intrinsic
Fréchet mean. The observed small resultant alone does not settle either issue.
[Fletcher et al., 2004](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Fletcher04.pdf).

### Principal nested spheres (PNS)

PNS works backwards through successively lower-dimensional subspheres. Allowing
small spheres captures latitude-like curved variation that great subspheres can
miss. This is a fitted representation of the observations, not evidence that
the training objective should produce such a structure. Compare great-only and
small-sphere variants using held-out reconstruction and angular/neighbour errors.
The original work explicitly discusses small-sphere overfitting and selection
procedures; their statistical assumptions should be reviewed separately before
adoption. [Jung, Dryden and Marron, 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3635703/).

The direct Python candidate is
[scikit-pns](https://github.com/JSS95/scikit-pns), with `skpns.IntrinsicPNS` and
`ExtrinsicPNS`, fitting, transformation and inverse transformation. PyPI reported
stable version 1.3.0 during this review. Check the selected release against the
documentation before implementation; do not assume `latest` documentation and
stable package are identical. Sequential fitting through 1,280 input coordinates
needs a real runtime/memory pilot.

[torch-pns](https://github.com/JSS95/torch-pns/blob/master/src/torchpns/intrinsic.py)
currently exposes `torchpns.InverseIntrinsicPNS`: a PyTorch inverse transform
constructed from a fitted scikit-pns estimator. It is not a GPU PNS fitting
implementation. Its source also records an intentional formula difference from
the referenced inverse implementation, making cross-library round-trip checks
necessary before relying on it.

### Optimizing an actual spherical layout

For display directions \(y_i\in S^2\), an explicit angular-stress candidate is

\[
L(Y)=\sum_{(i,j)\in E}w_{ij}
\left[\arccos(y_i^\top y_j)-\theta_{ij}\right]^2.
\]

The pair set and weights define the question: all pairs, neighbours, or a disclosed
mixture of neighbours and sampled distant pairs. If sampling estimates an all-pair
objective, use the appropriate sampling weights; otherwise call it a deliberately
reweighted objective. A fitted global distance scale is another explicit option,
not an unnoticed rescaling of angular units.

A smoother alternative fits dot products,
\(\sum w_{ij}(y_i^\top y_j-\cos\theta_{ij})^2\). It avoids `acos` gradients but
weights angular discrepancies differently, especially near zero and pi. Neither
objective guarantees a unique layout. Compare seeds and report residuals.
Spherical MDS has concrete research implementations in graph drawing, although
graph shortest-path distance and our angular input distance are different inputs.
[Miller, Huroyan and Kobourov, 2022](https://arxiv.org/abs/2209.00191).

Exact preservation of every angle on a globe is impossible for this matrix:
it would require `U @ U.T == Y @ Y.T`, but the latter has rank at most three and
the observed source has many more independent directions. This is an algebraic
limitation of the display, independent of any model for how prototypes are packed.

For custom optimization,
[Pymanopt's `Oblique(3, N)`](https://pymanopt.org/docs/stable/manifolds.html#oblique-manifold)
represents N unit columns, hence N display points. `Sphere(N, 3)` instead imposes
one Frobenius-norm constraint on the whole matrix and is the wrong constraint.
[Geoopt's sphere implementation](https://geoopt.readthedocs.io/en/latest/_modules/geoopt/manifolds/sphere.html)
offers a PyTorch route with the last axis representing a sphere point. Inspect
its dtype-dependent distance clamps before declaring parity with repository
angles. These are optimization building blocks, not turnkey validated layouts.

### UMAP, t-SNE and density-oriented variants

[UMAP input metrics](https://umap-learn.readthedocs.io/en/latest/parameters.html)
include cosine and precomputed distances. Using normalized vectors with cosine
is a sensible comparator; using the baseline-derived angular matrix is the
cleanest initial metric-controlled experiment. Monotone input transformations
retain neighbour order but change the fuzzy edge weights.

[Spherical output is explicitly supported](https://umap-learn.readthedocs.io/en/latest/embedding_space.html)
through `output_metric="haversine"`. The output has two angular coordinates;
`n_components=3` alone would normally mean a Euclidean 3D layout, not a sphere.
The documented conversion to XYZ is
`(sin(t)*cos(p), sin(t)*sin(p), cos(t))`.

There is a significant implementation convention:
[UMAP's source](https://github.com/lmcinnes/umap/blob/master/umap/distances.py)
uses latitude/longitude in `haversine`, but a pi/2 shift in
`haversine_grad`, which is the registered output metric with gradient. Its
first output coordinate therefore follows the colatitude convention used in
the tutorial. Test known poles, quarter-circle pairs, seams and the XYZ distance
identity against the pinned version. Do not reuse input latitude conversion
uncritically for output coordinates.

The existing [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
path already gives an angular-input planar reference.
[openTSNE](https://opentsne.readthedocs.io/en/stable/api/affinity.html) adds useful
affinity controls, including multiscale affinities and `PrecomputedAffinities`.
The latter expects affinities, not a raw angular-distance matrix. Neither is a
drop-in spherical-output t-SNE implementation.

[densMAP](https://umap-learn.readthedocs.io/en/latest/densmap_demo.html) is worth a
controlled comparison when preserving local density is a display objective.
Its density regularization does not turn screen areas into high-dimensional
spherical volumes. Compare it with actual cap counts and neighbour radii.
UMAP's local normalization is part of its construction, not an empirical finding
about these weights. [UMAP paper](https://arxiv.org/abs/1802.03426).

[DOSNES](https://yaolubrain.github.io/dosnes/) explicitly embeds on spheres after
making affinities doubly stochastic. For this task, the normalization is a major
scientific choice: equal affinity mass can remove variation we want to inspect.
The author's [repository](https://github.com/yaolubrain/DOSNES) contains MATLAB
fitting and a JS viewer and links a [separate Python port](https://github.com/Coni63/DOSNES).
That port documents limited checking and discrepancies from the MATLAB results.
Treat it as research code requiring verification, not the first production path.

Graph-distance methods such as Isomap and diffusion-based displays are additional
questions one can ask of a neighbour graph. Their graph path or diffusion
distances must be labelled separately from direct great-circle separation.
They are not needed to respect the unit-sphere input constraint.

## 4. Cartography applies after reduction to S²

Stereographic projection of \(S^{1279}\) gives 1,279 Euclidean coordinates, not
two. A globe or map still needs dimensional reduction, or an actual 2D slice.
Conformality of a later cartographic step does not undo losses in that reduction.

For an actual unit two-sphere, these azimuthal projections use angular distance
\(\theta\) from the map centre and retain bearing:

| Projection | Planar radius | Preserves | Display consequence |
| --- | --- | --- | --- |
| Orthographic | \(\sin\theta\) | Appearance from an infinitely distant camera | Visible hemisphere only; strong limb compression |
| Stereographic | \(2\tan(\theta/2)\) | Local angles | Antipode at infinity; large area distortion |
| Gnomonic | \(\tan\theta\) | Great circles as straight lines | Only an open hemisphere; diverges at 90° |
| Azimuthal equidistant | \(\theta\) | Distances from map centre | Other distances and areas distorted |
| Lambert azimuthal equal-area | \(2\sin(\theta/2)\) | Two-sphere area | Shapes and most distances distorted |

These are unit-sphere formulas, derived with unit scale at the centre.
[Snyder's USGS manual](https://pubs.usgs.gov/publication/pp1395) is the primary
reference for their geometry and domains.

For a radial map of the full \(S^{D-1}\) into a \((D-1)\)-dimensional tangent
space, the volume scaling away from singularities is
`r'(theta) * (r(theta)/sin(theta))**(D-2)`. This follows by dividing the radial
volume elements. A two-sphere equal-area projection therefore does not promise
volume preservation for the original 1,280-coordinate prototypes.

[D3-geo](https://d3js.org/d3-geo/azimuthal) supplies these projections, with
rotation and clipping. Its [coordinate interface](https://d3js.org/d3-geo/projection)
uses `[longitude, latitude]` in degrees. Keep this distinct from radians,
colatitude and XYZ used in numerical libraries.

## 5. Concrete implementation and rendering choices

### Python and computation

| Tool | Verified capability | Appropriate experiment |
| --- | --- | --- |
| Existing NumPy/PyTorch + sklearn | Current effective-weight extraction, baseline angles, PCA and angular t-SNE | Reuse the numerical foundation; implement anchor/slice formulas directly |
| [Geomstats](https://geomstats.github.io/getting_started/first-steps.html) | Hypersphere geometry, `TangentPCA`; NumPy and optional PyTorch backends | Validate log/exp and investigate regional tangent coordinates |
| [scikit-pns](https://scikit-pns.readthedocs.io/en/latest/?badge=latest) | Intrinsic/extrinsic nested-sphere estimators | Small pilot followed by measured scaling to the full matrix |
| Pymanopt / Geoopt | Sphere-constrained optimization primitives | Explicit angular or dot-product stress on a display sphere |
| umap-learn | Custom/precomputed input distances and non-Euclidean output metrics | Compare planar and spherical output with controlled inputs |
| openTSNE | Flexible affinity construction | Multiscale planar comparison without silently changing source geometry |
| [RAPIDS cuML UMAP](https://docs.rapids.ai/api/cuml/legacy/api/generated/cuml.manifold.umap/) | GPU UMAP, cosine metric and neighbour-graph inputs | Larger planar experiments when GPU cost is justified |

The reviewed cuML interface does not expose Python UMAP's `output_metric`; do
not assume spherical-output parity. Its API documentation also needs careful
version-specific reading for pairwise versus precomputed-neighbour inputs.
There is no performance claim here for these packages on this checkpoint.

### Browser and image labels

| Framework | Strength | Integration limit |
| --- | --- | --- |
| Existing Canvas + thumbnail overlay | Already supports this checkpoint, linked inspection and bounded photo requests | Extend first for anchor views; benchmark before replacing |
| [Three.js](https://threejs.org/docs/pages/SpriteMaterial.html) | Rotatable 3D globe, sprite image labels; `sizeAttenuation` control | Need hemisphere/depth handling, screen-space culling, picking and image budgets |
| [deck.gl IconLayer](https://deck.gl/docs/api-reference/layers/icon-layer) | GPU image icons, atlases and pixel-sized billboards | Its collision extension is not equivalent to our rectangle-overlap rule |
| [PAIR ScatterGL](https://github.com/PAIR-code/scatter-gl) | 2D/3D scatter, sprite-sheet images, selection and camera callbacks | Verify pinned dependency integration and atlas/culling behaviour |
| [regl-scatterplot](https://github.com/flekschas/regl-scatterplot) | WebGL planar point rendering and interaction | Add a separate bounded photo-label layer |
| D3-geo | Cartographic projection, rotation, clipping | Handles a two-sphere/map, not 1,280-dimensional reduction |

[deck.gl CollisionFilterExtension](https://deck.gl/docs/api-reference/extensions/collision-filter-extension)
checks feature anchors against rendered coverage and supports collision
priorities. That differs from testing the allowable fraction of overlap between
thumbnail rectangles. Preserve the current size/density/overlap semantics unless
a deliberate UI change is made. Transparent icon anchors also need special care
under its documented alpha handling.

[UMAP-JS](https://github.com/PAIR-code/umap-js) has custom input `distanceFn`,
asynchronous fitting and incremental steps. The reviewed API does not provide
Python UMAP's spherical output option; its README also notes differences in
initialization and specialized metric handling. Async fitting is not by itself
a worker-thread guarantee. Initially, fit in Python and serve coordinates; use
a worker if browser-side fitting later becomes useful.

For every renderer, a thumbnail is a class image label associated with a numerical
point. Culling, visual size and overlap may change label visibility but must not
move prototypes or change their distances. Optional label displacement should
retain an explicit connection to the fixed prototype point; the explorer's
**Push thumbnails apart** mode uses anchor markers and leader lines. A globe adds back-side occlusion to
the existing viewport and overlap tests. Preserve photo provenance and cache
budgets, and expose point counts even where image labels are culled.

## 6. Validation before choosing a default

1. **Geometry contract:** use the checkpoint's effective rows and exact class
   order. Compare against baseline z-derived angular distances, with the clamp
   and self-pair conventions recorded. Test identical, orthogonal, antipodal and
   nearly coincident directions. Verify every coordinate conversion separately.
2. **Neighbour fidelity:** recall at k = 1, 5, 12, 32 and 64; show per-class
   distributions and poor cases as well as a mean. Use original full-population
   neighbours even when selecting a subset of anchors for evaluation.
3. **Distance fidelity:** angular errors and Shepard plots on both near pairs
   and uniformly sampled unordered pairs. Report strata separately so the many
   distant pairs do not conceal failures among neighbours. Specify any scale
   fitting and distinguish it from absolute angular preservation.
4. **Two sources of display loss:** for spherical layouts, evaluate distances on
   the output sphere first, then the additional camera/cartographic distortion.
   Screen-space neighbour recall alone cannot evaluate a globe's geometry.
5. **Packing fidelity:** compare local radii and cap counts to apparent density.
   Do not label a dense t-SNE patch or a large slice cell as a measured
   high-dimensional volume. Inspect mutual neighbours and cross-tree links.
6. **Stability and cost:** multiple seeds, neighbourhood scales, optimization
   traces, peak memory, fit time and interactive latency. A layout change between
   checkpoints needs rotation/alignment treatment; compare original distances
   and neighbour changes regardless of display alignment.
7. **Synthetic probes:** algebraic edge cases, narrow caps, small-circle modes,
   uneven local densities, separated groups and independent directions. Match
   N and D when investigating packing scales. The existing six-point fixture
   with k=5 trivially retains every neighbour; 100% there is not evidence of a
   faithful projection.

No full N-by-N objective is automatically cheap: this checkpoint has 79,777,396
unordered off-diagonal pairs and about 638 MB per dense float32 square matrix
(609 MiB), before copies and working arrays. Offline projection, sparse retained
graphs, sampled diagnostics and periodic training-time summaries have different
cost boundaries. Keep them explicit.

## 7. Recommended next increments

**First: an exact-radius anchor atlas.** It extends the current photo browser,
answers a precise angular question and requires no new package. Add selectable
bearings, angular rings, unresolved-bearing handling and original-neighbour
overlays. Show that radii are exact while bearings are reduced.

**Second: a metric-controlled globe comparison.** Fit angular-input UMAP to a
plane and to spherical output, alongside the existing t-SNE. Add a small custom
spherical-stress pilot if fidelity warrants it. Inspect both original-angle
errors and neighbour recall before deciding whether a globe helps browsing.
Use Three.js for a genuinely rotatable sphere, with the current thumbnail policy.

**Third: interactive great-circle/great-sphere slices.** These address the
partition question directly, complementing the neighbour atlas and compressed
global map. Retain all prototype coefficients when evaluating winning regions;
photo labels identify the classes appearing in the slice.

PNS is a worthwhile research comparator after a bounded numerical/scaling pilot.
Its fitted subspheres answer a different question from browsing arbitrary
observed packing. No choice above changes the training constraint, baseline
distance transform, dendrogram, or colour semantics.

## 8. Evidence and reproducibility boundary

This review inspected primary papers, official documentation and selected
upstream source. Geomstats, Pymanopt, Geoopt, scikit-pns, torch-pns, openTSNE,
UMAP and cuML were not installed or runtime-tested in the shared environment.
Source inspection establishes the cited interface facts, not package parity,
scaling, numerical correctness, GPU support of every path or browser performance.
Pin versions and isolate dependencies when starting those experiments.

The PCA/t-SNE measurements come from the existing runnable explorer and its
stored real-checkpoint payload. The additional spectrum uses float32 normalized
effective rows, centring without whitening, and eigenvalues of `X.T @ X`.
Participation ratio is `(eigenvalues.sum()**2 / eigenvalues.square().sum())`;
component counts use cumulative nonnegative eigenvalues. Mean resultant is the
norm of the average normalized row. Angular quantiles invert the existing
stored z profiles. Medians use `torch.median` (lower middle for even counts).

Source URLs using `latest`, `stable`, `master` or `main` are moving references,
reviewed on the date above. Before implementation, record the exact package
versions/commits used and validate coordinate and numerical contracts against
them. Research references do not override the empirical baseline.
