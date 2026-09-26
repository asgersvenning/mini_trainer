# Prototype coordinates for classical statistical tools

Status: bounded empirical study completed, 2026-09-15. This is a research result,
not a change to inference, classifier weights, or the prototype viewer.

## Decision

**Keep original weights as the general-purpose representation.** PCA helps
Gaussian naive Bayes; full-rank whitening offers a modest neighbour-prediction
improvement. Stereographic coordinates are useful when an unrestricted chart
with an analytic inverse is required, but did not generally improve these tasks.

Validation-selected PCA raised held-out family macro recall from 73.89% to
81.00%, selecting 256, 768 and 512 components across the three splits. This
supports task-specific preprocessing, not one globally optimal dimension.

## Data and evaluation

The actual production classifier supplies 12,632 species prototypes, each with
1,280 effective FP32 weights. The export was extracted with mini_trainer's
`load_prototypes`, preserving class order, GBIF IDs and model hierarchy. Analysis
normalizes the tiny FP32 norm deviations and uses float64; it does not overwrite
the original matrix. Input SHA256 and exact commands are in the
[reproduction guide](../publication/experiments/prototype_linearization/README.md).

Supervised evaluation includes 12,473 species from 57 families with at least ten
species each; the remaining 159 species in 47 smaller families are excluded.
There are three fixed species splits, stratified by family: 8,731 fit, 1,871
validation and 1,871 test rows. All anchors, projections, whitening, radial maps,
kernel landmarks and downstream estimators are fitted using training rows only.
Ridge regularization is selected on validation macro recall. Genus 5-NN metrics
condition on the true genus being represented among training species; coverage
is recorded separately in the baseline results.

These are **held-out species-prototype tasks**, not held-out image classification.
The original model learned with hierarchical supervision; its taxonomy is not an
independent property of how these vectors were constructed. Gaussian naive Bayes
is a concrete classical Gaussian-model task; no claim is made about arbitrary
GMMs, calibrated density likelihoods, regression or Kalman tracking. The study is
exploratory: the follow-up methods were chosen after inspecting initial results,
so it is not a preregistered independent model-selection benchmark.

All table entries are percentages, mean ± sample SD across three splits. SD is
not a confidence interval; the splits overlap. Family “macro” means macro recall
(balanced accuracy), not mean one-vs-rest binary accuracy.

| Representation | Linear family macro recall | Linear family micro accuracy | Gaussian NB family macro recall | Genus 5-NN accuracy |
|---|---:|---:|---:|---:|
| ambient | 79.81 ± 2.61 | 95.40 ± 0.33 | 73.89 ± 2.42 | 78.88 ± 0.23 |
| log_mean | 79.85 ± 1.82 | 95.24 ± 0.23 | 73.71 ± 2.06 | 78.89 ± 0.56 |
| log_intrinsic | 79.26 ± 2.60 | 95.23 ± 0.25 | 73.67 ± 2.13 | 78.96 ± 0.51 |
| stereographic | 80.15 ± 1.52 | 95.24 ± 0.19 | 73.70 ± 2.03 | 78.60 ± 0.66 |
| equal_area | 79.75 ± 1.88 | 95.23 ± 0.25 | 73.66 ± 2.16 | 79.07 ± 0.46 |
| log_radial | 79.85 ± 1.82 | 95.24 ± 0.23 | 73.71 ± 2.06 | 78.84 ± 0.55 |
| pca128 | 39.17 ± 1.77 | 87.42 ± 0.64 | 77.69 ± 1.55 | 66.00 ± 1.77 |
| pca512 | 63.98 ± 1.87 | 93.37 ± 0.33 | 81.07 ± 1.94 | 76.80 ± 0.25 |
| pca1280_whiten | 79.56 ± 2.18 | 95.35 ± 0.23 | 69.81 ± 1.82 | 79.77 ± 0.41 |
| fast_pns32 | 21.98 ± 0.96 | 71.16 ± 2.62 | 66.55 ± 2.07 | 47.69 ± 0.28 |
| fast_pns128 | 35.64 ± 1.53 | 84.95 ± 0.57 | 76.41 ± 2.30 | 66.49 ± 2.07 |
| nystrom512_gamma0.01 | 44.28 ± 1.76 | 81.26 ± 0.65 | 67.07 ± 1.61 | 74.79 ± 0.69 |

The table's `equal_area` key means the Lambert radial formula extended to this
sphere; it **does not preserve high-dimensional volume**. `log_intrinsic` uses
40 mean-update iterations, not a certified global Fréchet mean. `log_radial` is
the proposed radial-warp idea implemented as a deliberately simple control.

## What the downstream evidence supports

- **Linear family prediction:** stereographic macro recall exceeds the original
  weights by 0.34 percentage points but loses micro accuracy. That small,
  split-dependent trade-off does not establish superiority; aggressive dimension
  reduction loses discriminative information.
- **Gaussian NB:** PCA helps; charts alone do not. Whitening does not materially
  change Gaussian NB here because its learned coordinate variances absorb scaling.
- **Genus neighbours:** full-rank whitening modestly improves prediction by
  changing the metric. Truncation and the tested kernel features generally lose.
- **Clustering:** 32-PC mini-batch k-means gives mean held-out family ARI 0.170 for
  original weights, 0.178 for log coordinates and 0.183 for stereographic
  coordinates. Cluster count is the number of eligible families, not selected
  using test labels. This does not establish species-clustering performance.
- **Reconstruction and neighbourhoods:** PCA at 128 dimensions gives about 70°
  mean angular reconstruction error in original space versus 78–81° for charts.
  Original ten-neighbour overlap is about 84% for stereographic, 89% for log and
  92% for Lambert coordinates; none preserves the original geometry exactly.

On the first training split, 2, 32, 128 and 512 PCs explain **0.39%, 5.20%,
17.76% and 54.42%** of variance. Information is spread across many dimensions:
a poor two-dimensional PCA view does not rule out useful high-dimensional
linear prediction.

## Principles, simplicity, cost and insertion/inversion

| Method | Principle and practical complexity | New point / inverse | Evidence-based position |
|---|---|---|---|
| Original unit weights | Extrinsic spherical analysis; no fitted transform. Classical tools already accept these vectors. Normalize an average if a unit-direction estimate is needed. | O(d); no information loss in original export. | Best default for linear prediction and fidelity. |
| Sphere log map | Riemannian normal coordinates preserve distances from the anchor. Small implementation with a Householder vector. Curvature remains in the metric. | O(d); analytic exp inverse on the valid chart. Unique log domain excludes the antipode; radii are < π. | Useful local chart, no material general benefit on this dispersed matrix. |
| Stereographic | Conformal chart; arbitrary finite Euclidean coordinates, except one excluded sphere point. Does not preserve global dot products/distances. | O(d); analytic inverse, with numerical conditioning near the excluded pole. | Best simple invertible unrestricted chart if that is explicitly required. |
| PCA | Orthogonal covariance directions; dimension reduction regularizes classical models. Very standard implementation. | O(dk); new points use saved mean/loadings. Exact only when retaining the full basis; truncated inverse is approximate. | Strongest demonstrated preprocessing benefit for Gaussian NB. |
| Full PCA whitening | Invertible covariance rescaling; Euclidean distance becomes a fitted Mahalanobis distance. | O(d²); save mean/loadings/scales. Invertible with positive retained eigenvalues. | Modest genus-neighbour improvement, not best for Gaussian NB. |
| Fast PNS | Successive least-squares small subspheres; more fitting and state than a single chart. Scores retain a circular coordinate and bounded residuals. | O(dp + p²) after reduction; deterministic insertion. Inverse to reduced sphere passes roundtrip, but discarded dimensions cannot be recovered. | Tested p=32 and p=128 versions do not improve overall on matched PCA. |
| Nyström RBF features | Approximate nonlinear kernel feature space for linear tools; 512 training landmarks. | O(md + m²) with dense normalization; approximate learned preimage, no guaranteed inverse. | Five tested kernel widths give no overall advantage here. |
| Radial quantile warp | Distribution-specific heuristic; radial normality does not imply multivariate Gaussianity. | O(d) plus quantile lookup; our clamped interpolation loses extreme held-out radii. | No demonstrated gain; do not promote this control into an export default. |

CPU observations (Intel i7-12800H, four BLAS threads): mean-anchor charts fit
in about 0.02 s and insert/invert a vector in 11–23 μs. The 40-step intrinsic-mean
estimate takes about 5 s with residual mean-log norm 0.00034; this does not certify
a global mean. PCA fits take about 0.4 s for the full-covariance solver versus
2 s for randomized 512-component PCA, so these are not dimension-scaling timings.
PCA-512 insertion takes about 0.2 ms; PNS-128 takes 9.4 s to fit including reduction
and 1.8 ms per insertion. The narrow-kernel follow-up takes about 0.5 s including
preimage fitting and 1.1–1.2 ms per insertion. These are host-specific observations,
not isolated throughput benchmarks or deployment guarantees.

Fast PNS uses Monem, Dryden & George (2025, §3.3), three initializations and a
bounded optimizer per subsphere. Selected stages reported convergence and
held-out reduced-sphere inversion passed; a known-small-circle check validates
the core fit. This does not certify global optima, agreement with the authors' R
package, or full 1,279-dimensional PNS, whose cost/state scales quadratically.

Kernel widths 0.01, 0.1, 1, 5 and 20 were tested. Broad kernels transferred better;
narrow kernels largely failed beyond training landmarks. The table shows γ=0.01,
not globally optimized kernels. At 512 coordinates, PCA substantially outperformed
Nyström for linear family prediction; kernel preimages remain approximate.

## Recommended next use

- Fit PCA on analysis training data and choose dimension on validation data;
  the observed 256–768 range is guidance, not a fixed default.
- Use whitening only when a changed neighbour metric is intended. A stereographic
  companion should retain its anchor, row norms, conventions and numerical error.
- Reduced PNS results do not justify escalating to full PNS or custom flows and
  autoencoders; the latter are untested, not empirically rejected.
- Image embeddings, domain shift, calibrated densities and time-series filtering
  need their own data. These recommendations concern the available prototype tasks.

## Evidence and references

The [reproduction guide](../publication/experiments/prototype_linearization/README.md)
owns the protocol, source references and commands. The original local bundle
`/tmp/prototype-linearization-study-20260915` held per-split results, indices,
executed source, environment details, a figure and checksums. It was absent when
checked on 2026-09-23. **The numerical
results above are retained historical findings, not independently reverified
results.** Reproduction requires the checksum-matched input and four fresh stages;
the scripts do not archive the raw evidence.

The original study recorded completion on the real matrix and three passing
numerical checks: chart inversion/insertion, a small-circle PNS fit with held-out
inversion, and radial endpoint failure. Documentation cleanup does not rerun or
extend that qualification. No model, dataset or large generated matrices are
tracked.
