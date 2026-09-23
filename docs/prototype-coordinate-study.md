# Prototype coordinates for classical statistical tools

Status: bounded empirical study completed, 2026-09-15. This is a research result,
not a change to inference, classifier weights, or the prototype viewer.

## Decision

There is no universal winning “linearization” for this matrix. **Keep the original
weights as the general-purpose representation; offer PCA as task-specific
preprocessing, and full-rank PCA whitening as an optional metric transform.**
A stereographic export is convenient when an explicitly unconstrained spherical
chart and analytic inverse are required, but it did not generally improve the
classical tasks tested here. Its convenience should not be confused with an
empirical downstream advantage.

The clearest improvement was **PCA before Gaussian naive Bayes**. Selecting the
number of components using validation data raised held-out family macro recall
from 73.89% to 81.00% on average. The three splits selected 256, 768 and 512
components, respectively. This is evidence for task-specific decorrelation and
regularization, not a guarantee that a particular global dimension is optimal.

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

1. **Linear family prediction:** original weights retain 95.40% micro accuracy.
   Stereographic coordinates have the highest chart macro recall (80.15%), only
   0.34 percentage points above original weights, while losing micro accuracy.
   This small, split-dependent trade-off does not justify declaring them superior.
   Aggressive dimensional reduction loses discriminative information.
2. **Gaussian-model classification:** PCA is useful. Fixed 512-component PCA
   reaches 81.07% macro recall; validation-selected PCA reaches 81.00%. Merely
   applying log/stereographic coordinates leaves macro recall around 74%.
   Whitening the PCA coordinates does not materially change Gaussian NB here:
   its learned per-coordinate variances already absorb coordinate rescaling.
3. **Genus neighbour prediction:** full-rank whitening gives 79.77% versus 78.88%
   for original weights. That modest improvement is a candidate for retrieval
   workflows, not evidence that whitening preserves the original metric. Charts
   remain around 79%; truncation and the tested kernel features generally lose.
4. **Clustering:** 32-PC mini-batch k-means gives mean held-out family ARI 0.170
   on original weights, 0.178 on log coordinates and 0.183 on stereographic
   coordinates. This is a small coarse-clustering improvement, not an overall
   winner or a validated species-clustering result. Cluster count is fixed to
   the number of eligible families, not selected using test labels.
5. **Reconstruction and neighbourhoods:** original-space PCA at 128 dimensions
   reconstructs at about 70° mean angular error; chart-PCA is roughly 78–81°.
   Stereographic coordinates preserve about 84% of original ten-neighbour sets,
   log coordinates about 89%, and the Lambert radial formula about 92%.
   These are approximate coordinates, not geometry-preserving replacements.

The covariance explains why simply removing a radial constraint does not rescue
low-dimensional PCA. On the first training split, two PCs explain **0.39%** of
variance, 32 explain **5.20%**, 128 explain **17.76%**, and 512 explain **54.42%**.
The information is spread across many dimensions. This does not mean that the
vectors lack useful taxonomic structure: the full-dimensional linear classifier
extracts it very well. A poor two-dimensional PCA picture and useful
high-dimensional linear prediction can coexist.

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

CPU observations on an Intel i7-12800H, with four BLAS threads: mean-anchor charts
fit in about 0.02 s and insert/invert one vector in roughly 11–23 μs. The 40-step
intrinsic-mean estimate takes about 5 s in a dedicated measurement and still has
mean-log residual norm 0.00034; it is not a converged global-mean claim. PCA fitting
ranges from about 0.4 s for this full-covariance solver to 2 s for the separate
randomized 512-component solver. Those timings use different algorithms and are
not a monotonic dimension-scaling comparison. PCA-512 insertion is about 0.2 ms.
Fast PNS-128 takes about 9.4 s including initial reduction and about 1.8 ms per
insertion. The narrow-kernel follow-up takes about 0.5 s including its fitted
preimage and about 1.1–1.2 ms per insertion. These are host-specific observations,
not deployment guarantees or carefully isolated throughput benchmarks.

The PNS implementation follows the fast approximation in Monem, Dryden & George
(2025, §3.3), with three initializations and a bounded optimizer per subsphere.
All selected optimizer stages reported convergence, and held-out reduced-sphere
inversion passed. This does not certify globally optimal axes or establish how
full 1,279-dimensional PNS would perform. Its cost/state scales quadratically in
dimension, making it a higher-investment option than the demonstrated benefit
currently warrants. A known-small-circle numerical check validates the core
subsphere fit; this is not cross-validation against the authors' R package.

Kernel widths 0.01, 0.1, 1, 5 and 20 were evaluated. Broad kernels were better;
narrow kernels largely failed to transfer beyond training landmarks. The table
shows γ=0.01 explicitly, not a claim of globally optimized kernels. At the same
512-coordinate count, PCA substantially outperforms this Nyström representation
for the linear family task. Approximate kernel preimages are not exact inverses.

## Recommended next use

- Retain the original matrix for linear classifiers, original angular similarity
  and fidelity-sensitive work.
- For Gaussian or diagonal-covariance models, fit PCA on the analysis training
  split and select the retained dimension using that task's validation data.
  The tested 256–768 range is useful guidance, not a universal fixed setting.
- Consider full-rank whitening for an explicitly changed nearest-neighbour metric.
- Supply a stereographic companion only when consumers need its unrestricted chart
  and analytic inverse. Save the anchor, row norms, transform conventions and
  numerical error; do not label it a generally improved representation.
- Do not invest in full PNS, custom normalizing flows or autoencoders yet. The
  reduced PNS experiment does not earn that escalation. Flows and autoencoders
  are untested alternatives, not empirically rejected methods.
- Image embedding insertion, domain shift, calibrated densities and time-series
  filtering need task-specific data before making performance claims. The present
  evidence answers which approaches help the **available prototype tasks**.

## Evidence and references

The [research directory](../publication/experiments/prototype_linearization/README.md)
contains the protocol, source references and executable scripts. The local study
bundle `/tmp/prototype-linearization-study-20260915` originally contained raw per-split results,
split indices, executed source, environment information, a comparison figure and
checksums. No model, dataset or large generated matrices are added to Git.

Repository consolidation on 2026-09-23 confirmed that this temporary bundle is
no longer present at that path. The results above are retained from the original
study and were not independently reverified during consolidation. Reproducing
them requires the original input matching the recorded checksum and a fresh run
of the four stages; the scripts alone do not archive the raw evidence.

Numerical checks: three focused tests pass for chart inversion and single-point
insertion, a known small-circle PNS fit with held-out inversion, and the radial
control's endpoint failure. Every benchmark stage completed on the real matrix.
The same comparison is not rerun merely for formatting/documentation edits.

Foundational sources:

- [Fletcher et al., 2004: Principal Geodesic Analysis](https://doi.org/10.1109/TMI.2004.831793).
- [Jung, Dryden & Marron, 2012: Analysis of Principal Nested Spheres](https://www.statistics.pitt.edu/sungkyu/papers/Biometrika-2012-Jung-551-68.pdf).
- [Monem, Dryden & George, 2025: Principal Nested Spheres for High-Dimensional Data](https://arxiv.org/html/2511.08398v1).
- [Williams & Seeger, 2000: Nyström approximation](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf).
- [Mika et al., 1998: Kernel PCA and the preimage problem](https://proceedings.neurips.cc/paper/1998/hash/226d1f15ecd35f784d2a20c3ecf56d7f-Abstract.html).
