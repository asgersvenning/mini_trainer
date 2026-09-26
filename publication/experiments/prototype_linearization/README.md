# Classical-tool representations of production classifier prototypes

Research scope: choose useful coordinates for existing classical statistical tools,
not storage compression or a replacement classifier. The immutable input is the
12,632 x 1,280 effective weight CSV exported through mini_trainer's prototype
loader, including original GBIF taxonomy. Input CSV SHA256:
`94b39070d891e87f129ae152a5754d8c4e5009cc843dffb0e0882911857eeb2f`.

## Protocol

- Three fixed 70/15/15 train/validation/test species splits (42, 43, 44), stratified
  by model family. Families with fewer than ten species are excluded from this
  supervised comparison; 12,473 rows remain. Original row indices are retained.
- Fit anchors, charts, PCA, radial warping, kernels and all predictors on training
  rows only. Choose ridge regularization by validation macro recall over
  `[0.01, 0.1, 1, 10, 100]`; report held-out micro accuracy and macro recall.
- Downstream tasks: family prediction with a classical ridge linear classifier
  and Gaussian naive Bayes; genus prediction with weighted Euclidean 5-NN on
  test genera represented in training; family clustering using mini-batch k-means
  on 32 PCs; and PCA reconstruction at 2, 32 and 128 dimensions.
- Retrieval: ten-neighbour overlap against original-space Euclidean neighbours
  (equivalent ordering to cosine/angular neighbours for unit vectors).
- One scalar RMS-radius normalization for ridge comparisons, not coordinate-wise
  standardization that would erase geometric differences. Gaussian NB is
  coordinate-dependent; PCA rotation itself can change its independence fit.
- Float64 analysis, four BLAS threads, CPU. Timing includes actual fit and batch
  transform but not CSV parsing unless explicitly stated. Each method runs once
  per split; timings are practical observations, not a performance guarantee.
- These are species-prototype tests, not image-level generalization, calibration,
  tracking/Kalman filtering or density-likelihood validation. The model was trained
  with taxonomy-aware objectives: taxonomy is useful external interpretation of
  these rows, but it is not independent of how the prototypes were learned.

## Methods and references

- Original ambient unit weights are the required baseline: classical tools can
  already operate on them; mean renormalization is an available extrinsic estimate.
- Sphere log map at normalized arithmetic mean, and at a 40-step local intrinsic
  mean estimate. This is tangent-space preprocessing, not an exact global PGA
  optimizer. See [Fletcher et al. 2004, DOI](https://doi.org/10.1109/TMI.2004.831793).
- Stereographic chart, with inverse and one excluded antipode. Conformal does not
  mean dot-product preserving. [Reference implementation and mathematics](https://people.math.sc.edu/burkardt/f_src/sphere_stereograph/sphere_stereograph.html).
- `equal_area` in initial result files is the **Lambert radial formula extended to
  high dimension**. It is area preserving on S², not volume preserving on S¹²⁷⁹;
  do not interpret the historical method key as a high-dimensional Jacobian claim.
  [USGS map projection reference](https://pubs.usgs.gov/pp/1395/report.pdf).
- `log_radial` is an explicitly experimental control inspired by the proposed
  radial CDF transform: piecewise-linear empirical quantiles to the large-dimension
  Gaussian-radius approximation. Endpoint clamping makes unseen extreme radii
  noninvertible. It is not a normalizing flow or a published sphere-unrolling method.
- PCA and PCA whitening use the training covariance. Truncated PCA is lossy;
  full-rank whitening is invertible with saved mean/loadings/scales. Whitening
  changes distances and regularization, not just coordinates.
- [Jung, Dryden & Marron 2012, Principal Nested Spheres](https://www.statistics.pitt.edu/sungkyu/papers/Biometrika-2012-Jung-551-68.pdf).
  The score space retains a circular coordinate and bounded residual intervals.
  Our small-sphere least-squares solver is a bounded reference implementation,
  with three initializations per stage, not the authors' R implementation.
- [Monem, Dryden & George 2025, fast PNS preprint](https://arxiv.org/html/2511.08398v1),
  section 3.3: fit PCA to orthogonal tangent coordinates, project log coordinates,
  exponentiate into a lower sphere, then fit PNS. Reducing to 32 dimensions is a
  computationally useful but lossy approximation; full PNS at 1,279 dimensions
  is not benchmarked. Check optimizer diagnostics and reduced-sphere roundtrip.
- [Williams & Seeger 2000, Nyström kernel approximation](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf).
  RBF kernel features are a scalable nonlinear input to linear tools. An explicit
  out-of-sample transform exists; the inverse is only an approximate ridge preimage.
  [Mika et al. 1998 explain the preimage problem](https://proceedings.neurips.cc/paper/1998/hash/226d1f15ecd35f784d2a20c3ecf56d7f-Abstract.html).

## Run

Use the existing environment without synchronizing it. This research uses installed
NumPy, SciPy, scikit-learn and threadpoolctl; the summary plot also uses Matplotlib.
It adds no mini_trainer dependency.

```bash
OPENBLAS_NUM_THREADS=4 .venv/bin/python publication/experiments/prototype_linearization/benchmark.py \
  --input /path/to/classifier_weights.csv.gz --output /tmp/prototype-linearization-baseline
OPENBLAS_NUM_THREADS=4 .venv/bin/python publication/experiments/prototype_linearization/advanced.py \
  --input /path/to/classifier_weights.csv.gz --baseline /tmp/prototype-linearization-baseline \
  --output /tmp/prototype-linearization-advanced
```

Output paths must be fresh. Each method saves results as it finishes. Split manifests
are local binary artifacts; the input, model and generated matrices are not committed.

The follow-up extends PCA to 512 coordinates, RBF widths to 0.01 and 0.1, and
fast PNS to 128 coordinates. `selection.py` selects Gaussian-NB PCA dimension
on validation macro recall from 32, 128, 256, 512, 768, 1024, 1280 and measures
single-point chart insertion/inversion. Both accept the same `--input`,
`--baseline` and fresh `--output` arguments as advanced.py.

Collect the four output directories as `baseline`, `advanced`, `followup` and
`selection` beneath a study directory, then run `summarize.py --study /path/to/study`
to generate the table, figure, summary and checksums. See
[the findings](../../../docs/prototype-coordinate-study.md) for conclusions,
limitations and recommendations.
