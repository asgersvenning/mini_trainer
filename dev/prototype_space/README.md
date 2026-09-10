# Prototype-space exploration

The reusable tool is installed as `mt_explore`; see the
[launcher and export guide](../../docs/prototype-explorer.md). Runtime code and
viewer assets live in `mini_trainer.visualization.prototype_space`. These developer
commands remain as compatibility entry points and regression tools. The accompanying
[log-domain diagnostic API](../../docs/prototype-diagnostics.md) also supports
evaluation logging during training. The existing weight parametrization,
similarity transform and class distance remain the empirical reference.
Alternative evaluations and displays are labelled explicitly. Begin with real
checkpoint evidence; synthetic cases probe calculations and rendering, not how
the learned space ought to organize.

Run from the feature worktree with the existing environment:

```bash
export PYTHONPATH="$PWD"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m dev.prototype_space.explore \
  --weights /absolute/path/to/best_global-lepi-production-w32-1_epoch4.pt \
  --output tmp/prototype-report --angular-tsne
```

Open `tmp/prototype-report/explorer.html` in a browser. The numerical views are self-contained. For optional GBIF image labels and a
clickable local HTTP URL:

```bash
.venv/bin/python -m dev.prototype_space.serve \
  --directory tmp/prototype-report --port 8765
```

Then open <http://localhost:8765/explorer.html>. This serves only the report
directory on loopback. Keep each concurrent task's output path and port distinct.

Generation uses CPU float32 repository APIs for the full class-by-class matrices
and SciPy for the unchanged Ward linkage. Several GB of RAM are needed for the
12,632-class example; this is an exploratory offline tool, not a bounded-memory
replacement for training-time logging. No dependency installation is performed.
The environment needs the existing visualization dependencies (including SciPy).

## Try the views

1. Search a checkpoint class ID or choose the strongest/weakest nearest pair.
   Follow neighbours directly from the table. The selected class updates the
   profile, highlighted tree path and global matrix position.
2. Click a collapsed dendrogram tip to focus on that subtree. Use **Parent**,
   **Whole tree** and **Locate selected class** to retain context. This is the
   current Ward tree in a different, progressively expanded layout.
3. Compare matrix block maxima with block minima using the same leaf order and
   shared colour scale. The latter retains isolated close pairs that maxima can
   conceal. Both exclude self-pairs and padding. Choose float32 log-domain tails
   or the legacy rounded-CDF view. Choose the familiar `1e-8` to `1` display
   window, an observed 0.1–99.9 percentile window, full range or manual log10
   bounds. Clipping affects colours only; each panel reports clipped block counts
   and hover retains raw values. These are not pixel-identical copies of the
   original heatmap.
4. In the local matrix, switch between **Baseline class distance**, **Pre-CDF
   z-score** and **−log₁₀ tail (float32)**. Hover a cell for its stored values. Local
   colour ranges adapt and are disclosed; use the table for exact comparisons.
5. Switch cases to independent unit vectors, four planted groups or algebraic
   edge cases. All use the real embedding width, but have much smaller class
   counts. Their nearest-neighbour distributions are not matched packing controls.

## Spatial projection

See the [hyperspherical visualization review](../../docs/hyperspherical-visualization.md)
for the geometry, measured compression losses, projection alternatives, Python
and browser implementations, and proposed anchor/slice experiments.

The map projects all prototype directions into a common 2D coordinate system.
Choose **Angular t-SNE**, **PCA 1–2** or **PCA 3–4**, scroll to zoom, drag to pan, and click a point
to update the existing inspector and image browser. **Fit selected neighbourhood**
zooms to the selected class and its original-space neighbours without refitting
the projection. Selection preserves the viewing transform; **Fit all** resets it.

The PCA calculation normalizes effective rows using the same directional geometry as
the cosine diagnostic, centers them, and computes PCA through float32 covariance
eigendecomposition. No whitening or per-feature standardization is applied.
Axes use equal spatial scale. Eigenvector signs are fixed for reproducibility;
degenerate eigenspaces can still rotate between numerical implementations. See
the [PCA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

Each plane reports its retained variance and the overlap between its Euclidean
top-k neighbours and the original z-ranked top-k neighbours, averaged across all
classes and for the selected class. Projected ties use checkpoint row order. Teal
points and links identify original-space neighbours; orange rings identify extra
2D neighbours. A low overlap is evidence that this view loses local geometry.
Neither projected proximity nor a visible gap replaces the original scores.
Each synthetic case is projected independently; axes are not aligned across cases.

For the epoch-4 checkpoint, PC 1–2 retains 0.6784% of directional variance and
0.6551% of original top-12 neighbours on average; PC 3–4 retains 0.6234% and
0.7026%, respectively. These measured values show that global two-axis PCA is
very lossy for these weights. It provides a linear reference for future
neighbourhood-preserving projections.

### Angular map

`--angular-tsne` also fits a nonlinear map using angular distances in radians,
recovered from the unchanged repository transform as `pi/2 - z/sqrt(D-2)`.
This retains its cosine clamping and avoids the saturated CDF. t-SNE uses those
precomputed angular distances, random initialization with seed 42, perplexity
`min(30, (classes-1)/3)`, automatic learning rate, 1,000 iterations and Barnes-Hut
angle 0.5. It does not use PCA preprocessing. The angular map is selected first
when present; omit the flag for the faster PCA-only report.

On the real checkpoint it retains 48.62% of original top-12 neighbours, compared
with 0.66% for PC1–2. This is an observed result for these settings and weights,
not evidence that map areas, gaps between groups or relative group sizes measure
spherical geometry. Variance explained is not defined for this map. Its
parameters, KL divergence and neighbour overlap are stored with the coordinates.
See the [t-SNE reference](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

### Thumbnails on the map

Enable **GBIF thumbnails on map** independently of the image-card panel. Configure
image size (32–160 CSS pixels), density (5–200 images per megapixel of map viewport)
and allowable pairwise rectangle overlap (0–75%). The density gives a maximum
budget, capped at 200 images; viewport and overlap culling can produce fewer.
The overlap fraction uses the full thumbnail rectangle including its class label
when labels are enabled, normalized by the smaller rectangle for unequal sizes.
Disable **Labels beneath images** for square image-only
thumbnails with no caption, border or padding; images fill the square by cropping
as needed. Enable **Original aspect ratio** to preserve image proportions without
cropping; the size slider then sets the longest image edge. This works with or
without labels. Collision forces use the measured rectangular footprint. Cached
image proportions also inform candidate culling; unknown images initially reserve
a conservative square until decoding supplies their dimensions. With pushing
disabled, unknown proportions reserve nonoverlapping squares so a later narrow
image cannot become fully covered under a misleading overlap fraction. Names, IDs and
credits remain available on hover or keyboard focus.

Images initially stay centered on actual projected points. Enable **Push thumbnails
apart** to admit candidates using the overlap setting, then separate their
rectangles with animated contact repulsion, spring attraction toward their anchors,
and damped velocity. Contacts use the rendered CSS border boxes (including labels
when enabled), with a two-pixel clearance and softly blended corner normals.
Separated boxes outside that clearance do not repel. Sizes, clearance, offsets
and velocities are measured in screen pixels, independent of projection zoom.
Images may move up to twice their configured size in screen pixels.
**Show thumbnail anchors** toggles the anchor markers and leader lines without
reloading images or restarting the simulation. They are hidden by default.
Settling never removes or fades an admitted thumbnail. The solver minimizes a
fixed sum of anchor displacement energy and squared, smoothed rectangle-contact
penetration. Contact curvature scales each coordinate's step; damped momentum
and backtracking keep accepted steps from increasing that energy. Anchor
attraction does not decay on a timer.

Rest requires eight consecutive steps with movement below 0.04 CSS pixels,
energy change below 0.002 per thumbnail, and maximum overlap-fraction change
below 0.0001. At most two solver steps run per animation frame. A 480-step work
limit bounds expensive cases; the status distinguishes **resting** from **work
limit** and reports overlapping pairs that remain visible. Neither resting nor
the work limit guarantees zero overlap or a global energy minimum. Viewport and
movement bounds can make overlap unavoidable. Reduced-motion preferences run
the same bounded solver without animation.

Numerical targets are separate from displayed thumbnail centers. Two cascaded
filters ease visible motion in and out, using elapsed seconds and CSS pixels:
a guide approaches the target at at most 140 px/s, and the displayed image
follows continuously with an 8/s response rate. This also bounds acceleration
(to 2240 px/s², including reversals). Substeps keep motion consistent across
refresh rates; background pauses advance at most 50 ms instead of catching up.
The animation continues after numerical convergence until displayed speed is
below 0.02 px/s and target error below 0.005 px, with no final snap. Image arrivals
join the existing frame loop and completion promise, retaining its clock, solver
destinations, and filter state. Each image waits independently for its appearance
delay before joining rectangle contacts; existing contacts keep advancing.
Targets follow class IDs even when admission order changes; pan/zoom translates it with the anchor. Hit rectangles
and optional tethers follow displayed positions. Smoothing can leave temporary
contacts during motion; energy descent describes numerical targets, not every
displayed frame. `node dev/prototype_space/check_layout.mjs` checks full-motion
speed and acceleration, reversals, and 30/60/120 Hz agreement.

The status reports the remaining visible slots. Increase allowable overlap to try
a denser candidate set;
this does not guarantee that every admitted image can be separated within the
movement bounds. Initial viewport/density/overlap admission remains separate
from settling; an admitted image is never removed merely because motion stops. Thumbnails culled
by viewport navigation fade out over 100 ms, with hit targets removed immediately.
Reduced-motion preferences remove them immediately.

Culling prioritizes the selected
class and its original-space neighbours, then uses a stable mixed class order.
Only fully visible, admitted rectangles trigger image requests. Up to four images
load concurrently. Each decoded image fades in at its anchor before it joins the
force simulation; unloaded images neither appear as blank cards nor exert forces.
Pan/zoom cancels stale requests and pauses the simulation. Loaded thumbnails
follow their transformed anchors immediately, retaining their screen-pixel
offsets and DOM image nodes. After a short pause, the visible candidate set is
updated and collision adjustment resumes; surviving images are neither reloaded
nor faded in again. No new images are fetched while dragging continuously.
Changing image size, aspect ratio mode, label mode, push mode, case or projection plane rebuilds the
layout because its footprints or coordinate meaning have changed. Cached metadata is
shared with the class cards, and changing a class example also refreshes map
thumbnails. Decoded image references are bounded to 128 entries; server caching
remains shared. Missing photos leave the underlying point visible. Retry clears
failed lookups. Synthetic cases make no photo requests.

Click a thumbnail to inspect the class. Hover or keyboard focus exposes the
scientific name, exact ID, creator, image license and source links below the map.
Culling and image repulsion never change scores, prototype coordinates or
neighbour ranks. The controls affect image labels and their displayed positions.

## Class image labels

Enable **Interpret numeric class IDs as GBIF taxa and load photos** to see the
selected class and its twelve nearest prototype directions as image cards. Click
an image to navigate to that class; **Another example** cycles available images.
The prototype is the learned target direction, and the photo provides a class
label. Selecting images by their measured embedding proximity is a separate
possible extension. Exact checkpoint IDs remain unchanged, including when GBIF
reports a different accepted taxon ID.

Only the visible neighbourhood is requested. Metadata and thumbnails are cached
in a sibling `prototype-report-gbif-cache` directory; upstream requests are
serialized, while cached images and already-resolved class metadata bypass that
queue. The browser can therefore load cached images while an upstream request
is still pending. Synthetic cases never trigger lookups. Missing images or network
access leave the numerical views usable. Media credits and licenses are shown
when supplied; occurrence-data licenses are never substituted for image licenses.
The service uses the [GBIF image API](https://techdocs.gbif.org/en/openapi/images).

## Numerical contract

- Supported inputs are float32 `Classifier` and `HierarchicalClassifier`
  checkpoints with a single linear prototype matrix. Other head families and
  weight encodings are rejected. This does not replace the general model loader.
- Reconstruct the parametrized linear layer with `Classifier._normalize_layer`
  and strict state loading. Keep effective magnitudes, biases and exact class
  order. Do not treat weight-norm direction parameters as effective weights.
- The diagnostic adapter exposes those effective weights to the unchanged
  `class_similarity(cdf=False)`, `class_similarity(cdf=True)` and `class_distance`
  APIs. It does not execute the backbone, hidden layers or hierarchical inference.
  Like the existing diagnostics, these describe directions, not sample occupancy.
- The global tree uses Ward linkage on the current distance. The global heatmap
  quantity is the current `1 − CDF(z)`. No alternative metric replaces either.
- Neighbours are explicitly ranked by pre-CDF z, with stable class-index ordering
  for exact z ties. This refines saturated distance ties rather than arbitrarily
  treating zero-distance pairs as identical prototypes.
- The primary tail view uses `class_log_similarity(model, complement=True)` and
  displays `−log_tail / log(10)` in float32. It neither materializes a near-one CDF
  nor promotes the z-score or log-probability matrix to float64.
- The separate float64 reference computes `−torch.special.log_ndtr(z.double())`
  for comparison (hover baseline distances in the neighbour table). It retains
  the existing z and does not apply the baseline probability floor. Reference
  neighbour distances are stored as float64; baseline and log-tail values remain
  float32. Numerical values are never recovered from colours.
- Linear probabilities can still underflow at extreme z even in float64; log-tail
  remains useful there. The algebraic fixture includes this case. A log-domain
  view improves numerical range, not the statistical assumptions of the transform.

The prototype checkpoint has 12,632 unit-length rows in 1,280 dimensions and zero
biases. The initial full baseline evaluation found 249,324 unordered off-diagonal
pairs with distance zero, involving every class. That is numerical evidence to
investigate, not evidence that the vectors are identical or that training failed.
All 249,324 become positive under direct float64 log-CDF evaluation; that reference
confirms the saturation independently of the primary float32 log-tail display.
Current run statistics and checkpoint SHA256 are in `summary.json`; the complete
view payload is in `report-data.json`. Generated data and weights stay out of Git.

## Validation and next directions

```bash
bash dev/check.sh static
node dev/prototype_space/check_layout.mjs
bash dev/check.sh test tests/utils/test_prototype_exploration.py \
  tests/utils/test_dendrogram.py tests/utils/test_plot.py
```

Tests cover effective weights with non-unit magnitudes, class ordering, unchanged
checkpoint bytes, unsupported heads, self-pair/padding exclusion, saturation ties,
exact local baseline submatrices and log-domain values against an independent
`math.erfc` reference. Browser interaction checks additionally exercise case
switching, class search, neighbour navigation, tree navigation and matrix modes.

Promising follow-ups after trying this prototype:

- Evaluate log-domain clustering with explicit representation and compatibility
  decisions. Compare any changed tree against this preserved baseline before
  promoting it into training-time logging.
- Add mutual-neighbour graph exploration to expose links across tree branches.
- Add taxonomy composition at multiple neighbourhood sizes, retaining unknown
  labels separately and avoiding any assumption of taxonomic agreement.
- Compare neighbour and crowding trajectories across checkpoints using class IDs;
  do not interpret unconstrained embedding rotations as motion between runs.
- With held-out embeddings, add image inspection, sample alignment, margins and
  observed confusion. With an explicitly chosen direction distribution, examine
  decision-region occupancy. Neither is established by the prototype matrix alone.

The code is on `feature/prototype-space`. See the [worktree guide](../worktrees.md)
and [agent coordination rules](../../.agents/rules/worktrees.md) for parallel work.

Optional browser interaction verification with an existing Chromium installation:

```bash
CHROMIUM_BIN=/absolute/path/to/chrome node dev/prototype_space/check_browser.mjs \
  http://localhost:8765/explorer.html tmp/prototype-browser-check
```

No browser dependency is installed by this helper. Screenshots and the interaction
check report stay in the ignored output directory.

## Packaged launcher checks

`tests/utils/test_prototype_launcher.py` checks portable generation, class order,
HTTP upload, invalid-file recovery, and separation of weights from served files.
`dev/check-wheel.sh` checks the installed entry point, bundled assets, and the
missing-extra message in a minimal environment.

For the browser file-picker path, start `mt_explore --no-browser` (or its module
equivalent) and run:

```bash
CHROMIUM_BIN=/path/to/chrome node dev/prototype_space/check_launcher.mjs \
  http://localhost:PORT/ tmp/launcher-check /absolute/path/to/small-weights.pt
```

This selects the file with Chromium's native input, generates the default angular
map, opens the report, and returns to the picker. Use a small supported checkpoint
for this bounded regression check. The original `check_browser.mjs` exercises the
full real-checkpoint report with its three synthetic comparison cases.
