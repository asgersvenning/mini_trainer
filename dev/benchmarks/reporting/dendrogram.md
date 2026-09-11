# Dendrogram scaling and artifact validation

The renderer uses an iterative SciPy-linkage layout, retaining the original
right-child-first leaf order, distance-based radial positions and flat-cluster
assignments. It does not construct recursive BioPython/pyCirclize tree objects.
Same-color branches share paths; adjacent same-color taxonomy bands are merged.
Circular arcs use bounded chords or cubic Bezier segments instead of dense sampled
polylines.
Every leaf label remains present, including repeated display names. Single-class
levels render without attempting linkage. Taxonomy colors now follow stable
first-occurrence order rather than unordered set iteration.

Training saves dendrogram SVGs with text elements instead of glyph outlines,
and retains all figure types under `model/logs/figures/epoch-NNNN/` even with the
default metrics-only logger. Matrix images are PNGs; dendrograms remain vector
SVGs. Rank zero owns local artifact writes. Existing external logging backends
still receive the figures. A failed export closes all created dendrogram figures.
Font appearance can depend on the SVG viewer's installed fonts.

Species-name resolution uses at most eight concurrent lookups, reuses resolved
class lists across epochs (a bounded process-local cache), and falls back to the
original labels when the service or its disk cache is unavailable. GBIF HTTP
requests now have a ten-second socket timeout. This is not a hard total deadline
for taxonomy preparation. A cold large taxonomy still requires real API requests;
the subsequent epochs reuse the result. Logs distinguish label resolution,
per-level rendering and final export time. Pairwise model distances and Ward
linkage remain quadratic in class count; this change does not establish full
production-taxonomy memory capacity or multi-GPU figure latency.

## Reproduce locally

Use the existing environment with the recommended plotting dependencies. Each
invocation is a fresh process and includes linkage, layout, render and SVG save.
Input distances are seeded synthetic values; model distance computation and
network name lookup are excluded. Both baseline and current renderer use SVG
text, so the comparison isolates layout/path changes.

```bash
git show 863a85c:mini_trainer/visualization/dendrogram.py > /tmp/dendrogram-before.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --module-file /tmp/dendrogram-before.py --output /tmp/dendrogram-baseline
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --output /tmp/dendrogram-current
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 1100 --deep \
    --output /tmp/dendrogram-deep
```

Representative local CPU measurements, Python 3.13, Matplotlib 3.10.9,
BioPython 1.87 and pyCirclize 1.10.1, from the same seeded workload:

| 3,422 classes | Previous renderer | New renderer |
| --- | ---: | ---: |
| Layout/render, including linkage | 18.38 s | 2.78 s |
| Including SVG export | 24.53 s | 6.82 s |
| SVG bytes | 4,309,322 | 2,816,344 |
| SVG path elements | 20,529 | 27 |
| Leaf labels | 3,422 | 3,422 |

In that first increment, path coordinates occupy 1,492,596 bytes (53%) and the 3,422
text groups occupy 1,305,053 bytes (46%). Batching removes path-element overhead,
but retains the branch geometry. Matplotlib repeats the font-family list, style
and position/rotation attributes for each label. Shared text styles and lower
coordinate precision were not yet applied in that measurement; see the next
section for the subsequent optimization.
Gzip compresses this example to 769,175 bytes without changing its contents.

The 1,100-leaf comb tree previously raised `RecursionError` during branch
coloring. The new renderer exported all leaves with seven SVG paths in about
1.9 seconds without changing Python's recursion limit. These are host-specific
single-process measurements, not production training performance promises.

Regression checks cover deep trees, singleton levels, duplicate names, linkage
geometry and cluster identity, label-cache reuse and outages, SVG text retention,
figure cleanup after failure, and local saving without TensorBoard/W&B.

## Bounded simplification and compact SVG export

The next increment replaces short circular arcs with chords only when the
sagitta is at most 0.02 points in figure space (using the full figure width as a
conservative scale bound). This is at most 0.027 CSS pixels at native size, or
0.11 pixels at 4x zoom. Larger arcs retain cubic curves, now at most 45 degrees
per segment; their radial approximation error is below 0.006 points at the
maximum figure size. Branch endpoints, leaf order, cluster assignments and all
labels remain intact. Narrow taxonomy bands consequently become quadrilaterals.
Backend path simplification is disabled on the branch paths to keep it from
adding a second uncontrolled approximation. Magnification far beyond the design
scale can reveal these approximations; this is not exact analytic geometry.

`save_dendrogram_svg` rounds path coordinates to 0.001 points (at most 0.00071
points Euclidean rounding error), deduplicates identical text styles, and removes
empty per-label group wrappers. Label strings, fonts, colors, positions and
rotations remain unchanged. Ordinary returned Matplotlib figures benefit from
simpler geometry; training's SVG export additionally applies the markup changes.

Like-for-like local CPU results for the same 3,422-class input, relative to the
iterative renderer introduced in `909cb09`:

| Measurement | Before simplification | After simplification |
| --- | ---: | ---: |
| Geometry vertices, including curve controls and bands | 63,719 | 28,980 |
| Figure construction, including linkage | 2.70 s | 1.03 s |
| Construction and SVG export | 6.48 s | 3.22 s |
| SVG bytes | 2,816,344 | 1,041,576 |
| SVG paths / leaf labels | 27 / 3,422 | 27 / 3,422 |
| Chromium decode + raster, 1,200 px square | 329 ms | 252 ms |
| Chromium decode + raster, 4,800 px square | 421 ms | 357 ms |

Browser values are medians of five interleaved before/after measurements, each
in a fresh browser context, using headless Chromium 151. They include SVG image
decode, canvas drawing and forced pixel readback, excluding process startup,
network transfer and PNG encoding. They do not measure interactive panning or
production dashboard integration. Rasterization alone improved more modestly
(189 to 177 ms at 1,200 px; 311 to 285 ms at 4,800 px); labels still cost work.
These timings are host-specific, not GPU training guarantees.

Reproduce the figure benchmark with the prior layout and the current exporter:

```bash
git show 909cb09:mini_trainer/visualization/_dendrogram_layout.py > /tmp/layout-before.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --layout-file /tmp/layout-before.py --output /tmp/svg-before
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --output /tmp/svg-after
```

Browser verification uses a disposable environment, without adding dependencies
to training. The benchmark saves raster images and all timing samples:

```bash
uv venv /tmp/svg-browser-env
uv pip install --python /tmp/svg-browser-env/bin/python playwright
/tmp/svg-browser-env/bin/python -m playwright install chromium
/tmp/svg-browser-env/bin/python dev/benchmarks/reporting/svg_browser.py \
    /tmp/svg-before/dendrogram.svg /tmp/svg-after/dendrogram.svg \
    --output /tmp/svg-browser-results
```

Regression tests check chord deviation, cubic radial error, unchanged endpoints,
text/style/transform retention, and export idempotence, alongside the existing
large/deep-tree and logging checks. Pixel comparisons complement those geometric
checks; they cannot establish visibility at arbitrary zoom levels.
