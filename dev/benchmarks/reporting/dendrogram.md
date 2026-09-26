# Dendrogram scaling and artifact validation

Training retains dendrogram SVGs under `model/logs/figures/epoch-NNNN/`, including
with the metrics-only logger. Rank zero writes local artifacts and forwards figures
to configured logging backends; failed exports close created figures. Labels remain
selectable text, so their appearance depends on the viewer's installed fonts.

The iterative SciPy-linkage renderer preserves right-child-first leaf order,
distance-based radii, cluster assignments and every label, including duplicate
display names. Singleton levels bypass linkage. Taxonomy colors follow stable
first-occurrence order; branches and adjacent taxonomy bands share paths by color.

Name resolution uses at most eight concurrent lookups and a bounded process-local
cache across epochs, falling back to original labels on service/cache errors.
Cold taxonomies still require network requests. The current GBIF callers use the
process-wide socket default; there is no lookup or batch deadline. Logs separate
label resolution, rendering and export. Model distances and Ward linkage remain
quadratic in class count; production-taxonomy capacity needs separate qualification.

## Bounded simplification and compact SVG export

Short arcs use chords with at most 0.02-point sagitta in figure space (0.027 CSS
pixels at native size, 0.11 at 4x zoom). Larger arcs use cubic segments of at most
45 degrees, with radial error below 0.006 points at the maximum figure size.
Endpoints remain exact; backend path simplification is disabled on branches.
These are bounded approximations, not guarantees at arbitrary magnification.

`save_dendrogram_svg` rounds path coordinates to 0.001 points (maximum Euclidean
rounding error 0.00071 points), shares text styles and removes empty label wrappers.
Text, fonts, colors, positions and rotations are preserved. Geometry simplification
also applies to ordinary Matplotlib figures; markup compaction requires this exporter.
See [_dendrogram_layout.py](../../../mini_trainer/visualization/_dendrogram_layout.py)
and [_svg.py](../../../mini_trainer/visualization/_svg.py) for the implementation.

## Recorded measurements

These are historical local CPU measurements on seeded 3,422-class inputs with
Python 3.13, Matplotlib 3.10.9, BioPython 1.87 and pyCirclize 1.10.1. Each comparison
includes linkage, layout and SVG export, excluding model distances and name lookup.
They are separate paired measurements, not production training guarantees.

| First comparison | Recursive renderer | Iterative renderer |
| --- | ---: | ---: |
| Layout/render, including linkage | 18.38 s | 2.78 s |
| Including SVG export | 24.53 s | 6.82 s |
| SVG bytes | 4,309,322 | 2,816,344 |
| SVG path elements | 20,529 | 27 |
| Leaf labels | 3,422 | 3,422 |

The 1,100-leaf comb tree previously raised `RecursionError`; the iterative renderer
exported all leaves in about 1.9 seconds with seven paths and no recursion-limit change.

| Compaction comparison, relative to `909cb09` | Before | After |
| --- | ---: | ---: |
| Geometry vertices, including curve controls and bands | 63,719 | 28,980 |
| Figure construction, including linkage | 2.70 s | 1.03 s |
| Construction and SVG export | 6.48 s | 3.22 s |
| SVG bytes | 2,816,344 | 1,041,576 |
| SVG paths / leaf labels | 27 / 3,422 | 27 / 3,422 |
| Chromium decode + raster, 1,200 px square | 329 ms | 252 ms |
| Chromium decode + raster, 4,800 px square | 421 ms | 357 ms |

Browser values are medians of five interleaved trials per image/size in fresh
headless Chromium 151 contexts: decode, canvas draw and forced pixel readback.
Rasterization alone changed from 189 to 177 ms at 1,200 px and 311 to 285 ms at
4,800 px; labels still cost work. Startup, network and PNG encoding are excluded;
interactive panning and live dashboard integration were not measured.

## Reproduce locally

Use the existing environment with plotting dependencies and a new output directory
per invocation. Both baseline paths export SVG text. Run each case in a fresh process:

```bash
git show 863a85c:mini_trainer/visualization/dendrogram.py > /tmp/dendrogram-before.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --module-file /tmp/dendrogram-before.py --output /tmp/dendrogram-baseline
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 1100 --deep \
    --output /tmp/dendrogram-deep

git show 909cb09:mini_trainer/visualization/_dendrogram_layout.py > /tmp/layout-before.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --layout-file /tmp/layout-before.py --output /tmp/svg-before
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.dendrogram --classes 3422 \
    --output /tmp/svg-after
```

The current command measures the current checkout; reproducing an intermediate
historical result also requires its renderer revision. Browser verification uses
a disposable environment and retains raster images and all samples:

```bash
uv venv /tmp/svg-browser-env
uv pip install --python /tmp/svg-browser-env/bin/python playwright
/tmp/svg-browser-env/bin/python -m playwright install chromium
/tmp/svg-browser-env/bin/python dev/benchmarks/reporting/svg_browser.py \
    /tmp/svg-before/dendrogram.svg /tmp/svg-after/dendrogram.svg \
    --output /tmp/svg-browser-results
```

[Regression tests](../../../tests/utils/test_dendrogram.py) cover geometry/error
bounds, deep and singleton trees, duplicate labels, cache behavior, SVG text/styles,
export idempotence and figure cleanup. Pixel comparisons complement geometric checks;
they cannot establish visibility at arbitrary zoom levels.
