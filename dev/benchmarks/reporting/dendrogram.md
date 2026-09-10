# Dendrogram scaling and artifact validation

The renderer uses an iterative SciPy-linkage layout, retaining the original
right-child-first leaf order, distance-based radial positions and flat-cluster
assignments. It does not construct recursive BioPython/pyCirclize tree objects.
Same-color branches share paths; adjacent same-color taxonomy bands are merged.
Circular arcs use cubic Bezier segments instead of dense sampled polylines.
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

The 1,100-leaf comb tree previously raised `RecursionError` during branch
coloring. The new renderer exported all leaves with seven SVG paths in about
1.9 seconds without changing Python's recursion limit. These are host-specific
single-process measurements, not production training performance promises.

Regression checks cover deep trees, singleton levels, duplicate names, linkage
geometry and cluster identity, label-cache reuse and outages, SVG text retention,
figure cleanup after failure, and local saving without TensorBoard/W&B.
