# Whole-matrix confusion diagnostics

Confusion figures retain every model output index on both axes, including
prediction-only and unobserved classes, in the same order across epochs. Rows
are true classes and columns are predicted classes. Class names can be resolved
from the saved model/config classifier metadata. No taxonomy regrouping or
selected-pair view replaces the full matrix.

The specialized reporting logic lives in `mini_trainer/logging/confusion.py`:
rank reduction, whole-matrix overviews, palette encoding and numerical artifacts.
The existing `visualization/plot.py` retains generic heatmap/colorbar rendering;
its byte-color mapping now uses bounded chunks with regression checks against
the previous RGB pixels. No new runtime dependencies or background service are
introduced.

## Dashboard images and local detail

TensorBoard/W&B receive RGB previews with a matrix side of at most 1,536 pixels,
plus a 200-pixel legend and a 24-pixel caption. Each preview covers the entire
matrix. For N classes its block width is `ceil(N / 1536)`; a preview cell is the
arithmetic mean of the row-normalized probabilities in that block. Partial edge
blocks use their actual cell counts. This is an overview of probabilities, not
an aggregated or renormalized confusion matrix. The caption, `/overview_mean`
tag and metadata explicitly identify the reduction. Small images enlarge with
nearest-neighbor sampling; there are no cell borders or spatial smoothing.
Isolated errors may become less visible in the mean overview; native-resolution
images and numerical matrices retain them.

Soft images use 128 positive logarithmic color levels over probabilities
`[1e-6, 1]`, plus dedicated black zero and magenta invalid/negative colors. Values
below the positive floor use the lowest positive color, never the zero color.
Hard images retain the 256-color magma mapping on the same fixed scale. Scales
remain comparable across epochs. Posterization changes displayed colors only;
raw values are retained. The full soft PNG is indexed (palette mode), without
dithering. The matching discrete colorbar is saved separately, so its text does
not expand the matrix palette. Dashboard previews include the legend.

Each epoch saves the following under
`model/logs/figures/epoch-NNNN/Confusion_matrix_lvlL/` and
`Soft_confusion_matrix_lvlL/`:

- `matrix.png`: one pixel per original class pair, without the old 5,000-class
  maximum-pooling limit. The separate colorbar does not change matrix dimensions.
- `counts.npz` for hard matrices: exact integer COO arrays `rows`, `columns`,
  `counts`, plus `shape`. Absent entries are zero.
- `probability_sums.npy` for soft matrices: the original accumulated float32
  probability sums, before normalization, clipping or posterization.
- `row_support.npy`: hard true-label counts or the soft row probability mass
  used for normalization. Empty rows remain zero.
- `metadata.json`: original class indices, orientation, normalization, palette,
  invalid-cell count and preview block shape; and `colorbar.png`.

The soft NPY is deliberately uncompressed to avoid spending logging time
compressing high-entropy floats. It is about 100 MB for 5,000 classes and 400 MB
for 10,000 classes per saved epoch/level. Local storage therefore increases even
though dashboard traffic decreases. NumPy can inspect it with `mmap_mode='r'`.
Storage throughput on the production filesystem needs qualification. Accumulation
and exact matrix storage still scale quadratically with class count. Full-size
PNGs remain large decoded images; the bounded preview is what protects dashboard
responsiveness. This increment does not add a tile server or custom viewer.

## Distributed reporting

Every rank participates in confusion collection. Small shape descriptors allow
even a rank with no validation samples to participate; count and probability
matrices are summed onto rank zero in bounded chunks. NCCL uses temporary CUDA
chunks rather than placing the entire matrix on the GPU. Only rank zero renders,
writes artifacts and forwards preview images. Soft accumulation buffers are not
mutated, so repeated reporting does not double counts. Counts include validation
sampler padding, matching the samples actually reported; sample-ID deduplication
is outside this change.

CPU two-process tests cover an empty rank, prediction-only classes and repeated
collection. CUDA/NCCL and live TensorBoard/W&B uploads remain target-job checks.

## Reproduce rendering measurements

The benchmark uses seeded dense soft probabilities with 20 large diagonal blocks
and a diagonal signal. Timing includes normalization, image rendering and file
writes. The new path also writes the exact matrix and full-resolution PNG, which
the old path did not retain. RSS includes imports and the input matrix; it is not
training's peak memory. Run each case in a fresh process:

```bash
git show d521cef:mini_trainer/visualization/plot.py > /tmp/heatmap-before.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.confusion --classes 5000 \
    --baseline /tmp/heatmap-before.py --output /tmp/confusion-before
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
    .venv/bin/python -m dev.benchmarks.reporting.confusion --classes 5000 \
    --output /tmp/confusion-after
# Repeat with --classes 10000 and new output directories.
```

Representative local CPU results (decimal MB for file sizes):

| Classes | Old/new seconds | Old/new peak RSS MiB | Old/new dashboard PNG MB |
| --- | ---: | ---: | ---: |
| 5,000 | 9.58 / 2.77 | 2,994 / 907 | 47.18 / 1.34 |
| 10,000 | 12.12 / 6.62 | 3,236 / 1,258 | 33.00 / 1.11 |

The new full-resolution soft PNGs are 15.78 MB and 62.67 MB respectively. The old
10,000-class dashboard was already pooled to 5,000 cells per side; the new local
PNG preserves all 10,000. These are synthetic, host-specific measurements,
excluding distributed reduction and remote dashboard upload.

The existing browser benchmark also accepts PNGs. With the disposable browser
environment described in [dendrogram.md](dendrogram.md), run:

```bash
/tmp/svg-browser-env/bin/python dev/benchmarks/reporting/svg_browser.py \
    /tmp/confusion-before/dashboard.png /tmp/confusion-after/dashboard.png \
    --output /tmp/confusion-browser
```

For the final 5,000-class dashboard PNGs, headless Chromium's median decode plus
forced raster time (three interleaved runs per image/size, fresh contexts) was
1,132 ms before versus 52.6 ms after at a 1,200-pixel display, and 1,242 ms versus
170 ms at 4,800 pixels. This compares the old large dashboard image with the new
captioned whole-matrix preview; it does not imply that the full-resolution local
PNG is cheap to display. Timings exclude Python/browser startup, network transfer
and PNG encoding. Live TensorBoard/W&B end-to-end performance was not measured.
