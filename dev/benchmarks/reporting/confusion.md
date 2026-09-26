# Whole-matrix confusion diagnostics

Confusion figures keep every model output index on both axes, including
prediction-only and unobserved classes. Rows are truth and columns predictions;
resolve names from saved classifier metadata. Class order stays fixed across epochs.
[logging/confusion.py](../../../mini_trainer/logging/confusion.py) owns distributed
reduction, previews and numerical artifacts; generic heatmaps remain in
`visualization/plot.py`.

## Dashboard images and local detail

TensorBoard/W&B receive an RGB whole-matrix preview with at most 1,536 cells per
side, plus a legend and caption. For N classes, block width is `ceil(N / 1536)`.
Each preview cell is the arithmetic mean of row-normalized probabilities in its
block; edge blocks use their actual counts. This is not a regrouped or renormalized
confusion matrix. The `/overview_mean` tag and metadata identify the reduction.
Small previews enlarge with nearest-neighbor sampling. Means may obscure isolated
errors; full-resolution images and numerical data retain them.

Both palettes use a fixed logarithmic `[1e-6, 1]` probability scale across epochs:
soft images have 128 positive levels; hard images use 256-color magma. Black means
exact zero, magenta means invalid/negative, and positive values below the floor
use the lowest positive color. Soft PNGs use an indexed palette without dithering.
Rendering does not change raw values; a separate colorbar preserves the matrix palette.

Each epoch writes beneath `model/logs/figures/epoch-NNNN/Confusion_matrix_lvlL/`
and `Soft_confusion_matrix_lvlL/`:

| Artifact | Meaning |
| --- | --- |
| `matrix.png` | One pixel per original class pair, with no class-count pooling limit |
| `counts.npz` (hard) | Exact integer COO `rows`, `columns`, `counts` and `shape`; omitted entries are zero |
| `probability_sums.npy` (soft) | Accumulated float32 sums before normalization, clipping or palette mapping |
| `row_support.npy` | Hard truth counts or soft row probability mass used for normalization; empty rows stay zero |
| `metadata.json` | Class indices, orientation, normalization, palette, invalid-cell count and preview block shape |
| `colorbar.png` | Matching legend, also included in dashboard previews |

Soft sums are uncompressed to avoid compression cost during logging: about 100 MB
at 5,000 classes or 400 MB at 10,000, per saved epoch/level. Inspect with NumPy
`mmap_mode='r'`. Accumulation/storage remain quadratic, and full PNGs are large when
decoded. Bounded previews reduce dashboard traffic; production storage throughput
still needs qualification.

## Distributed reporting

All ranks participate, including ranks without validation samples. Shape descriptors
coordinate bounded reductions onto rank zero; NCCL uses temporary CUDA chunks.
Only rank zero renders, writes and forwards images. Soft buffers remain unchanged
across repeated reports. Counts include validation sampler padding; no sample-ID
deduplication is performed.

[CPU two-process tests](../../../tests/logging/test_confusion.py) cover an empty
rank, prediction-only classes and repeated collection. CUDA/NCCL and live dashboard
uploads require target-job checks.

## Reproduce rendering measurements

The seeded workload has dense soft probabilities, 20 diagonal blocks and a diagonal
signal. Timing includes normalization, rendering and file writes; the new path also
saves exact data and full-resolution PNGs absent from the old path. RSS includes
imports and inputs. Use a fresh process and output directory for each case:

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

Recorded local CPU results (decimal MB for file sizes):

| Classes | Old/new seconds | Old/new peak RSS MiB | Old/new dashboard PNG MB |
| --- | ---: | ---: | ---: |
| 5,000 | 9.58 / 2.77 | 2,994 / 907 | 47.18 / 1.34 |
| 10,000 | 12.12 / 6.62 | 3,236 / 1,258 | 33.00 / 1.11 |

New full-resolution soft PNGs occupy 15.78 and 62.67 MB respectively. The old
10,000-class dashboard was pooled to 5,000 cells per side; new local PNGs preserve
all classes. These synthetic host-specific measurements exclude distributed
reduction and remote upload, and do not establish training peak memory.

For PNG browser measurements, use the disposable environment in
[dendrogram.md](dendrogram.md):

```bash
/tmp/svg-browser-env/bin/python dev/benchmarks/reporting/svg_browser.py \
    /tmp/confusion-before/dashboard.png /tmp/confusion-after/dashboard.png \
    --repeats 3 --output /tmp/confusion-browser
```

At 5,000 classes, headless Chromium's median decode plus forced raster time was
1,132 / 52.6 ms (old/new) at 1,200 pixels and 1,242 / 170 ms at 4,800 pixels.
These were three interleaved trials per image/size in fresh contexts, comparing the
old dashboard with the new captioned preview. Startup, network and PNG encoding
are excluded. This does not measure full-resolution PNG or live dashboard performance.
