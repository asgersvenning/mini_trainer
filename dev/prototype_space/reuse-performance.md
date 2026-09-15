# Reuse and responsiveness qualification

2026-09-11, epoch-26 checkpoint recorded in the explorer roadmap. Baseline viewer
assets: `3907f30`. Host: Intel i7-12800H, WSL2 Linux 5.15.167.4, Python 3.13.7,
torch 2.12.0+cu130, NumPy 2.4.6, headless Chromium 151. CPU analysis used four
threads; no GPU-correctness claim is made. These are local measurements, not
universal performance guarantees.

## Preparation and numerical preservation

| Run | Wall time | Peak RSS |
| --- | ---: | ---: |
| Uncached baseline | 82.02 s | 5,424,156 KiB |
| First cache-populating analysis | 88.59 s | 5,428,248 KiB |
| Cache reopen | 3.98 s | 1,080,716 KiB |

The independent cold runs have identical stored distance/log-tail arrays, class
ordering, tree, neighbourhoods, PCA and t-SNE coordinates. Differences are source
provenance and the t-SNE KL diagnostic scalar (1.9826414585 versus 1.9826412201).
Cached/reopened numerical JSON is byte-identical, SHA-256
`a474f491471b16ccb3197831a23a1dff7c17726eb8a0cc20644197c29d860b7d`.
The cache deliberately retains the original analysis provenance.

Reproduce with `/usr/bin/time -v`, using distinct output directories for the cold
and warm runs, the same cache directory and these environment settings:

```bash
export PYTHONPATH="$PWD"
export MPLCONFIGDIR=/tmp/mini-trainer-prototype-mpl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/usr/bin/time -v .venv/bin/python -m mini_trainer.visualization.prototype_space \
  /absolute/path/to/best_global-lepi-production-w32-1_epoch26.pt \
  --export --cache-dir tmp/qualification-cache --output tmp/qualification-cold
# Repeat with --output tmp/qualification-warm; use --no-cache for an uncached run.
```

## Interaction and thumbnail arrivals

`measure_interaction.mjs` runs the same real-coordinate workload against a frozen
baseline report and the updated report. It emits raw samples and browser identity
in `performance.json`. The pan workload sends 30 pointer moves per frame over 20
frames at 1440 × 1100. Frame times include the subsequent animation-frame callback;
handler times alone are insufficient to establish responsiveness.

The loading workload uses a fixed 500 ms metadata delay, a tiny local image, and
three navigation updates 250 ms apart. It isolates wasted metadata requests; it
does not measure GBIF network throughput or large-image decode time.

| Measure | Before | After |
| --- | ---: | ---: |
| Frame median | 16.9 ms | 17.3 ms |
| Frame p95 | 83.7 ms | 18.1 ms |
| First visible thumbnail | 1.484 s | 0.714 s |
| All 31 visible thumbnails | 5.064 s | 4.253 s |
| Metadata requests / navigation aborts | 43 / 12 | 31 / 0 |

The improvement is in frame-time outliers and avoiding repeated lookups during
navigation; the median frame interval did not improve. Started metadata requests
are shared across generations/cards, with at most four active. Queued obsolete
requests are discarded. Image loading still follows the visible candidate set,
and existing appearance, motion, attribution and culling checks remain applicable.
Local matrix recolouring now transforms only its selected submatrix rather than
all classes' local matrices on every refresh.

```bash
CHROMIUM_BIN=/path/to/chrome node dev/prototype_space/measure_interaction.mjs \
  file:///absolute/path/to/explorer.html tmp/interaction-measurement
CHROMIUM_BIN=/path/to/chrome node dev/prototype_space/check_session_browser.mjs \
  http://localhost:PORT/explorer.html tmp/session-check
```

Serve the epoch-26 model with the launcher for the session check. That check saves
a view through the server, clears browser storage, suppresses page-hide saving,
reloads and verifies restoration. The Python session test separately verifies
reuse across new workspace instances with different tokens.

## Validation scope

Targeted launcher, extraction and photo tests passed (15 tests, including a targeted real-process cancellation check), plus 158 explorer
browser assertions, the real-checkpoint session check, motion/log-tail checks,
static checks and wheel validation. The broad suite was stopped at the user's
request; it is not reported as passed. Later lifecycle-only changes are checked
with launcher tests, not unrelated training suites.

Timing artifacts live in ignored `tmp/reuse-{baseline,cold,warm}-epoch26` and
`tmp/interaction-{before,after}-complete`. Re-run on the same host with unchanged
workload before making further speed claims. Live GBIF latency remains variable.
