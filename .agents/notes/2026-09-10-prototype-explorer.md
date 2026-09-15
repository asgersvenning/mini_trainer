# Prototype explorer handoff

Status: completed
Updated: 2026-09-10
Scope: initial prototype-space exploration, log-domain diagnostics, GBIF image labels
Related: `42cef5c`, `d4663e7`; [reproduction guide](../../dev/prototype_space/README.md)

## Context and decision

The feature is isolated on `feature/prototype-space` in
`/home/asger/mini_trainer/.worktrees/prototype-space`. The main `quant` checkout
received only the worktree workflow/rules; its pre-existing roadmap edits were
verified unchanged. No subagents were started. Shared `.venv` was used read-only.

The real epoch-4 checkpoint supplies the baseline geometry. Existing effective
weight handling, cosine-to-z transform and Ward distances are retained. Added
float32 log-tail computation and adjustable colour clipping distinguish numerical
saturation from actual prototype identity. Public GBIF examples visually label
classes at their learned target directions; measured image embedding proximity
would be a separate selection criterion. Taxon names never replace checkpoint IDs.

## Evidence and limits

- Final source revision: `d4663e7` (includes numerical commit `42cef5c`).
- Full `bash dev/check.sh all`: 655 passed, 162 skipped, 1 expected failure;
  exit 0 in 598.56 seconds. Ruff, formatting and both import contracts passed.
  Used absolute worktree `PYTHONPATH`, `MPLCONFIGDIR=/tmp/mini-trainer-prototype-mpl`,
  `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`, and
  `PYTEST_ADDOPTS='-o faulthandler_timeout=120'`. CUDA was hidden. This does not
  establish GPU execution or performance. No package installation was performed.
- A preceding sandboxed full run was interrupted after waiting on loader workers:
  340 passed, 13 skipped. The successful rerun allowed local multiprocessing
  sockets. Do not confuse that interrupted result with full-suite success.
- Browser helper: 25 interaction checks, zero JavaScript errors, including
  colour controls, baseline comparisons, tree/search/neighbour navigation,
  synthetic cases, photo cycling and photo navigation. Photo controls used
  deterministic fixtures; a separate live check rendered actual GBIF images for
  class 1837646 and its neighbours with credits.
- Real payload: 12,632 x 1,280 prototypes; 249,324 unordered baseline zero-distance
  pairs, all positive under the independent float64 log-CDF reference; all
  79,777,396 off-diagonal float32 log-tail pairs finite. Three labelled synthetic
  probes accompany the checkpoint; their smaller class counts are not matched
  packing controls.
- Checkpoint SHA256:
  `42d50f56cb0c6e3ee17ae64fbbcc4eb2ae24335ffe7080197c479c130fff5a81`.
- Ignored, machine-local artifacts: `tmp/prototype-report/`,
  `tmp/prototype-browser-check/`, `tmp/prototype-live-browser-check/`.
  The generated HTML was checked against current templates and stored numerical
  JSON. The JSON records the numerical-generation script hash; later UI-only
  refreshes did not recompute or relabel numerical provenance.
- Preview was healthy at `http://localhost:8765/explorer.html` on completion,
  served with `python -m dev.prototype_space.serve`; it is not a persistent service.
  GBIF metadata/thumbnails are cached outside the served directory. Images require
  optional network access; the numerical report is self-contained.
- Evaluation logging changes apply to this feature branch, not to an already
  running training process. Existing Ward clustering remains the baseline;
  alternate stable clustering and measured sample occupancy are future work.

## Next actions

None required for this initial exploration. Gather feedback on the runnable views
before choosing further directions listed in the reproduction guide. Keep source
integration into `quant` separate from this completed feature-branch handoff.
