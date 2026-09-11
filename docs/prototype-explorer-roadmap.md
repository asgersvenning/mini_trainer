# Prototype explorer roadmap

Updated 2026-09-11. This is the maintained development order for the explorer;
research detail belongs in the [geometry review](hyperspherical-visualization.md),
and operation in the [usage guide](prototype-explorer.md). Later stages are
proposals, not active implementation commitments.

## Delivered

Implemented on `feature/prototype-space` (not yet integrated into the main
development branch; reusable-package baseline `4763041`):

- Reusable package, `mt_explore`, browser weight selection, and portable export.
- Baseline distance matrices and Ward tree; direct log-tail diagnostics and
  configurable display clipping; linked class and neighbourhood inspection.
- PCA and angular t-SNE, with measured neighbourhood retention.
- Focused feature layouts with related panels, responsive projection aspect ratio,
  and retained class/plane/subtree state; verified at landscape, portrait and mobile sizes.
- GBIF photo labels, configurable footprints/culling, continuous arrival motion,
  pan/zoom retention, and quick navigation exit fades.
- Optional asynchronous taxon names with visible-label priority and offline aliases;
  z/chance-alignment probability views with explicit reference semantics.
- Display-density projection rendering and viewport-height focused panels; map
  details and settings overlays prevent hover/status text from resizing the canvas.
  Projection focus fills the viewport with collapsible floating navigation and controls.

Evidence: real checkpoint diagnostics match the preserved baseline; 84 browser
checks, 12 affected Python tests, motion checks, static checks, and installed-wheel
CLI/file-picker checks passed. Current extraction supports float32 linear
`Classifier` and `HierarchicalClassifier` heads; dense analysis needs several GB
for the 12,632-class case. These checks do not establish full-suite/GPU coverage.

The names/probability/layout increment additionally passed 133 browser assertions,
11 affected Python tests, log-tail reference and motion checks, and static checks.

Current real-data reference: `tmp/best_global-lepi-production-w32-1_epoch26.pt`
(training progress reported as 26/30 epochs), SHA-256
`abb9f66d95fe867bd31847cb33782a353b1d3faf4c79c4b10d93037f33fe61f4`.
Verified extraction: 12,632 × 1,280 float32 normalized prototypes, zero biases,
and unchanged class ordering/hierarchy relative to epoch 4. Retain epoch 4 for
longitudinal comparisons. Browser inference parity is still an open gate.

## Standalone client direction

The [goal-ready implementation plan](prototype-browser-implementation.md) defines
the bundle contract, ordered milestones, and end-to-end completion gates.

Target: browser inference and embedding inspection without a Python/PyTorch/CUDA
runtime. Export preparation may use mini_trainer once; the distributed viewer
loads an ONNX bundle, class/prototype metadata, and executable preprocessing.
The existing `.pt` picker remains a Python workflow, not this standalone path.

Next feasibility gate: export actual prediction outputs plus the precise
preclassification embedding; verify preprocessing, embeddings and scores against
real-image reference results in ONNX Runtime Web (WASM first, then WebGPU).
CPU ONNX parity alone does not establish browser compatibility or performance.

Then add image selection, original-space neighbours/margins and query overlays.
Save PCA means/bases for new points; define and validate placement into the fixed
t-SNE map separately. Anchor views can expose direct query-to-prototype angles.
Geometry explains relationships; observed probabilities retain actual head semantics.

Deliver a static/installable viewer with locally selected models and images.
Precomputed global diagnostics are an initial delivery option; moving their
construction client-side is a separate milestone requiring blocked computation,
workers, memory/latency measurements and baseline numerical parity. Keep direct
log-domain semantics when porting diagnostics. GBIF networking/offline photo
availability must be explicit and independent of local inference.

References: [current export contract](onnx.md),
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/),
[browser deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html).
This is planned work; no browser model inference is implemented or validated yet.

## Ordered next work

| Order / status | Increment | Done when |
| --- | --- | --- |
| 1 · Next | Reliable repeated exploration: progress/cancellation, analysis cache keyed by checkpoint and numerical configuration, saved viewer state. | Reopening avoids unnecessary analysis; cancellation leaves a usable session; loading, peak memory and frame motion are measured on the real case. |
| 2 · Planned | Local packing: angular cap counts, neighbour-radius distributions, mutual-neighbour links and taxonomy composition across scales. | Selected-class profiles and linked outliers agree with exact original-space calculations; unknown taxonomy remains explicit. |
| 3 · Planned | Anchor atlas, then two-anchor comparison: angular rings, selectable bearings and existing photo controls. | Radius matches the chosen baseline angular convention; bearing degeneracies and non-anchor distortion are disclosed and tested. |
| 4 · Research | Great-circle/great-sphere slices with winners, competing classes and margins. | Slice scores match direct high-dimensional evaluation; geometric regions are distinguished from the full classifier decision rule and global cell volumes. |
| 5 · Research | Compare further planar/spherical layouts; add a globe only if useful. | Near/far angular error, neighbour retention, seed stability, fit cost and display distortion justify a new option. |
| 6 · Later | Checkpoint trajectories, then held-out embedding overlays and observed confusion. | Comparisons align class IDs and distinguish rotation/layout changes from changes in relationships; empirical claims identify their sample data. |

Take one bounded increment at a time. Qualify standalone inference before committing
to its runtime design. The next geometry feature is local packing
profiles followed by the anchor atlas; reliability work supports both. Integration
into the target branch requires review and validation of the combined result.

## Rules for research and prioritization

- Observed effective weights and existing distance functions are the baseline.
  Synthetic cases test hypotheses and edge cases; they do not impose a packing prior.
- Preserve direct log-domain computation. Treat clipping as display configuration.
  Alternative clustering inputs require explicit comparison with the retained tree.
- Do not infer spherical density or volume from projected density or slice area.
- Profile before replacing rendering or introducing approximations. Validate blocked,
  sparse or sampled computations against exact results, disclosing approximation error.
- Broader head/dtype support needs an extraction contract and checkpoint parity tests.
  It is a compatibility backlog item, not a reason to guess unsupported weights.

Update this page with each completed increment or priority change: move delivered
work out of the queue, attach compact evidence, and revise the next step. Keep
experiment logs and detailed method comparisons in their existing linked documents.
