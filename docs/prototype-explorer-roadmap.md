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

The latest viewer increment passed 150 browser assertions; the preceding
names/probability increment also passed
11 affected Python tests and log-tail reference and motion checks. Static checks
passed for the latest changes. These are scoped results, not a new full-suite audit.

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
| 1 · Next | Reliable repeated exploration: saved viewer state, versioned analysis cache, preparation progress/cancellation. | Compatible reopening restores state and skips analysis; cancelled/stale jobs cannot replace valid results; real-case load and memory costs are recorded. |
| 2 · Qualification | Real-model browser inference: standalone milestones 1–3. | Exact preprocessing/embedding/output contracts are identified; epoch-26 predictions and embeddings pass real-image WASM browser parity with latency/memory evidence. |
| 3 · Planned | Local packing: angular cap counts, neighbour-radius curves, mutual neighbours and taxonomy composition. | Radius/rank selection links the same original-space class set across views; calculations match exact references. |
| 4 · Planned | Single-anchor view, then bearings and two-anchor comparison. | Angular radii match direct calculation; degeneracies and non-anchor distortion are disclosed; existing image controls work. |
| 5 · Depends on 2 | Query inference UI and embedding placement: standalone milestones 4–5. | Actual predictions, query geometry and fixed-transform PCA/anchor placement are linked; nonlinear insertion is separately qualified. |
| 6 · Depends on 2 and 5 | Client-side global diagnostics and standalone distribution: milestones 6–7. | The real bundle generates global views locally and works offline without Python/PyTorch/CUDA; numerical and resource gates pass. |
| Later · Research | Great-sphere slices, alternative layouts, checkpoint trajectories and empirical calibration. | Each proposal earns implementation through a bounded experiment and original-space validation. |

The [next-increment implementation plan](prototype-explorer-implementation.md)
specifies code boundaries, deliverables, tests and the recommended next goal.
This table is the unified execution order; the standalone plan provides detailed
acceptance gates, not a second competing priority list. Geometry increments 3–4
can proceed if browser qualification needs an external input, with the blocker
recorded. Do not treat such progress as completion of browser qualification.

Take one bounded increment at a time. Preserve the full-pane UX as the baseline;
further presentation changes should support a diagnostic or demonstrated issue.
Integration into the target branch requires review and combined validation.

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
