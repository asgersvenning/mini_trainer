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
- Saved views (browser storage, local-server persistence, JSON import/export),
  completed-analysis caching, stage/elapsed progress and cancellable replacement jobs.
- Coalesced pan/zoom drawing and shared bounded thumbnail metadata lookups;
  see the [reuse and responsiveness qualification](../dev/prototype_space/reuse-performance.md).

Evidence: real checkpoint diagnostics match the preserved baseline; 84 browser
checks, 12 affected Python tests, motion checks, static checks, and installed-wheel
CLI/file-picker checks passed. Current extraction supports float32 linear
`Classifier` and `HierarchicalClassifier` heads; dense analysis needs several GB
for the 12,632-class case. These checks do not establish full-suite/GPU coverage.

The latest viewer increment passed 150 browser assertions; the preceding
names/probability increment also passed
11 affected Python tests and log-tail reference and motion checks. Static checks
passed for the latest changes. These are scoped results, not a new full-suite audit.

## Current browser delivery

The subsequent `feature/prototype-browser-inference` increment (`1eb2a32`)
includes master `1d2dae0` and implements browser WASM inference, exported
preclassification embeddings and browser-side insertion into the fixed t-SNE map.
See the [browser guide](prototype-browser.md) and
[distribution plan](production-release-integration.md).

The final production checkpoint replaces epoch 26 as the browser reference:
SHA-256 `174b9214bfea2df69e4f5c5d16afd841fec961db4274f3e6bf474cef9cab5e8a`.
It retains 12,632 species and 1,280-dimensional normalized prototypes. Earlier
checkpoint evidence above remains historical. Browser inference was checked on
Chromium/WASM with identical tensors and one real image; this is not broad browser
or accuracy qualification. Global diagnostic construction remains offline
preparation. The current portable-UX increment replaces the name/photo Python
transport with a shared browser client and optional packaged names; see its
[qualification evidence](../dev/prototype_space/portable-qualification.md).
Human layout review and integration remain pending.

## Ordered next work

| Order / status | Increment | Done when |
| --- | --- | --- |
| Delivered | Reliable repeated exploration and first browser inference/query placement. | Existing evidence is retained; limitations remain explicit in the browser guide. |
| **Implemented · Review pending** | **Coherent UI and portable GBIF integration.** One global class-ID setting, shared browser-side names/photos and predicted-species thumbnails. | Static hosting needs no Python name/photo service; all consumers obey one namespace setting; attributed prediction thumbnails and desktop/mobile navigation pass the implementation plan's acceptance checks. |
| Planned | Local packing: angular cap counts, neighbour-radius curves, mutual neighbours and taxonomy composition. | Radius/rank selection links the same original-space class set across views; calculations match exact references. |
| Planned | Single-anchor view, then bearings and two-anchor comparison. | Angular radii match direct calculation; degeneracies and non-anchor distortion are disclosed. |
| Later | Client-side global diagnostics and offline distribution. | Real bundles generate global views locally within measured resource bounds; packaged data and optional GBIF networking have explicit offline behavior. |
| Later · Compatibility | Alternate-browser qualification, starting with Firefox and then Safari. Separate from the current UI/GBIF goal. | Verify model loading, preprocessing/output parity, one-image inference, fixed-map t-SNE insertion, GBIF names/photos, saved state and core interactions on named browser versions; document capability limits. WebGPU qualification remains a separate decision. |
| Later · Research | Great-sphere slices, alternative layouts, checkpoint trajectories and empirical calibration. | Each proposal earns implementation through a bounded experiment and original-space validation. |

The [next-increment implementation plan](prototype-explorer-implementation.md)
contains the executable goal, code boundaries, state migration and acceptance
matrix. The [standalone plan](prototype-browser-implementation.md) retains broader
inference/distribution contracts, not a competing priority order.

UI cleanup is now an explicit priority, extending earlier focused-layout work in
response to the combined viewer's usability problems. Keep expert controls
accessible and preserve the full-pane map and human review of previews.
Take one bounded increment at a time. Integration into master requires review and
combined validation; do not overwrite the published production artifact during
implementation.

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
