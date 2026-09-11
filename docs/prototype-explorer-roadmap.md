# Prototype explorer roadmap

Updated 2026-09-11. This is the maintained development order for the explorer;
research detail belongs in the [geometry review](hyperspherical-visualization.md),
and operation in the [usage guide](prototype-explorer.md). Later stages are
proposals, not active implementation commitments.

## Delivered

On `feature/prototype-space`, through `4763041` (not yet integrated into the
main development branch):

- Reusable package, `mt_explore`, browser weight selection, and portable export.
- Baseline distance matrices and Ward tree; direct log-tail diagnostics and
  configurable display clipping; linked class and neighbourhood inspection.
- PCA and angular t-SNE, with measured neighbourhood retention.
- GBIF photo labels, configurable footprints/culling, continuous arrival motion,
  pan/zoom retention, and quick navigation exit fades.

Evidence: real checkpoint diagnostics match the preserved baseline; 84 browser
checks, 12 affected Python tests, motion checks, static checks, and installed-wheel
CLI/file-picker checks passed. Current extraction supports float32 linear
`Classifier` and `HierarchicalClassifier` heads; dense analysis needs several GB
for the 12,632-class case. These checks do not establish full-suite/GPU coverage.

## Ordered next work

| Order / status | Increment | Done when |
| --- | --- | --- |
| 1 · Next | Reliable repeated exploration: progress/cancellation, analysis cache keyed by checkpoint and numerical configuration, saved viewer state. | Reopening avoids unnecessary analysis; cancellation leaves a usable session; loading, peak memory and frame motion are measured on the real case. |
| 2 · Planned | Local packing: angular cap counts, neighbour-radius distributions, mutual-neighbour links and taxonomy composition across scales. | Selected-class profiles and linked outliers agree with exact original-space calculations; unknown taxonomy remains explicit. |
| 3 · Planned | Anchor atlas, then two-anchor comparison: angular rings, selectable bearings and existing photo controls. | Radius matches the chosen baseline angular convention; bearing degeneracies and non-anchor distortion are disclosed and tested. |
| 4 · Research | Great-circle/great-sphere slices with winners, competing classes and margins. | Slice scores match direct high-dimensional evaluation; geometric regions are distinguished from the full classifier decision rule and global cell volumes. |
| 5 · Research | Compare further planar/spherical layouts; add a globe only if useful. | Near/far angular error, neighbour retention, seed stability, fit cost and display distortion justify a new option. |
| 6 · Later | Checkpoint trajectories, then held-out embedding overlays and observed confusion. | Comparisons align class IDs and distinguish rotation/layout changes from changes in relationships; empirical claims identify their sample data. |

Take one bounded increment at a time. The next scientific feature is local packing
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
