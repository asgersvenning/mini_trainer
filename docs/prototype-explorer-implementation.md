# Prototype explorer: next implementation increments

Status: proposed execution plan, ready for bounded goal setting. No goal is active
and no implementation below is claimed complete. The [roadmap](prototype-explorer-roadmap.md)
owns priorities; the [standalone plan](prototype-browser-implementation.md) owns
the complete browser inference and distribution contract.

## Starting point

Baseline: `feature/prototype-space`, through `05aafee`. Preserve the full-pane map,
floating controls, names, score semantics, thumbnail continuity and existing
Python entry points. Continue in `/home/asger/mini_trainer/.worktrees/prototype-space`;
verify its status before work. Use assigned worktrees for any authorized parallel
implementation; integration remains a separate reviewed step.

Use the epoch-26 checkpoint and hash recorded in the roadmap. Epoch 4 is the
longitudinal comparison. Existing effective-weight extraction, class ordering,
distance and log-tail functions remain the numerical baseline. Synthetic cases
exercise edge conditions without specifying how the real prototypes should pack.

## 1. Reliable repeated exploration — next implementation goal

> Make the current explorer resumable and reusable: restore compatible viewer
> state, reuse completed analysis, expose preparation progress, and cancel or
> replace preparation without publishing stale results. Validate with the real
> epoch-26 case and preserve existing numerical outputs and interaction behavior.

Deliver in three reviewable slices:

| Slice | Implementation boundary | Acceptance |
| --- | --- | --- |
| 1a · Viewer state | Versioned state serializer in the packaged viewer. Save case/model identity, selected class ID, focused view, plane and pan/zoom, subtree, colour range, name preferences and thumbnail settings. Provide explicit reset and JSON export/import. | Reload restores a compatible view without refitting or changing class order. Different models and malformed/older state fall back clearly. Storage failure does not prevent use. Do not persist transient force positions or local image files. |
| 1b · Analysis reuse | Extend `launch.py` and `explore.py` with an explicit persistent cache location and a manifest keyed by checkpoint content hash, extraction/analysis version, numerical settings and projection method/seed. Store completed numerical artifacts separately from viewer assets. Publish atomically; bound cache size and expose clearing. | A second open skips analysis; changed weights/settings invalidate the entry; partial or corrupt entries rebuild safely. Viewer-only changes render current assets over cached data. Cache hit/miss and provenance are visible. |
| 1c · Preparation lifecycle | Extend the launcher's existing child-process build into an owned cancellable job. Report named stages and elapsed time, not invented percentage progress. Reuse its subprocess boundary for non-interruptible library stages. Use job identities for publication. | Cancel, replacement, failure and shutdown release owned resources. A cancelled/older job never replaces a newer report or cache entry. The last completed report remains usable. |

Before changing behavior, record cold preparation time, reopening time, peak RSS,
HTML size, browser load time and interaction frame timing on the same machine.
Repeat after implementation with identical numerical options; distinguish disk,
analysis and browser costs. A cache hit must execute no analysis stages. Do not
turn environment-dependent timing into brittle regression assertions.

Checks: extend `test_prototype_launcher.py` for invalidation, interrupted writes
and stale-job races; use `test_prototype_exploration.py` for unchanged results.
Extend the browser harness for restoration, model mismatch and storage failures.
Run affected Python tests and static checks; run the full harness if changes
extend into model loading or the validation harness as required by repository policy.
Refresh the real-data preview after each accepted slice.

## 2. Real-model browser feasibility — next decision gate

Execute milestones 1–3 of the standalone plan as a bounded qualification goal:
contract and fixtures, opt-in predictions-plus-embedding export, then WASM browser
parity. Inspect the actual architecture and preprocessing first. The current
prototype-only loader does not establish that a complete inference model can be
reconstructed without additional metadata.

Deliver a versioned bundle proposal, reproducible export/qualification command,
real-image reference fixtures, and a minimal browser inference harness. Start
with identical preprocessed tensors, then qualify image decoding/preprocessing.
Measure cold load, warm latency and peak memory on a named browser/machine.
Use the existing ONNX exporter and output semantics; do not duplicate the head.

Pass only when real checkpoint embeddings and predictions satisfy the standalone
plan's parity gates. If required metadata/images are missing or runtime operators
fail, report the concrete missing input or operation and the smallest next step.
Do not replace real-model evidence with a tiny model or silently relax tolerances.
Full inference UI, WebGPU and client-side global analysis remain later milestones.

## 3. Local packing diagnostics — next geometry feature

Add a selected-class packing panel linked to the existing inspector. Compute
angular cap counts and neighbour-radius curves from original-space relationships;
let the user select a radius or rank and highlight the same class set in map,
matrix and tree. Add mutual-neighbour membership and taxonomy composition using
available checkpoint hierarchy; missing names/taxonomy stay explicit.

Start with selected-class computation rather than another dense all-pairs payload.
Record the source, clamp convention, self-exclusion and tie handling. Keep empirical
counts separate from any optional uniform-sphere reference. Chance-alignment q
remains a reference tail, not a learned-class posterior or an automatic significance
decision across many selected pairs.

Gate: exact agreement with repository-derived relationships on epoch 26 and
algebraic fixtures (duplicates, antipodes, orthogonality and ties). Selection must
not refit the projection or restart thumbnail motion. Provide a preview where a
radius change visibly identifies the same neighbourhood across linked views.

## 4. Anchor view — then query embedding UI

Implement a single-anchor view with angular radius, followed by selectable bearing
and two-anchor comparison only after the first view is validated. Define bearing
degeneracy at parallel/antipodal directions and disclose non-anchor distortion.
Reuse image labels, floating controls and culling rather than creating a new viewer.

Gate: anchor radii agree with direct angular calculations; changing display bearings
does not change neighbourhood membership. This provides a geometrically defined
view for query embeddings once browser inference is qualified.

Then execute standalone milestones 4–5: local image selection, actual predictions,
query-to-prototype diagnostics and fixed-transform PCA/anchor placement. Qualify
fixed-map t-SNE insertion separately. Finish milestones 6–7 for client-side global
diagnostic construction and standalone/offline distribution; inference alone does
not complete the standalone goal.

## Research queue and reporting

Keep slices, alternative layouts and checkpoint trajectories behind these gates.
Formalizing chance-alignment inference, multiplicity and empirical calibration is
a research task; it must not change the baseline numerical semantics implicitly.

Each increment ends with a runnable preview, focused commit, tests and compact
real-data evidence, plus roadmap status and remaining limitations. Do not start
all increments as one implementation task. The first quoted goal above is the
recommended next goal; no token/time budget is implied.
