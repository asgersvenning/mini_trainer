# Prototype explorer: next implementation increment

Status: implemented and qualified; human layout review and PR integration remain pending. Updated 2026-09-11. The
[roadmap](prototype-explorer-roadmap.md) owns priorities; this document is the
executable plan for the next bounded increment.

## Objective and baseline

Make the existing static prototype explorer coherent to navigate and portable for
GBIF-labelled models: one global class-ID interpretation, browser-side names and
reference images, and predicted-species thumbnails integrated with the existing
map and inspector.

Start from `feature/prototype-browser-inference` at `1eb2a32`, which includes
master `1d2dae0251989b067908a7c06c0701a890b11580`. Before implementation, check the
current master head and integrate any newer changes through the normal reviewed
branch workflow. Preserve unrelated work and the existing published release.
Use the assigned `prototype-browser-inference` worktree or a newly assigned
worktree following repository rules; do not use the older prototype-space worktree.

The first browser inference and fixed-map t-SNE insertion are already implemented.
Preserve their model/preprocessing contracts, class order, scores, embeddings,
projection coordinates, pan/zoom, selection and saved-view functionality. No new
projection method, model re-export or inference engine is required for this increment.
Prefer the existing architecture where practical. Evaluate a framework if it offers
clear maintainability or usability benefits; explain the migration cost and
compatibility implications before committing to that direction.

## Problems at the starting point

- `report.html` resolves names through `/api/taxon/`.
- `photos.js` fetches albums through `/api/gbif/`; it and `thumbnails.js` probe
  `/api/health`, and images use the Python `/gbif-image/` proxy.
- `gbif-names`, `photo-enabled` and `map-photos` independently imply GBIF
  interpretation; `state.js` persists those independent controls.
- `inference.js` renders raw class IDs and scores without reference thumbnails.
- Model loading, prediction, exploration and numerous settings compete for space.
  Earlier focused-layout work does not resolve the current combined interface.

## Reviewable implementation slices

### 1. Global class identity and shared browser GBIF access

Add one model-scoped `classIdNamespace` setting with `generic` and `gbif` values.
Expose it once in shared settings. Numeric strings alone must not enable GBIF.
Allow explicit report/bundle metadata to supply a default, with the user's saved
choice taking precedence. Generic is the fallback when neither exists; synthetic
cases make no GBIF requests. Treat names and photos as presentation preferences,
not alternative namespace settings.

Extend `state.js` rather than creating a parallel state store. Version the saved
state and migrate older views: an explicitly enabled legacy GBIF name/photo option
maps to GBIF mode; otherwise retain generic mode. Record this migration in tests
and documentation. Keep imported aliases usable in either mode. Use one shared
label resolver, with imported aliases, packaged names, resolved online names and
raw IDs in that order. Namespace/model changes cancel obsolete work and invalidate
render generations so late results cannot overwrite current labels or images.

Put GBIF transport and metadata normalization in one logical `gbif.js` module.
Share it across name labels, selected-class cards, map thumbnails and prediction
cards. Reuse existing request deduplication, visible-item prioritization and photo
rendering where possible. Use session caching and bounded requests, seeded by an optional versioned GBIF
metadata snapshot packaged with the artifact. Precompute scientific names for the
model vocabulary so labels can render offline without thousands of initial calls.
Include snapshot provenance, retrieval dates and original taxon IDs; allow partial
snapshots and fetch missing entries on demand. Photo metadata (URLs and attribution)
may also be included, but does not imply that remote image bytes work offline.
Keep snapshot preparation optional and resumable; do not make live API availability
a prerequisite for export or build a general taxonomy service.

Use public GBIF species and occurrence APIs directly from the browser. Verify
actual CORS access from a static HTTP origin before expanding implementation.
Select a browser-accessible thumbnail source, preserving creator, licence,
occurrence link and source URL. Respect the chosen endpoint's documented request
limits. Bound timeouts/retries and handle 429, unavailable images and offline
operation without blocking numerical views. Do not replace the Python dependency
with a mandatory third-party proxy. If image delivery needs a fallback, prove it
on the static host before claiming portability.

Accepted-name/synonym resolution is display metadata: retain original class IDs
and model row ordering. Validate occurrence taxonomy before associating an image
with a predicted species; do not silently use unrelated or merely same-genus
images. Render external text safely and restrict media/link URL schemes.

Keep existing Python endpoints compatible for local users during this increment,
but remove their use from browser name/photo features. The local weight picker
and optional server-side saved state remain separate Python capabilities.

### 2. Prediction cards with species reference images

Extend `inference.js` to use the shared label resolver and photo components.
Species results show name, original ID, unchanged score and a small reference
thumbnail with accessible attribution. Label these as reference examples, distinct
from the user's inference image. Load only visible results; share albums with map
and inspector consumers, and discard stale requests after new inference/model changes.

Genus/family results use shared names but need not acquire species thumbnails.
Do not imply that an arbitrary descendant is the predicted species. Preserve
existing map linkage where meaningful and make any parent-rank selection behavior
explicit. Missing images, unknown taxa and offline operation retain useful ID/score
rows. Generic classes render normally with no GBIF calls.

### 3. Cohesive layout using existing components

Organize the interface around Explore and Predict, with shared Settings. Keep the
full-pane map available and the current query overlay linked to prediction results.
Place model loading and image selection together, with one clear progress/error
area. Group prediction results by rank; put advanced projection, thumbnail and
numerical controls behind labelled disclosure panels. Consolidate duplicated
controls rather than adding another toolbar.

Preserve keyboard operation, visible focus, readable status, attribution access,
mobile usability and pan/zoom across navigation. Prepare desktop and narrow-screen
previews for human review before polishing. Do not redesign the underlying
analytical views or remove expert controls to simplify the first screen.

## Acceptance and validation

| Area | Required evidence |
| --- | --- |
| Static portability | Serve the exported viewer using a plain static server with no Python API endpoints. Names, map photos, inspector photos and prediction thumbnails work. Network assertions find no `/api/taxon`, `/api/gbif`, `/gbif-image` or GBIF health-probe dependency. Test the intended HTTPS hosting origin as well. |
| One setting | A single global namespace choice governs every consumer. Switching to generic prevents new GBIF requests and stale results; numeric generic IDs and synthetic cases remain generic. |
| Compatibility | Legacy saved views migrate deterministically; aliases, class order, hierarchy, scores and query positions stay unchanged. Browser storage failure does not prevent use. |
| Shared resources | Multiple consumers requesting one taxon share in-flight work and cached metadata. Navigation/inference replacement does not create unbounded queues or stale cards. |
| Failure handling | Mock timeout, 429, malformed responses, missing ranks/images, synonym records and offline mode. Useful labels and predictions remain; retries are bounded. |
| Predictions | Known species has a credited thumbnail; missing-image species has a stable placeholder; genus/family rows do not masquerade as species. Local query pixels are never sent to GBIF. |
| UX | Desktop and narrow-screen screenshots plus browser checks cover navigation, keyboard focus, settings, prediction cards, map selection, zoom retention and readable loading/error states. Obtain human review of the preview. |
| Performance | Record same-machine initial display, interaction responsiveness and visible-thumbnail request counts before/after. Do not resolve all 12,632 names or fetch all species albums at startup. |

Browser acceptance for this increment targets the existing Chromium/WASM setup.
Narrow-screen checks use that browser; Firefox, Safari and other runtime backends
are deferred to the roadmap and are not completion gates for this goal.

Extend the existing browser harnesses in `dev/prototype_space/` with mocked GBIF
responses for deterministic checks and a small separate live static-host smoke test.
Run `bash dev/check.sh static` and affected prototype Python tests; use the existing
inference harness to establish unchanged predictions and embedding placement.
If packaged assets change, run the installed-wheel smoke check. Follow repository
full-suite requirements if model loading or the validation harness scope expands.
Report live-network failures separately from deterministic regressions.

## Completion and distribution

Deliver focused normal commits for the state/client contract, prediction integration
and layout changes, with tests alongside each. Keep agent-only notes separate.
Update usage and browser guides, this plan and the roadmap. Integrate through a
reviewed PR against current master and validate the combined state. Prepare a
versioned static preview with provenance and asset hashes; do not overwrite the
published production release as a side effect of development.

## Executable goal

> Implement the portable viewer UX increment in
> `docs/prototype-explorer-implementation.md`, keeping compatibility with current
> master. Consolidate GBIF interpretation into one model-scoped global setting;
> replace Python-backed name/photo lookups with one shared browser GBIF client;
> show attributed predicted-species thumbnails; and reorganize existing controls
> into coherent Explore, Predict and Settings flows. Preserve numerical behavior,
> class IDs/order, saved-view compatibility and fixed-map t-SNE query placement.
> Complete the acceptance matrix, static-host browser proof, focused checks and
> human-reviewable desktop/mobile preview. Deliver focused commits and a reviewed
> integration PR, updated documentation and a versioned release candidate. Prefer
> the existing architecture where practical; a framework may be justified by clear
> maintainability or usability benefits, with migration cost and compatibility
> implications explained before committing to that direction. Do not
> change training/inference semantics or implement new geometry,
> require a proxy service, or overwrite the existing public release. Report any
> externally blocked check explicitly rather than claiming it passed.

This goal is active. The [qualification record](../dev/prototype_space/portable-qualification.md)
contains passing check evidence. Remaining completion gates are human review,
the integration PR and candidate delivery. No token or time
budget is implied.
