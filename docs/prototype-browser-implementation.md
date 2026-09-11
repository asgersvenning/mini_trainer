# Standalone prototype explorer: implementation plan

Status: ready for goal setting; implementation has not started. This plan defines
execution and acceptance for the standalone direction in the
[compact roadmap](prototype-explorer-roadmap.md). It does not create an active goal.

The [unified execution plan](prototype-explorer-implementation.md) schedules
milestones 1–3 as a bounded feasibility goal before the inference UI. The goal
below remains the full seven-milestone destination, not the next single increment.

## Goal text

> Build and validate a standalone prototype explorer that loads a mini_trainer
> export bundle and user-selected images, runs model inference in the browser,
> and explains predictions through embeddings in the model's prototype space.
> The distributed tool must require no Python, PyTorch, CUDA, or inference backend.
> Support local preprocessing, predictions, embedding-to-prototype diagnostics,
> query placement, and generation of the existing global exploration views.
> One-time model export may use mini_trainer. Precomputed diagnostics may accelerate
> opening a bundle, but must not be the only way to obtain the global views.
> Use the epoch-26 production checkpoint and existing numerical functions as the
> reference, preserve current explorer interactions, and provide reproducible
> browser parity, performance, offline-operation and installation evidence.

## Reference and boundaries

- Checkpoint: `/home/asger/mini_trainer/tmp/best_global-lepi-production-w32-1_epoch26.pt`.
  SHA-256: `abb9f66d95fe867bd31847cb33782a353b1d3faf4c79c4b10d93037f33fe61f4`.
- Verified prototype shape: 12,632 × 1,280; normalized float32 effective rows,
  zero biases. Epoch 4 remains a class-aligned comparison, not a replacement model.
- Initial supported heads: the explorer's current `Classifier` and
  `HierarchicalClassifier` linear heads. Reject unsupported contracts explicitly.
- Start in `/home/asger/mini_trainer/.worktrees/prototype-space`, on
  `feature/prototype-space`; revalidate HEAD and status before work. Further
  concurrent implementation requires individually assigned worktrees and scopes.
- Preserve current Python entry points and training/export defaults. Export is an
  authoring step; the browser consumes the resulting bundle, not arbitrary `.pt`.
- No architecture substitution, calibration claim, assumed packing distribution,
  or unlabelled approximation. Browser inference does not establish GPU-training
  parity or pixel-attribution explainability.

## Architecture and deliverables

Extend the existing exporter and explorer; avoid a second classifier implementation.
Keep authoring in Python, and inference/analysis in browser workers with a static
HTML/JS/WASM distribution. Prefer WASM CPU qualification first; add WebGPU only
with provider-specific parity and capability checks. Pin runtime assets locally.

Define a versioned bundle with:

| Artifact | Required content |
| --- | --- |
| Model | ONNX graph and every external tensor-data file; actual evaluation outputs plus a named preclassification embedding output. |
| Manifest | File hashes, checkpoint identity, output-tree semantics, effective class/hierarchy order, shapes/dtypes, embedding stage, preprocessing contract, supported runtime versions. |
| Prototypes | Binary effective weights, biases and necessary score metadata; retain magnitudes separately from directional geometry. |
| Projection state | PCA centering vector and basis; optional global coordinates/diagnostics with method, seed and numerical-configuration provenance. |
| Verification fixture | Legally usable real-image references or reproducible local fixture instructions, hashes, preprocessed tensors, embeddings and actual output tensors. |

Allow selection of the complete bundle through the UI, including external ONNX
data. Choose and test a container/loading format in milestone 1; do not assume a
single ONNX file is a complete deployment. Keep large tensors binary rather than
embedding extra copies in JSON. No browser-to-server model/image upload.

## Ordered implementation milestones

### 1. Contract and reference fixture

Inspect the epoch-26 architecture and exact evaluation preprocessing using the
existing checkpoint loader. Identify the embedding after head preclassification,
including any hidden transform and normalization; do not substitute raw backbone
features. Locate suitable real validation images locally and record their hashes.
Missing validation data must be reported, never replaced by synthetic-only evidence.

Implement bundle schema validation and small flat/hierarchical fixtures. Define
explicit unsupported-input and version errors. Record the baseline environment,
model outputs and extraction semantics before changing export code.

**Gate:** fixture outputs reproduce the current model, and every bundle field has
an identified producer and browser consumer.

### 2. Export predictions plus embeddings

Extend `mini_trainer/modeling/onnx.py` with an opt-in, export-compatible path for
both output kinds. Reuse the actual forward and do not depend on the training-only
`EmbeddingContext`: export currently rejects active training/supervision contexts.
Avoid a duplicate backbone pass. Preserve hierarchical output structure and masks.

Add authoring support through the existing export/explorer commands, with weights
as the only required model argument. Extract executable preprocessing from supported
transforms; fail with a specific unsupported-transform error instead of guessing.

**Gate:** real images and batches 1/2/4 match the PyTorch reference for predictions
and embeddings. Start with existing export tolerances (`rtol=1e-4`, `atol=1e-5`);
report error distributions and ranking differences. Any tolerance revision requires
an explained numerical investigation. Source checkpoint and model state stay intact.

### 3. Browser inference qualification

Load the real exported model with ONNX Runtime Web in a worker. First compare using
identical preprocessed tensors, isolating runtime errors from image decoding errors.
Then implement and verify image decoding, orientation, channel order, resizing,
cropping, scaling and normalization end to end. Include portrait/landscape images
and relevant alpha/orientation cases. General browser canvas resizing is not assumed
to reproduce the repository transform.

**Gate:** WASM inference passes real-image parity; WebGPU is enabled only after its
own checks. Record runtime/browser versions, cold load, warm latency and peak memory.
Unsupported providers produce an actionable result and a verified fallback.

### 4. Inference and geometric explanation UI

Add model/image selection, progress, cancellation, bounded image batches, prediction
inspection and query thumbnails. Show actual head scores with their semantics.
Compute query-to-prototype geometry using the existing clamp/self-pair conventions;
port log-domain operations with independent tail/extreme-value tests. Query images
are not automatically class members or labelled correctly.

Show angular neighbours, competing predictions, score margins and neighbourhood
profiles; distinguish geometric proximity from the full classification rule.

**Gate:** browser calculations agree with repository references; latest selection
wins under rapid changes; cancelled work cannot publish stale results; loading does
not block navigation or restart thumbnail motion.

### 5. Place inference embeddings in the views

Implement fixed-transform PCA placement from saved means/bases and direct angular
anchor inspection. Add a separately validated method for inserting queries into a
fixed t-SNE layout; keep prototype coordinates fixed and disclose placement error.
Do not place images at predicted-class coordinates or label neighbour interpolation
as exact t-SNE projection. Evaluate poor as well as favourable query cases.

**Gate:** PCA/anchor geometry matches direct computation; nonlinear placement has
measured neighbourhood/distance distortion; query selection links maps, neighbour
inspection and prototype/class photos without changing the underlying prototypes.

### 6. Generate global diagnostics client-side

Port the computations necessary for the current matrix summaries, Ward dendrogram,
PCA and angular t-SNE to workers/WASM/WebGPU as appropriate. Use blocked data access,
bounded allocations, persistent caches, cancellation and progress. Estimate real
peak allocations before running the full case. A single dense float32 matrix for
this model is about 638 MB; several simultaneous copies are not an acceptable default.

**Gate:** the epoch-26 bundle can construct the global views without precomputed
results or a Python service on a documented reference machine. Distances and log tails
match baseline tolerances; tie/order differences in linkage are explained and tested.
Projection equivalence is assessed by its objective, fidelity and stability, not
pixel equality. Alternative/approximate algorithms remain explicitly selectable;
they do not silently replace the baseline. If this gate is infeasible, report it
as unfinished rather than declaring an inference-only viewer complete.

### 7. Standalone distribution and release audit

Package as a static application with an installable/offline route. Static hosting
may serve files, but must perform no inference or numerical analysis. Do not promise
that double-clicking `file://` supports every browser capability. Bundle runtime
assets; record required secure-context/isolation settings and tested browsers.

Use local model/image selection and verify offline reopening after installation.
Treat GBIF images as optional network content with attribution and explicit offline
behaviour. Test model replacement, memory/resource disposal and failure recovery.

**Gate:** operate from a clean environment without Python/PyTorch/CUDA, with model
and images retained locally. Network inspection proves no inference-service calls.
All existing explorer interactions still work, including labels/aspect ratios,
arrival continuity, culling fades and fixed prototype positions.

## Validation and completion evidence

Extend `tests/export/test_onnx.py`, the affected `tests/utils/test_prototype_*`
coverage, and the existing browser checks. Keep new fixtures small for regression
runs; maintain a separate reproducible real-checkpoint qualification command.

Run `bash dev/check.sh static`, affected tests, and `bash dev/check.sh all` for the
export/loading changes as required by repository policy. Run
`bash dev/check-wheel.sh` for packaging/extras, plus installed-bundle browser tests.
Use isolated dependency environments with an explicit PyTorch backend; never sync
the shared CUDA environment implicitly. Preserve real fixtures and outputs outside
Git; commit schemas, tests, commands and compact evidence summaries.

Completion requires all seven gates, a runnable distribution, updated usage docs,
and an evidence table recording artifact hashes, platforms, numerical errors,
latency/memory, skips and unresolved limitations. A miniature model, CPU ONNX check,
precomputed-only viewer or screenshot alone is insufficient. Commit bounded stable
increments and update the compact roadmap; do not mark the overall goal complete
when only an earlier milestone has passed.
