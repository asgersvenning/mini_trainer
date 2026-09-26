# Repository roadmap

Updated 26 September 2026. This is the cross-campaign priority map; specialist
pages own procedures and evidence. Delivered work is context, not a new checklist.

## Current state and next delivery

| Area | Delivered | Remaining boundary |
| --- | --- | --- |
| MAMBO V3 | PyTorch/ONNX adapter, presets/custom lists, embeddings, TTA, Flemming/in-domain comparisons and laptop/B200 timings; installed release candidate qualified. | Renamed training distribution, independent publication Actions and a small Space demo are prepared; installed-candidate records gate publication. Account setup, publication and live endpoint checks remain owner tasks. See [final qualification](../dev/releases/mambo_v3/final-qualification.md) and [publication handoff](../dev/releases/mambo_v3/publication.md). Do not restart completed release experiments. |
| Training | Four-GPU UCloud production run completed; loader, optimizer-step and checkpoint safeguards implemented. | Before another production run, deliver the recovery/evaluation workflow below. |
| Quantization | Merged opt-in native CUDA INT8 Linear training, x86 PTQ/QAT, checkpoint/export tools. | Useful target-machine trade-offs remain unqualified; [specialist roadmap](quantization-roadmap.md). |
| Generic export | `mt_export`, manifests and CPU float32 ONNX qualification across representative heads/backbones. | Broader backend qualification and generic Hugging Face bundle integration; MAMBO packaging does not establish either for every model. |
| Portable viewer | Existing feature-branch implementation and published inspection candidate. | Reconcile review/integration state and complete physical-device acceptance; see below. |

For the **next training campaign**, prioritize durable stage state and recovery,
prepare evaluation/export before allocating GPUs, and qualify storage separately
from compute. The [training post-mortem](training-workflow-postmortem.md) explains
why. Use normal CLIs, preserve operator overrides, figures and W&B. These workflow
improvements remain planned; another broad optimization matrix is not a prerequisite.

The following order governs general development. Finish bounded increments with
observable compatibility checks rather than starting every listed campaign.

## 1. Agent guidance and development safeguards

Delivered: shared static/runtime checks, import contracts, installed-wheel smoke
checks, locked CI and a separate latest-compatible dependency matrix for Python
3.12–3.14. Commands and limitations belong in the [development guide](../dev/README.md);
agent rules belong in [AGENTS.md](../AGENTS.md).

Remaining: verify required-check behavior on the first hosted agent-only PR under
actual branch protection. Preserve one validation entry point, static architecture
checks that do not initialize CUDA, and minimal installed-package coverage.

## 2. Behavior-preserving simplification

Checkpoint tests cover live/reloaded predictions, state restoration and controlled
CPU continuation with fixed data order and the original epoch budget. They do not
establish arbitrary RNG/sampler or active AMP continuation. Loader coverage includes
sampling, class/label order, caching, worker budgets and transfer ownership.

**EMA remains unsupported.** Evaluation populates classifier caches whose shapes
can differ between the training and EMA models, breaking averaging. Keep the strict
expected-failure regression and leave EMA disabled until a separate repair is
qualified; do not infer support from checkpoint restoration alone.

Extract orchestration from builders/training only when it removes an identified
responsibility conflict or duplication. Preserve public imports, defaults, outputs,
checkpoint formats and errors; separate behavior changes from cleanup. Use existing
CPU training/DDP and checkpoint contracts for cross-cutting changes.

## 3. ONNX export and Hugging Face integration

The [export guide](onnx.md) owns preprocessing, score and operator contracts.
Current generic evidence covers CPU float32, dynamic batches, head families,
masks/priors and caller-state preservation on representative offline backbones.
GPU/quantized providers and the full architecture catalog need their own evidence.

Next: prepare a generic local bundle containing weights/export, preprocessing,
class mappings, immutable provenance, model card, evaluation and an inference
example. Reuse the existing manifest/export boundaries and lessons from MAMBO.
Validate the installed bundle before adding upload commands. Artifact hosting and
a live inference service are separate deliverables; publication is not automatic.

## 4. Training efficiency and augmentation

Follow the [quantization roadmap](quantization-roadmap.md) for targeted performance
work. Floating-point defaults remain; native QT DDP/FSDP and deeper integer
convolution training are unsupported. Local memory savings do not establish useful
HPC, desktop/Spark or ARM throughput or convergence.

For the next production run, implement the post-mortem's bounded workflow slice:
versioned stage inputs, durable completion/recovery, preflighted evaluation/export,
and distinct compute/storage qualification. Preserve supplied splits and taxonomy.
Demonstrate interruption/resumption without silently repeating completed work.

Then use the [training feature protocol](training-feature-validation.md) for paired
optimizer, loss and augmentation comparisons. Preserve accumulation, AMP skip,
scheduler and resume semantics. Add loading controls only for an identified
end-to-end bottleneck, not to expand the configuration surface.

### Optional dataset preparation for scalable loading

Measure bounded concurrent reads/staging on a representative shared-storage working
set before building shards. Small warm subsets do not predict full-dataset IO.
If preparation is justified, use the existing metadata/reader boundaries and
start with indexed uncompressed TAR rather than a new container format.

Required contract: preserve encoded bytes, sample identity, supplied splits,
class ordering, multilabel/hierarchical targets and current sampler/DDP coverage.
Apply stochastic transforms at runtime. Use versioned provenance and integrity,
bounded resumable preparation, atomic completion and explicit corruption errors.
Local staging/cache must coordinate ranks, bound disk/memory and protect active
reads. Locality-aware shuffle is a separate behavior change.

Deliver compatibility fixtures and a round trip, then indexed shards, then optional
shared staging. Compare cold and warm end-to-end training including preparation,
validation and figures; report amortization in epochs. Keep original inputs and
public APIs usable. This remains planned, not an implemented storage backend.

## 5. mini_metrics and continuous model evaluation

MAMBO has retained quality/threshold/support evidence through `mini_metrics`;
[release evidence policy](../dev/releases/mambo_v3/evidence-policy.md) defines that
campaign's conventions. General training-feature comparisons and continuous GPU
allocation remain separate work.

Build on [existing benchmark reporting](../dev/benchmarks/reporting.md): fixed
representative subsets, supplied splits, paired seeds and matching measurement
scopes. Keep all metrics and negative results. AdamW/SGD step tracking and AMP-skip
scheduler/EMA gating have regression coverage; comparative quality is still planned
and EMA itself is unsupported.

Before scheduling on-demand GPUs, qualify an isolated trusted controller with
restricted revisions/profiles, credentials unavailable to candidate code, persistent
spending limits, bounded concurrency/duration and reconciliation before retries.
Mock interruption, duplicate submission and budget exhaustion, then run a capped
live success/failure/cancellation and publishing retry. Missing runs must stay
visible. No automatic allocation extension or full-device claims from MIG results.

Keep evaluation optional. The recorded `mini_metrics` integration requires Python
3.13 while the trainer supports 3.12; select a compatible evaluation environment
rather than depending on the sibling checkout. Preserve prediction CSV/sample IDs,
truth, class ordering, score/threshold meaning and missing-ancestor semantics.
`publication/experiments/statistics/boot_metrics.py` remains a research integration
entry point. Model-zoo manifests should bind immutable weights, preprocessing,
data/split identity, dependencies and results so new protocols cannot silently
rewrite old baselines.

Deferred CLI consolidation: `mt_hpredict` already shares the generic prediction
CLI and supports flat heads. Before replacing entry points, qualify metadata-based
builder selection, older weights with missing metadata, explicit overrides and
prediction/metric output compatibility. This is not a production-run prerequisite.

## 6. Additional dataset formats (low priority)

Use shared train/inference source adapters and keep sample discovery independent
of model vocabulary and training partitioning. Require tiny round-trip fixtures
for flat/hierarchical/multilabel inputs, lazy loading and supplied splits before
adding a format. Keep format dependencies optional.

Carry forward these concrete findings; verify their current callers before fixes:

- `create_taxonomy` / `select_levels` use an inclusive deepest-rank index, while
  some callers pass a level count. Audit remaining callers and test actual ranks.
- Ambiguous folder layouts need explicit source/split policy; reconcile prediction
  `data_index` and source selection without restoring vocabulary-filtered discovery.
- Missing ancestors need explicit provenance/metric handling; retain valid species
  truth without inventing unseen ancestor classes or masking retrieval failures.
- `is_image` opens in `r+b`, potentially excluding readable read-only datasets.
- Bare-head `classification_module` can cache an empty attribute name and fail on
  subsequent lookup; ordinary built backbones are unaffected.

These are recorded follow-ups, not fixes delivered by documentation cleanup.

## Deferred portable prototype viewer completion

Resume the existing `feature/prototype-browser-inference` implementation rather
than rebuilding it. The recorded candidate is `global-lepi-viewer-20260915-rc2`;
[plan at b174426](https://github.com/asgersvenning/mini_trainer/blob/b174426/docs/prototype-explorer-implementation.md)
and its branch's `dev/prototype_space/portable-qualification.md` retain acceptance
criteria. Reconcile current PR #3/branch state before resuming; historical publication
does not establish present integration status.

Remaining acceptance: human desktop/mobile review and physical-phone pinch zoom,
camera capture, EXIF, large images and actionable format errors; resolve findings
and validate integration against current master. Preserve GBIF settings/names/photos
and attribution, Explore/Predict/Settings organization, class identity/order, saved
views and fixed-map t-SNE placement. Keep static hosting independent of a Python
service. Publish a new inspection candidate only if review requires changes.

The [prototype-coordinate study](prototype-coordinate-study.md) is separate
research evidence; its conclusions do not authorize new geometry or changes to
training/prediction semantics during viewer completion.
