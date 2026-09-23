# UCloud model release roadmap

Status: release preparation started, 2026-09-23. This document does not publish
artifacts or claim deployment qualification. Target: the completed 10–11 September
2026 UCloud model, not a new training campaign.

The [first input-audit increment](../dev/releases/mambo_v3/README.md) now pins and
verifies 44 retrieved files, including candidate PyTorch/ONNX weights and historical
MAMBO weights. It recovers both regional presets, confirms identical old/new class
and parent mappings, and captures a small legacy output fixture. Its regional-scope
table now includes reproducible Parquet filters: Europe uses the metadata continent
field; northern Europe has an exact country-filter reconstruction with documented
ambiguity for membership-neutral additions such as Ireland. Adapter compatibility
and inference qualification remain outstanding.
Local evaluation will use Flemming; the large in-domain dataset remains on UCloud
and must be evaluated there using the original supplied split.

## Branch and integration policy

Release development takes place on `release/mambo-v3`, created after the package
minor-version bump from `0.2.0` to `0.3.0` on `master`. The package version is
separate from the proposed `MAMBO_v3` model release tag; no release tag or public
promotion is implied by creating this branch.

Direct commits are limited to release assets, deployment adapters, presets,
packaging, documentation and release-specific compatibility/evaluation tooling.
Fixes or refactors to shared core code (including model loading, preprocessing,
prediction, hierarchy and export internals) must originate on `master` or a
dedicated feature/fix branch, pass their relevant checks, and then be merged into
the release branch. Review the merge scope and validate affected combined behavior.
Apply this rule to the existing browser/export branch too; integrate the required
reviewed work rather than reimplementing core changes directly here.

Classify a change by its purpose and affected boundary, not just its filename:
release adapters may use existing core interfaces, but a prerequisite core fix
remains a separate upstream change. Keep unrelated improvements out of this branch.

## Release objective and scope

Ship a versioned successor to the public MAMBO deployment release that is easy to
install, embed and operate without a GPU, internet access, administrator rights or
a writable installation directory. Preserve a clear migration path for existing
Python and CLI users. Add acceleration only through separately qualified profiles.

The release targets **backwards compatibility with MAMBO_v2**, with native PyTorch
and standard floating-point ONNX as equal supported paths. Use the existing raw
PyTorch weights, standard prediction-only ONNX, and tested floating-point
prediction-plus-embedding ONNX derivative. Preserve the original files and pipeline
identities. This release ships the native PyTorch and standard ONNX models with
the region-specific presets. Quantization is deferred to a later release: no PTQ
artifact packaging, calibration, quantized benchmarks or quantization acceptance
gates belong to this increment. New FP16 graph conversions, TensorRT and new model
formats are also outside its scope.

Both backends must support `full`, `europe`, `north_europe`, custom class lists,
and predictions with or without embeddings through aligned interfaces. Compare
usefulness on the in-domain test set and out-of-domain **Flemming** expert dataset,
plus speed and memory on this laptop's CPU and GPU. Small ONNX score variations are
expected and acceptable: micro-numerical parity is not a release objective. Existing
export checks remain intact; new release gates concern behavior, task quality and
practical trade-offs.

Training resume state and the research archive remain optional downloads. Additional
OS/browser/accelerator qualification and Hub hosting follow the core comparison;
they do not delay a release with an honestly scoped support matrix.

This release work takes precedence over the next-training-run improvements in the
[training post-mortem](training-workflow-postmortem.md). Retraining, loader redesign,
EMA repair, full INT8 training and prototype-coordinate research are not release
prerequisites. The [portable viewer work](roadmap.md#deferred-portable-prototype-viewer-completion)
remains a separate integration track; reuse its implementation where relevant.

## Verified starting point

The public GitHub releases API was read on 2026-09-23. Local tagged source was
inspected alongside it; a local tag alone does not establish a published release.

| Published release | Established interface and behavior | Alignment required |
| --- | --- | --- |
| [MAMBO_v2](https://github.com/asgersvenning/mini_trainer/releases/tag/MAMBO_v2), 17 April 2026 | Latest published release; BioCLIP-2 model; `mambo_predict`; `mini_trainer.deploy.Predictor`; `full`, `europe`, `north_europe` aliases; default region Europe; weights downloaded from ERDA | Required backwards-compatibility baseline. Preserve old version pins, document architecture and vocabulary changes, and explicitly decide the successor's default |
| [MAMBO_v0](https://github.com/asgersvenning/mini_trainer/releases/tag/MAMBO_v0), 15 April 2026, prerelease | Earlier MAMBO deployment wrapper and northern-European model emphasis | Historical context; no additional backwards-compatibility gate |
| [UKCEH_v0](https://github.com/asgersvenning/mini_trainer/releases/tag/UKCEH_v0), 3 February 2026 | Northern-European EfficientNetV2-M model; `python predict.py`; automatic model download | Historical context; MAMBO_v2 is the compatibility target |

None of these release records has attached binary assets. The MAMBO_v2 source
resolves weights through an ERDA URL template and an implicit cache. Its README
still points installation commands at MAMBO_v0; correct this in the new release's
instructions. MAMBO_v1 exists as a local tag but was absent from the public release
listing. Do not treat it as an independently published baseline without evidence.

MAMBO_v2's `Predictor` defaults to CUDA, batches all supplied images together, and
supports class masks and embeddings. Current master has `mt_predict`/`mt_hpredict`
and the generic exporter, but no `mini_trainer.deploy` module or `mambo_predict`
entry point. Restoring compatibility therefore requires implementation and tests,
not just substituting a new weight URL. Audit exact return values and CSV schemas
from the tagged code before promising drop-in compatibility.

The [campaign record](training-workflow-postmortem.md#outcomes-and-strength-of-evidence)
reports EfficientNetV2-S, a normalized hierarchical head, 384-pixel inputs, 30 epochs,
and floating training on four B200 GPUs. It reports completed evaluation and
verified archival checksums, but does not contain the final immutable artifact
identities. The feature-branch follow-up below identifies the public distribution
and adds limited real-image evidence. The original FP32 ONNX parity was synthetic;
PTQ loadability did not establish retained accuracy or integer execution. The
production checkpoint is not a native INT8 checkpoint.

### Located production artifacts and existing browser work

Follow-up inspection on 2026-09-23 used `feature/prototype-browser-inference` at
`b174426a22e42618424fcb0345610ede4a415d01`, without switching or modifying its
worktree. Its [release integration record](https://github.com/asgersvenning/mini_trainer/blob/b174426a22e42618424fcb0345610ede4a415d01/docs/production-release-integration.md)
identifies an already-published production distribution. This release roadmap is
therefore about consolidating and qualifying a successor consumer release, not
locating or publishing those original files for the first time.

Public root: [global-lepi-production-release-20260911T150236Z](https://anon.erda.au.dk/share_redirect/HE90eyuZCT/global-lepi-production-release-20260911T150236Z/index.html).
Paths below are relative to that directory:

| Artifact | Identity / role |
| --- | --- |
| `models/pytorch/best.pt` | Final selected checkpoint; SHA-256 `174b9214bfea2df69e4f5c5d16afd841fec961db4274f3e6bf474cef9cab5e8a` |
| `models/onnx-fp32/model.onnx` | Original prediction graph; SHA-256 `aa02baa22765a04de03c5ba46029e2a66ca7e430bfddce0a001af5cec2e7c15d` |
| `models/onnx-fp32/model.onnx.data` | External tensors; SHA-256 `9ffb389ec4c6fe9864a4dfb16b167cf68950d7fa35b3fa39d84b1987b1845f4e` |
| `models/onnx-ptq/` | Historical experimental artifacts only; preserve in the original archive, exclude from this consumer release |
| `training/`, `evaluation/`, `export/` | Retained configuration, logs, resume state, predictions and export/calibration evidence |
| `provenance.json`, `SHA256SUMS` | Packaging inventory and original file integrity records |
| `viewer/browser-model/model.onnx` | Separate prediction-plus-embedding graph; SHA-256 `70130c3dbc2b8a6bc4610bb213a1aaf029816fb634faf104bbb27ffa997dfc44` |
| `viewer/verification.json`, `viewer/SHA256SUMS` | Browser verification and separate viewer integrity scope |

The source record reports 1,224 original files totaling 25,455,849,324 bytes,
verified ZIP/internal checksums, remote size inventory and selected binary readback.
This follow-up retrieved the public index, README, provenance, original checksum
list, FP32 manifest, browser manifests and verification report. README, provenance
and FP32 manifest bytes matched their original checksum entries. The large weights,
full archive and browser execution were **not** downloaded/reverified in this pass;
the binary hashes above are recorded identities, not fresh binary hash checks.

The FP32 manifest records opset 18, float32 NCHW input `[batch, 3, 384, 384]`, and
ordered outputs of 12,632 species, 4,476 genera and 104 families. Packaging provenance
records checkout `52954edae5dae31a62ecb639533e6f8573d57055` and mini-trainer `0.1.1`,
but explicitly warns these are packaging-time identities, not necessarily training
identities. Recover the latter from retained logs rather than copying this commit.
Earlier epoch-4/epoch-26 explorer checkpoints are not the release checkpoint.

The [separate RC2 browser manifest](https://anon.erda.au.dk/share_redirect/HE90eyuZCT/global-lepi-viewer-20260915-rc2/browser-model/manifest.json)
references the same final checkpoint, graph and tensor hashes as the production
browser bundle. UI publication and model release identity remain separate.
Reuse these implemented branch components after review/integration:

| Existing component | Reuse and remaining boundary |
| --- | --- |
| `mt_export --include-embeddings`, export tests | Opt-in actual prediction outputs plus 1,280-dimensional preclassification embedding; preserve default export behavior and head restrictions |
| `mini_trainer/visualization/prototype_space/browser.py`, `tests/utils/test_browser_bundle.py` | Atomic packaging, checked source hashes/class order, external tensors, pinned ONNX Runtime Web 1.24.3 assets and license; adapt metadata instead of inventing another exporter |
| Browser inference worker and preprocessing | Single-thread CPU WASM and explicit `nearest-square-uint8-bilinear-center-imagenet-v1` recipe, size 384/resize 438; bounded existing contract, not arbitrary transforms |
| `check_inference.mjs`, `check_mobile.mjs`, `check_portable.mjs` | Existing numerical/image, orientation/transparency, touch and static-host checks; broaden evidence only for affected behavior and newly claimed targets |
| Shared GBIF client and packaged names | Optional network enrichment, IDs remain usable offline; packaged names are partial and remote photos are not offline assets |

The public [browser verification report](https://anon.erda.au.dk/share_redirect/HE90eyuZCT/global-lepi-production-release-20260911T150236Z/viewer/verification.json)
records one real-image fixture: identical-input prediction max error about
`1.72e-5`, embedding error `1.80e-7`; end-to-end image prediction error about
`0.00552`, embedding error `0.000250`, with top predictions matching. This is useful
existing evidence, **not** strict end-to-end equality or broad accuracy acceptance.
Preserve that distinction when setting gates; do not restart browser inference as
if absent, or apply its evidence to the original prediction-only graph untested.

The branch's newer image adapter applies EXIF rotation/mirroring and white alpha
compositing; the core reader disables EXIF orientation. Reconcile and version these
policies before claiming one shared image contract. The historical recipe name alone
does not encode this later adapter behavior. Also retain the actual hosting lesson:
ERDA previously lacked `.mjs`/`.wasm` MIME declarations and cross-origin permission;
the verified same-origin deployment uses an unchanged runtime module renamed `.js`
and explicit runtime paths. Current header behavior needs a targeted recheck before
new hosting claims. Model/runtime assets contain executable code; authoring hash
checks do not imply that the current browser worker validates every fetch.

The branch's [qualification record](https://github.com/asgersvenning/mini_trainer/blob/b174426a22e42618424fcb0345610ede4a415d01/dev/prototype_space/portable-qualification.md)
already reports focused Python/browser and installed-wheel checks. Integration,
physical-device review, broader browsers/providers and complete offline installation
remain distinct work. Review/reuse the branch export and deployment changes for A/B;
do not require completion of all explorer UI or global-analysis milestones.

## 1. Freeze identity and the compatibility contract — P0

Produce a compact release inventory before changing inference behavior:

- Start from the identified public distribution above; retrieve and verify the
  required model files against its checksums. Complete the training source and
  package/harness revision audit, and inventory the resolved configuration,
  taxonomy, dataset split identities and evaluation files.
  Confirm that the reported best epoch and the selected weights agree. Record
  hashes and sizes, including every ONNX external tensor file.
- Retrieve the actual previous public weights and hash them. Record the baseline
  as release tag plus weight hash: a historical URL alone is not an immutable model
  identity. Keep original files and releases available for rollback.
- Diff species/genus/family IDs, output order, parent mappings, regional masks,
  preprocessing and score semantics. Report additions, removals and remappings;
  never align arrays by position across models. Do not assume the roughly similar
  class counts mean identical vocabularies.
- Record the old wrapper's constructor/call arguments, accepted inputs, return
  structure, rank/top-k behavior, embedding interface, errors and CLI/CSV fields.
  Define preserved behavior and explicit migration exceptions in a compatibility
  table with tiny fixtures before implementing adapters.
- Propose `MAMBO_v3` as the next model-release name, subject to checking tag
  availability at publication. Give the package version, model ID, artifact
  revision and manifest schema separate identities. An artifact repair may create
  a new revision; it must not replace bytes behind a published version.

Retain Europe as the default for the successor's MAMBO compatibility
interface, matching MAMBO_v2; make `full` and `north_europe` explicit choices.
Version aliases within a release. Do not silently redirect old pinned consumers
to new weights. For the new deployment API, prefer explicit model-bundle selection.
Store regional lists with provenance and hashes, and disclose excluded true labels.

**Done when:** immutable candidate and baseline inventories exist, compatibility
fixtures are specified, and release identity/default decisions are recorded. The
public artifacts and historical weights have now been retrieved and hashed; see
the input-audit increment above. Training-revision/best-epoch provenance and full
wrapper/CLI fixtures remain outstanding before compatibility certification.

## 2. Build a self-contained portable bundle — P0

Reuse the existing public FP32 artifacts where unchanged, and review/integrate the
feature branch's export and browser packaging changes. Extend [the existing ONNX
exporter](onnx.md) and its manifest rather than creating a second exporter. A
release-level manifest can reference its unchanged export manifest and add
deployment metadata with an explicit schema version.

Proposed deployment contents:

```text
release.json                 # identity, hashes, sizes, profiles, compatibility
models/pytorch/best.pt       # original inference weights
models/onnx/model.onnx       # original prediction graph + external tensors
models/onnx-embedding/       # existing floating prediction+embedding derivative
models/*/manifest.json       # source/export metadata for each artifact
preprocessing.json           # complete machine-readable input recipe
classes.json                 # ordered stable IDs, ranks and parent mappings
regions/                     # versioned candidate lists with provenance
conformance/                 # redistributable inputs, tensors, expected outputs
examples/                    # small Python and non-Python usage examples
MODEL_CARD.md
LICENSES/                    # weights, code, runtime and bundled data notices
```

The bundle must describe tensor names/layout/dtype/range, supported batch sizes,
fixed spatial size, graph opset, output structure and actual score meaning. Specify
whether scores are logits, normalized values or probabilities per rank; never add
softmax by guesswork. Include threshold/abstention policy and candidate-filter order.
A bare `repr(preprocess)` or `requires_configuration: true` is insufficient for a
release claiming image-level interoperability.

Specify and test decode, RGB conversion, alpha/grayscale handling, EXIF orientation,
resize geometry, interpolation, antialiasing, crop, scaling and normalization.
The current repository reader disables EXIF orientation; first recover the actual
campaign pipeline. Any changed orientation policy is an explicit versioned behavior
change, not an unnoticed browser/Python discrepancy. JPEG decoders and resize
implementations may differ: preserve the recipe and inspect task-level effects
rather than making universal pixel identity a release gate.

Keep local IDs and taxonomy names sufficient for prediction. GBIF name/photo lookup
is optional enrichment and must not be a hidden inference dependency. Record its
provenance separately from fixed model class identity.

Use ONNX as the initial deployment path that does not require Python checkpoint
unpickling or model constructor downloads. Keep weights-only PyTorch loading for
compatible legacy workflows; do not enable unrestricted pickle loading as an
automatic fallback. New tensor formats are deferred; this increment retains the
tested raw PyTorch and standard ONNX artifacts.

**Done when:** the complete directory can be copied, relocated, checked for integrity
and used offline without the training checkout, constructor downloads or live
taxonomy. The ONNX path must work without PyTorch; the native path uses explicit
PyTorch/backend dependencies. Each backend has its own clean-install check.

## 3. Align the MAMBO_v2 API across backends — P0

Restore `mini_trainer.deploy.Predictor` and `mambo_predict` with MAMBO_v2-compatible
calls and results. Preserve `Predictor()`, `Predictor(model="europe")`, `predict`,
`__call__`, `predict_with_embeddings`, local weight overrides, `class_mask`, top-k
and the existing hierarchy/result accessors. In particular,
`predict_with_embeddings` retains `(predictions, embeddings)`. Keep the native
PyTorch route as the compatibility default; backend selection is additive. Do not
silently change the legacy CUDA default or legacy output types. New portable usage
examples should explicitly select CPU. Document compatibility exceptions before
implementation if tagged fixtures expose a behavior that cannot be retained.

Proposed additive controls (API/CLI spellings to finalize against existing arguments):
`backend="torch"|"onnx"`, `class_list=...`, explicit device, bounded batch size,
thread budget, cache directory and offline mode. Presets continue to work through
`model`/`-M`; keep model revision separate from vocabulary selection internally.
ONNX-only installation must avoid a mandatory PyTorch dependency; reuse a small
runtime-neutral result adapter while preserving the legacy public contract.

| Backend / output mode | Artifact and behavior | Qualification target |
| --- | --- | --- |
| PyTorch, predictions | Original `best.pt`, actual evaluation head | CPU and local CUDA |
| PyTorch, predictions + embeddings | Same weights and preclassification embedding, one backbone pass | CPU and local CUDA |
| ONNX, predictions | Existing standard FP32 prediction graph | CPUExecutionProvider and local CUDAExecutionProvider |
| ONNX, predictions + embeddings | Existing floating prediction+embedding graph from browser work | CPUExecutionProvider and local CUDAExecutionProvider |

Every row supports the same presets and custom lists. The prediction-only ONNX graph
has no embedding output: select the identified embedding-enabled graph explicitly
when requested, without silent PyTorch fallback or synthetic embeddings. Exporting a
replacement is necessary only if a recovered artifact cannot satisfy the contract.
Do not assume that omitting an output fetch removes its computation; benchmark the
actual graph chosen. Keep returned embedding stage, sample order, dimensions and
normalization consistent; do not add a second backbone pass. Expose conversion/copy
costs and output device/type policy while retaining legacy behavior.

### One vocabulary and postprocessing contract

Use stable species IDs and versioned preset files. Allow a custom UTF-8 class-list
file and an equivalent Python sequence. Define `full` as all classes in this
checkpoint; preserve old preset membership where available, report missing IDs and
version deliberate additions separately. Species absent from the model cannot be
added by a list. Deduplicate, preserve model order rather than caller order, report
unknown entries, and reject an empty overlap before processing images.

A custom list explicitly replaces a named preset for the new `class_list` option;
record the resolved list and hash. Preserve legacy `class_mask` semantics separately,
including reset behavior, and reject simultaneous `class_mask` and `class_list` as
ambiguous. A pre-masked artifact cannot recover absent classes. Keep independent
predictors isolated so changing one filter cannot affect another.

Filtering must precede ranking and confidence normalization, with genus/family
scores recomputed from retained leaves. In the current hierarchical head, parent
scores use grouped log-sum-exp. For standard ONNX, gather the retained leaf scores
and apply the same hierarchy aggregation and normalization in shared postprocessing;
slicing the existing full-vocabulary parent outputs is incorrect. Cover priors,
parent mappings, masks and ordering with small behavioral fixtures against native
PyTorch. Unsupported head semantics fail explicitly. This avoids exporting one
graph per custom list and keeps presets as metadata, without changing model weights.

Use the tested campaign preprocessing for both Python backends. Browser EXIF/alpha
improvements stay an explicitly separate adapter until deliberately aligned; they
must not silently change this release's native/ONNX image pipeline. Share sample
identity, hierarchy metadata, top-k and score conventions, thresholds, errors and
collector schema. Different backends need not have bit-identical scores or ranking
for near ties. Reuse the JavaScript work later as another consumer of this contract.

**Done when:** tagged MAMBO_v2 compatibility fixtures and tiny backend × output-mode
× class-list tests pass; all four rows run bounded batches and return aligned
results. Include custom/preset equivalence, unknown/empty/duplicate lists, mask
reset, prediction-only versus embedding-enabled behavior, and finite correctly
shaped embeddings. No new micro-numerical ONNX validation study is required.

## 4. Qualify deployment profiles and restricted operation — P0/P1

These are proposed test targets, not current support claims. Record exact hardware,
OS, architecture, runtime/provider versions, driver where applicable and evidence
for every claimed profile. “ONNX compatible” is not a qualification result.

| Priority/profile | Initial target | Required evidence / fallback |
| --- | --- | --- |
| P0 local CPU | Laptop Intel Core i7-12800H, x86-64; PyTorch and ONNX | Both output modes, task metrics, clean install, offline/read-only behavior, fixed thread budgets and RAM |
| P0 local GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16 GiB; PyTorch CUDA and ONNX CUDA provider | Both output modes, task metrics, latency/throughput/VRAM and actual provider placement |
| P1 other desktops | Windows x64 and macOS arm64 CPU | Same contracts and install/restriction fixtures on actual OS/hardware; support claims only after checks |
| P1 edge ARM | Linux aarch64 CPU | Target RAM/latency and runtime availability; reduced batch profile |
| P1 browser | Existing Chromium WASM implementation, then other browsers/devices | Reuse existing evidence; new preprocessing/hosting claims separately qualified |
| Later release | Quantization, FP16 graph conversions, TensorRT and other providers | No implementation, packaging or qualification in this release |

The laptop GPU/CPU were queried on 2026-09-23; the GPU reports driver 610.47.
The ordinary sandbox blocked NVML, while the permitted host query succeeded.
This identifies the available hardware, not successful PyTorch/ONNX CUDA execution.
Preflight actual runtime/provider availability before benchmarking, using the
existing environment without implicit synchronization. Prepare an isolated GPU
runtime environment if needed; do not replace the working CUDA wheels.
Record actual OS/kernel/virtualization and effective CPU affinity in results; this
host is not evidence for every Linux or Windows deployment. Other OS targets remain
part of the portability roadmap, not prerequisites for this local comparison.

ONNX Runtime offers multiple [execution providers](https://onnxruntime.ai/docs/execution-providers/),
but availability and operator coverage must be checked against the pinned runtime.
Measure actual placement; selecting a provider name does not prove the whole graph
runs there. Build ordinary TensorRT engines for a declared target configuration and
retain ONNX as the interchange artifact; [TensorRT describes these as hardware-specific
engines](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html).
Any compatibility mode needs its own evidence and runtime constraints.

Treat restrictions as independent test cases, not just another operating system:

| Restriction | Required behavior and test |
| --- | --- |
| No outbound network / air gap | Explicit prefetch or manual transfer; verify locally before load; no automatic model, GBIF, font, CDN or telemetry requests; test with egress disabled |
| No root / no containers / no compiler | Prebuilt runtime installation in a user environment; offline dependency set for each claimed OS/architecture; container is an optional delivery format |
| Read-only installation and weights | Inference reads only; explicit writable cache/temp/output paths when needed; operation with caching disabled; never require writes beside weights |
| Proxy / internal mirror | Configurable approved artifact source and CA trust; serial-download fallback when HEAD/range requests fail; retain TLS verification |
| Interrupted or concurrent download | Hash and size verification, bounded retries, atomic cache publication and locking; incomplete files cannot count as valid cached models |
| No subprocesses / restricted threads | In-process path and explicit single-thread/zero-worker settings; resource limits reported rather than guessed from host core count |
| Restricted browser | Same-origin runtime assets and documented minimal CSP; test without cross-origin isolation, blocked GPU and denied storage; actionable unsupported-policy error |
| Integrity and supply-chain policy | Pinned dependencies, license inventory/SBOM, release manifest checksums and authenticated provenance; keep credentials/raw user images out of artifacts and logs |

For browsers, WASM multithreading needs cross-origin isolation; single-thread mode
is available. The proxy worker uses Blob and can conflict with restrictive CSP;
qualify an external same-origin worker where needed. JS and WASM files must come
from the same build. See [runtime configuration](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html).
WebGPU requires a secure context, and deployment must include the required runtime
assets; see [web deployment](https://onnxruntime.ai/docs/tutorials/web/deploy.html).
Do not promise local `file://` execution. A policy forbidding WASM entirely cannot
be fixed by a WASM fallback: offer the native client or an explicitly chosen service.
Do not upload images to a service as an automatic fallback.

Checksums detect corruption but do not authenticate a publisher. Define a trusted
release channel and, where required, signed provenance verifiable with offline
trust material. Review model/runtime inputs and archive paths before extraction;
keep ONNX external-data references inside the verified bundle. These are concrete
release-loader requirements, not a claim that any model format is risk-free.

## 5. Measure in-domain/Flemming quality and local inference cost — P0

### Evaluation inputs and reusable machinery

Use the supplied in-domain test split (632,913 images in the campaign) and the
Flemming camera-trap expert set (58,640 images, 522 species). Verify recovered
manifests/counts and identity before associating local paths with those datasets.
Preserve all labels and original splits; never drop excluded or unknown species
when applying a regional/custom candidate list. No random re-splitting or test-based
threshold selection. Existing archived PyTorch predictions are a reference only
when their weights, preprocessing, precision, list and sample identities match.

Start with the published `evaluation/` reports/CSVs to establish the baseline and
locate the original image/staging manifests. Archived predictions can reproduce
metrics without image access, but cannot supply new ONNX predictions or end-to-end
speed. Resolve local data roots or explicitly stage a bounded selection before the
first run. Do not silently substitute a different dataset for missing Flemming data.

Reuse [the inference benchmark modules](../dev/benchmarks/inference.md):
`prepare_inputs` for ordered identity-bearing batches, `dataset_inference` for ONNX
collection, and `quality_compare` for metric comparisons. The current collector
supports ONNX/TensorRT, **not native PyTorch**: add the small native adapter using the
release predictor and extend shared postprocessing for filters/embeddings. Preserve
raw-image streaming for large datasets instead of requiring all decoded images or
embeddings in RAM. Prepared tensors may isolate runtime costs but must not replace
the image-to-result benchmark. `quality_compare` currently requires identical class
mappings; use it for matched new-model variants, with separate ID-aligned reporting
for MAMBO_v2 or different-vocabulary comparisons.

Keep the existing canonical `mini_metric.csv` output for compatibility. If using
benchmark modules' long-form tables, provide a tested conversion or their existing
metric route; do not assume the two schemas are interchangeable. Run metrics in a
separate prepared Python 3.13 environment at the campaign's mini_metrics revision
`70cc69adc05362863439277048e06386c1f885e1`, with resolved dependencies recorded.
The existing helper supports this concrete path once each variant has completed
both canonical prediction files:

```bash
MT_TEST_CSV=/path/to/variant/indomain/mini_metric.csv \
MT_EXPERT_CSV=/path/to/variant/flemming/mini_metric.csv \
  bash dev/ucloud/evaluate-results.sh all /path/to/fresh/variant-metrics
```

The helper calls the Flemming dataset `expert` and records all-label, known-only and
per-class reports, hashes and completion markers. Initial `uvx` dependency setup
needs networking; pre-provision/pin the metric environment for offline runs. It does
not change the training environment. Selected-prediction CSVs cannot establish top-5
accuracy; retain top-k output explicitly if reporting that metric.

### Bounded comparison matrix

1. Run tiny contract checks across both backends, both output modes and all three
   presets plus one representative custom list; include preset-as-custom-list
   equivalence. Run on CPU and the laptop GPU. This is behavioral coverage, not
   a full-dataset Cartesian product.
2. Freeze a reproducible, bounded qualification subset from each original test
   dataset for all eight backend × embedding × device configurations, using `full`
   and `europe` first. Select by a recorded seed/ID list, preserve unknown labels,
   record class coverage and label subset metrics as such. Choose the count after
   a brief throughput probe so this first comparison is practical on the laptop.
3. Collect full in-domain and full Flemming predictions for native PyTorch and
   standard ONNX on one selected qualified device, initially `full` and `europe`.
   Reuse matching completed native runs. Evaluate `north_europe` and the representative
   custom list from retained full leaf scores through the shared reducer where
   semantics permit; otherwise make explicit additional inference runs. Record
   coverage, avoid choosing lists from test outcomes, and retain bounded score
   shards only when their reuse justifies disk cost.
4. Check embedding-enabled and CPU/GPU variants on the same qualification samples.
   Extend their quality run only if task-level differences or a changed pipeline
   require it. Do not claim separate full-dataset evidence for a variant tested
   only on the subset. Report the embedding mode's measured time/memory overhead.
5. Include MAMBO_v2 as the historical consumer/model reference, using matching images
   and each model's own preprocessing. Report both common-vocabulary and all-label
   results. A backbone change is not automatically an accuracy improvement.

Report per-rank micro accuracy, Macro-F1, Macro-Recall, Macro-Precision, Coverage and
Theil's U, plus all-label/known-only and per-class results. Keep sample counts,
active-list coverage and abstention coverage separate. Preserve undefined metrics.
Flemming's archived species accuracy (57.59% all-label, 66.74% known-only) is context,
not an acceptance threshold for every list/variant. Keep raw/unthresholded results;
any operational thresholds are fixed from separate validation/calibration data.

Accept small numerical differences. Compare aggregate task metrics, prediction
agreement and material threshold/coverage changes; investigate substantive regressions,
not every score delta. Keep finite-value, shape, sample-order, class-ID and hierarchy
checks. Do not demand identical scores or perfect top-1 agreement on near ties, and
do not create new max-absolute-error release gates. Preserve existing exporter tests
and record prior parity evidence without rerunning a micro-numerical study.

### CPU/GPU speed and resource protocol

Use the same laptop, inputs and declared list/output settings for paired measurements.
Measure PyTorch CPU/CUDA and ONNX CPU/CUDA, with and without embeddings. Begin with
FP32 for a matched baseline; additionally retain the MAMBO-compatible native CUDA
autocast behavior as a clearly labelled practical mode. Record actual dtype,
autocast/TF32 settings and runtime versions; do not compare mixed precision as if
precision were matched. No new FP16 ONNX conversion is required.

- Separate model/session load, first prediction and steady-state work. Measure both
  complete image-to-consumer-result time (decode, resize, transfer, postprocess,
  embedding copies included) and prepared-tensor runtime time with its boundary
  stated. Do not present only kernel timing as application speed.
- Start with batch 1 for interactive latency, then a small common batch sweep such
  as 8 and 32, stopping at the memory budget. Report the best practical batch per
  variant separately from matched-batch comparisons. OOM is a recorded capacity
  result, not permission for a silent batch/provider change.
- Use explicit CPU thread counts (one and a fixed practical allocation), identical
  decode-worker budgets and bounded streaming. For GPU timing wait for completed
  work through correct synchronization or completed host outputs; include transfer
  in end-to-end measurements. Verify ONNX provider placement/fallback.
- Use fresh processes for load and memory measurements. After explicit warmup,
  collect repeated timings in at least three alternating-order trials; report
  median/p95 latency, images/s, peak RSS, peak/observed VRAM with measurement method,
  failures, and variance. Avoid concurrent heavy jobs; record power mode, plugged-in
  status and thermal/throttling observations. These are laptop-specific results.
- Measure `full` versus a preset and embeddings on/off. A class list applied after
  ONNX execution does not reduce backbone/graph work; any native head speed benefit
  or ONNX postprocessing overhead must be measured rather than inferred.

Write one comparison table keyed by artifact hash, backend, device, precision,
embedding mode, list hash and dataset/split hash. Include quality deltas, counts,
latency/throughput, resource cost and evidence scope. Retain commands/configs,
prediction files, metric reports, raw trial timings and completion/failure markers.
Recommend defaults from the measured quality/speed/memory trade-off, allowing users
to choose a slower compatible or more restricted-environment-friendly route.

**Done when:** a reproducible runner and concise report cover in-domain and Flemming
metrics, CPU/GPU costs, both backends and both output modes, with preset/custom-list
behavior tested and subset/full-data evidence distinguished. Material quality or
behavioral failures are resolved; small ONNX numerical variation is accepted.

## 6. Package, stage and promote — P0

Keep the GitHub release as the primary discovery and migration entry point,
consistent with previous releases. Inventory artifact sizes before choosing GitHub
assets versus an immutable ERDA/object-store location; publish verified URLs and
hashes in either case. A Hugging Face mirror is useful for discoverability, but
must contain the same identified artifacts and must not become an inference-time
requirement. Follow its [model-card metadata format](https://huggingface.co/docs/hub/model-cards)
if using the Hub; hosting is separate from deploying an inference service.

Publish standard ONNX and native PyTorch inference assets, evaluation evidence
and training-resume archives. Include license and redistribution review for the
weights, backbone, runtime, taxonomy and example images; do not infer a weight/data
license from the repository's code license. Preserve evaluation provenance without
shipping private paths, access tokens or the full 23.71-GiB campaign directory.

Stage a prerelease with immutable versioned artifact names, migration guide, model
card, measured support matrix, known limitations, checksums/provenance and tested
installation commands. Download it as a consumer would, verify bytes, install in
clean target environments, and exercise offline prediction. Reuse unchanged
qualification evidence; repeat only packaging/readback and affected checks.

Before public promotion, present the concrete candidate, compatibility changes,
quality comparison, supported targets and rollback instructions for release review.
Promotion changes only the declared release/default pointers after that review;
retain the old model and pinned installation path. Never overwrite MAMBO_v2 or the
existing public viewer. A later rollout problem should be recoverable by selecting
the previous model revision without changing the consumer's data or environment.

**Done when:** published assets can be retrieved and verified, documented examples
run against them, the stable pointer identifies the reviewed candidate, and rollback
has been exercised. Release notes distinguish model changes from package/API changes.

## Bounded implementation sequence

| Increment | Concrete deliverable | Dependency / completion gate |
| --- | --- | --- |
| A — identity and compatibility | Verify existing artifacts; recover MAMBO_v2 API/preset fixtures and data identities | Known public distribution, tagged code and evaluation manifests |
| B — aligned inference | Native PyTorch + standard ONNX; presets/custom lists; predictions ± embeddings | A; shared preprocessing, hierarchy reduction and installed API checks |
| C — quality and local cost | Reusable variant runner; in-domain/Flemming metrics; laptop CPU/GPU speed and memory | B; bounded matrix first, then needed full-dataset comparisons |
| D — staged release | Consumer bundles, migration notes, measured trade-offs, offline checks and rollback | A–C; concrete reviewed candidate |
| E — broader portability | Additional OS/browser profiles and distribution channels | Core release preserved; qualify only new boundaries |

Start with **A and B**, then use C to make the release recommendation concrete.
The next-training-run orchestration plan and experimental quantization are not on
this release's critical path.

Open work during A: verify candidate binaries and historical weights; locate local
in-domain/Flemming images and manifests; recover training revision and actual preset
memberships; pin the compatible runtime/metric environments. Final package/version
and publication choices follow the measured candidate. CPU/GPU qualification uses
the identified laptop; additional machines are needed only for later support claims.
