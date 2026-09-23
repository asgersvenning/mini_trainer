# UCloud model release roadmap

Status: proposed development roadmap, 2026-09-23. This document does not publish
artifacts or claim deployment qualification. Target: the completed 10–11 September
2026 UCloud model, not a new training campaign.

## Release objective and scope

Ship a versioned successor to the public MAMBO deployment release that is easy to
install, embed and operate without a GPU, internet access, administrator rights or
a writable installation directory. Preserve a clear migration path for existing
Python and CLI users. Add acceleration only through separately qualified profiles.

The first release slice is an immutable FP32 ONNX deployment bundle, a small CPU
inference interface, legacy-interface compatibility, and measured release evidence.
Keep original PyTorch weights available for established workflows. Training resume
state and the large research archive are separate optional downloads. Hosting on
Hugging Face, browser integration and additional accelerators build on that same
bundle; they must not each invent preprocessing or taxonomy conventions.

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
| [MAMBO_v2](https://github.com/asgersvenning/mini_trainer/releases/tag/MAMBO_v2), 17 April 2026 | Latest published release; BioCLIP-2 model; `mambo_predict`; `mini_trainer.deploy.Predictor`; `full`, `europe`, `north_europe` aliases; default region Europe; weights downloaded from ERDA | Primary migration baseline. Preserve old version pins, document architecture and vocabulary changes, and explicitly decide the successor's default |
| [MAMBO_v0](https://github.com/asgersvenning/mini_trainer/releases/tag/MAMBO_v0), 15 April 2026, prerelease | Earlier MAMBO deployment wrapper and northern-European model emphasis | Include older pinned consumers in the migration guide |
| [UKCEH_v0](https://github.com/asgersvenning/mini_trainer/releases/tag/UKCEH_v0), 3 February 2026 | Northern-European EfficientNetV2-M model; `python predict.py`; automatic model download | Document migration from script invocation and its original regional vocabulary |

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
identities. This planning pass has not independently opened the training archive.
FP32 ONNX parity was synthetic; PTQ loadability did not establish retained accuracy
or integer execution. The production checkpoint is not a native INT8 checkpoint.

## 1. Freeze identity and the compatibility contract — P0

Produce a compact release inventory before changing inference behavior:

- Locate the retained archive, verify its checksum, and identify the selected
  inference checkpoint, training source revision, package/harness revisions,
  resolved configuration, taxonomy, dataset split identities and evaluation files.
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

Retain Europe as the proposed default for the successor's MAMBO compatibility
interface, matching MAMBO_v2; make `full` and `north_europe` explicit choices.
Version aliases within a release. Do not silently redirect old pinned consumers
to new weights. For the new deployment API, prefer explicit model-bundle selection.
Store regional lists with provenance and hashes, and disclose excluded true labels.

**Done when:** immutable candidate and baseline inventories exist, compatibility
fixtures are specified, and release identity/default decisions are recorded. Missing
archive or weight access blocks certification, not drafting the remainder of the plan.

## 2. Build a self-contained portable bundle — P0

Extend [the existing ONNX exporter](onnx.md) and its manifest rather than creating a
second exporter. A release-level manifest can reference its unchanged export
manifest and add deployment metadata with an explicit schema version.

Proposed deployment contents:

```text
release.json                 # identity, hashes, sizes, profiles, compatibility
model/model.onnx             # plus ALL referenced external tensors
model/manifest.json          # existing export metadata and numerical verification
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
implementations may differ: retain intermediate tensors and justified tolerances
rather than promising universal pixel identity.

Keep local IDs and taxonomy names sufficient for prediction. GBIF name/photo lookup
is optional enrichment and must not be a hidden inference dependency. Record its
provenance separately from fixed model class identity.

Use ONNX as the initial deployment path that does not require Python checkpoint
unpickling or model constructor downloads. Keep weights-only PyTorch loading for
compatible legacy workflows; do not enable unrestricted pickle loading as an
automatic fallback. Evaluate safetensors plus explicit construction metadata only
if consumers need a portable tensor checkpoint; that is not already implemented.

**Done when:** the complete directory can be copied, relocated, checked for integrity
and used offline by a clean CPU process without the training repository, PyTorch,
backend downloads or a live taxonomy service.

## 3. Deliver small integration interfaces — P0

Build on existing prediction, class-filtering and result-collection boundaries.
Keep deployment dependencies optional; do not force the training stack into an
ONNX-only consumer. Decide the smallest package boundary after inspecting those
imports, and validate its installed distribution outside the checkout.

- Provide a documented Python image-to-prediction example using ONNX Runtime and
  an explicit decoder, plus a tensor-in/tensor-out example. Add one non-Python
  consumer, preferably JavaScript from the existing viewer work, using identical
  conformance data. Additional language SDKs can follow demonstrated demand.
- Restore a thin `mini_trainer.deploy.Predictor`/`mambo_predict` compatibility layer
  or ship an explicitly versioned migration package. Preserve documented result
  contracts, masking and embedding access where promised; embedding dimensions
  and coordinates from different backbones are not interchangeable.
- Bound batching and memory. Support explicit batch/thread limits, local bundle
  paths and clear per-input error handling. Separate library results from logging;
  expose model revision, active vocabulary and selected runtime in provenance.
- Make device selection predictable: portable API defaults to CPU or a documented
  capability-based mode; explicit accelerator requests fail clearly when unmet.
  Allow CPU fallback only under a declared policy and report its use. Do not
  silently mutate existing core CLI defaults during this release work.
- Supply copy-pastable pinned installation and prediction commands for POSIX shells
  and PowerShell. Test paths containing spaces and Unicode. Avoid requiring a
  repository clone, shell bootstrap script or administrator install for inference.

**Done when:** an existing MAMBO consumer has a tested migration example and a new
consumer can run one image, a bounded batch and an empty/error case from an installed
package or standalone example. Core package imports remain independent of optional
runtime integrations.

## 4. Qualify deployment profiles and restricted operation — P0/P1

These are proposed test targets, not current support claims. Record exact hardware,
OS, architecture, runtime/provider versions, driver where applicable and evidence
for every claimed profile. “ONNX compatible” is not a qualification result.

| Priority/profile | Initial target | Required evidence / fallback |
| --- | --- | --- |
| P0 portable CPU | Linux x86-64 and Windows x64, CPU FP32 | Clean install, real-image conformance, bounded threads/RAM, offline/read-only operation; record tested CPU instruction requirements |
| P0 desktop ARM | macOS arm64, CPU FP32 | Native hardware check with the same fixtures and resource measurements; do not infer support from Linux x86 |
| P1 edge ARM | Linux aarch64, CPU | Actual target RAM/latency and runtime wheel availability; reduced batch profile; exclude untested devices from support claims |
| P1 NVIDIA | Linux/Windows CUDA on selected supported driver/runtime combinations | Operator placement and transfer profiling, quality parity, cold/warm latency and VRAM; explicit CPU fallback policy |
| P1 browser | Existing static viewer, desktop Chromium first; then Safari/Firefox and physical phones | Self-hosted JS/WASM/model assets, fixed preprocessing, memory/startup, single-thread fallback; WebGPU separately qualified |
| P2 specialized accelerators | TensorRT, CoreML, OpenVINO, DirectML/WebNN or other requested provider | Add one at a time only for a consumer need and measured benefit; retain the portable baseline |

The proposed first GA scope is the three P0 desktop/CPU targets. If hardware access
is unavailable, either complete qualification before claiming that target or narrow
the published support matrix explicitly. Experimental profiles do not hold up a
correctly scoped portable release.

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

## 5. Establish quality and efficiency gates — P0, then P1 variants

Use three distinct comparisons: previous public model versus the new model; the
selected new PyTorch checkpoint versus portable FP32 ONNX; portable FP32 versus
any optimized variant. Do not confuse export parity with model improvement.

1. Retain supplied train/validation/test assignments and taxonomy. Freeze sample
   identities and hashes. Recover the existing full test/expert predictions where
   valid rather than rerunning expensive work without a question to answer.
2. Compare old/new models on matching images using stable taxon IDs. Report the
   common-vocabulary slice and the full consumer workload, with excluded/unseen
   labels visible. Stratify by rank, region and rare classes. Use the pinned
   [mini_metrics workflow](../dev/ucloud/evaluate-results.md); preserve metric
   definitions and distinguish macro recall from other “macro accuracy” measures.
3. Run raw-image conformance through decoding, preprocessing, graph and decoding of
   outputs, including batches 1, 2, 4 and a declared upper bound, grayscale/alpha,
   EXIF, unusual aspect ratios, and malformed/oversized inputs. Use redistributable
   fixtures; keep private evaluation images outside the public bundle.
4. Predeclare per-profile score tolerances, top-1/top-k agreement, per-rank quality
   limits, coverage and resource budgets before accepting a candidate. Record
   max/percentile errors, near-tie changes, nonfinite outputs and threshold crossings.
   The campaign used `rtol=1e-4, atol=1e-4` for synthetic FP32 parity; do not silently
   replace exporter defaults or treat that result as end-to-end qualification.
5. Fit thresholds/calibration on suitable validation data, then freeze for test.
   Previously optimized test thresholds remain exploratory. Regional filtering
   changes scores, so global thresholds are not automatically transferable.
6. Measure download/bundle size, session creation, first image, warm p50/p95 latency,
   throughput at stated batch/concurrency, peak RSS/VRAM and thread count. Use a
   bounded representative workload on each target; compare on the same device.

The expert-set result in the campaign record (57.59% species micro accuracy across
all labels versus 66.74% known-only) must remain visible in the model card. Strong
in-domain scores do not establish universal field accuracy or open-set rejection.
Audit overlap/leakage and dataset provenance before claiming improvement over older
releases; mark gaps explicitly when historical training identities are unavailable.

FP16 and PTQ are optional named derivatives with their own hashes, recipe, quality
report and qualified profiles. The existing 128-image PTQ artifact is a candidate,
not the default. Quantization must earn inclusion through retained quality and a
measured latency/memory benefit; no requirement to ship INT8 merely because it exists.

**Done when:** a release report records pass/fail/untested per gate and profile,
with thresholds and evidence attached. Numerical or resource failures are resolved
or the affected profile is excluded; no silent widening of tolerances.

## 6. Package, stage and promote — P0

Keep the GitHub release as the primary discovery and migration entry point,
consistent with previous releases. Inventory artifact sizes before choosing GitHub
assets versus an immutable ERDA/object-store location; publish verified URLs and
hashes in either case. A Hugging Face mirror is useful for discoverability, but
must contain the same identified artifacts and must not become an inference-time
requirement. Follow its [model-card metadata format](https://huggingface.co/docs/hub/model-cards)
if using the Hub; hosting is separate from deploying an inference service.

Publish separate portable runtime, optional PyTorch inference, evaluation evidence
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
| A — identity and migration | Candidate/baseline inventory; vocabulary and API diff; release/default decisions | Archive and historical weight access; section 1 |
| B — portable vertical slice | FP32 bundle, executable preprocessing, tiny CPU client and real-image fixtures | A; relocated offline inference in a clean Linux environment |
| C — consumer compatibility | MAMBO adapter, bounded batches, explicit cache/download policy, pinned installs | B; tagged-interface fixtures and clean installed-package checks |
| D — release qualification | Windows/macOS CPU checks, restriction matrix, old/new quality report and frozen gates | B/C; publish only evidenced support and quality claims |
| E — staged release | Versioned prerelease, readback, model card, migration and rollback; promotion review | A–D; concrete release checklist and evidence |
| F — optional acceleration | One selected GPU/browser/quantized profile at a time | Portable baseline; demonstrated demand and quality/resource benefit |

Start with **A and B**. They resolve the main uncertainty—exactly which bytes and
input/output contract are being released—and produce a reviewable usable artifact.
The next-training-run orchestration plan is not on this release's critical path.

Open decisions to resolve during A: retained archive location and checksum; exact
previous weight identities; target consumer examples; final release/package versions;
artifact host; numerical/quality/resource budgets; and access to Windows/macOS target
machines. This roadmap proposes defaults where possible but does not invent evidence.
