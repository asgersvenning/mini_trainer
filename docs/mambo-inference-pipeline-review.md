# V2/V3 inference pipeline review

Review date: 25 September 2026. Code baseline: `0eb90b2`; V2 source:
`32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`. The review below records the original design baseline; the
[implementation follow-up](#implementation-follow-up) records subsequent changes. Evidence is the completed UCloud campaign and source
inspection. Subsequent B200 measurements are recorded in the implementation follow-up.

The concern is substantially justified: several V3 changes restored efficiency
lost by the initial portable adapter, and the streaming implementation accumulated
coordination mechanisms around an expensive CPU/NumPy boundary. The next change
should replace that structure, not add another independent queue or worker knob.
This does not mean reverting release features or copying V2 wholesale.

## What the measurements establish

Global vocabulary; median of three process trials, images/s:

| Pipeline | CPU request B=8 | B200 request B=32 | B200 request B=256 | B200 streaming B=256 | Prepared-input B=256 |
|---|---:|---:|---:|---:|---:|
| V2 | 2.0 | 539.5 | Not tested | Not tested | Not tested |
| V3 Torch | 28.7 | 480.7 | 473.3 | 624.9 | 2,957.3 |
| V3 ONNX | 35.1 | 442.2 | 439.5 | 548.4 | 2,243.8 |

Sources: [request observations](assets/mambo-indomain-speed.csv),
[streaming observations](assets/mambo-indomain-streaming-speed.csv), and the
raw benchmark reports in `local-evidence/ucloud-2026-09-25/mambo-results/runs-transfers/`.
Prepared-input diagnostics include transfers and model execution, exclude image
preparation and public result processing, and are not a pure GPU-compute ceiling.
The Torch head still computes hierarchy inside that diagnostic; the report's
"no decode/reduction" wording is too broad for Torch and means no adapter-side
result reduction. Streaming measures only 1,024 images including startup.
CPU tests use four runtime threads, not a tuned full 48-vCPU node.

At the common GPU batch size, V3 Torch is about 11% slower than V2 and ONNX 18%
slower. V3's separate streaming path surpasses V2's request rate, but at a larger
batch and with a different execution mode. That is not evidence of a controlled
backend speedup. Conversely, the CPU improvements are real measured pipeline
improvements; it is inaccurate to say that every V3 gain merely recovered a V2
regression. V2 and V3 also differ in backbone, input resolution, preprocessing,
and runtime precision. There is no matched V2 prepared-input benchmark to infer
their isolated model speed ratio from this campaign.

The full Torch single-view collection provides stronger pipeline evidence than
GPU utilization samples:

| Recorded quantity | Seconds |
|---|---:|
| Entire collection, including setup | 875.7 |
| Consumer waiting for staged input | 549.3 |
| Runtime call wall time | 265.2 |
| Model CUDA-stream event intervals | 212.9 |
| Background assembly wall time | 647.2 |
| H2D CUDA-event intervals | 27.5 |
| D2H CUDA-event intervals | 1.36 |
| Background prediction construction | 121.4 |
| Background CSV writing | 11.2 |

These overlap and must not be added. Runtime wall time includes device completion;
model event intervals can include device idle gaps while the host submits work.
Sampled GPU utilization is neither pipeline time attribution nor SM occupancy.
The dominant observed single-view problem is delivery of ready batches, not
bulk PCIe output bandwidth or CSV storage. The latest H2D staging change did not
produce a demonstrated end-to-end improvement on B200.

An unresolved TTA signal is important: Torch model event time is 2,524.5 seconds
for three views, versus 212.9 seconds for one view, while its input wait falls to
6.7 seconds. ONNX TTA instead records 1,750.7 seconds waiting for input and 768.6
seconds in runtime. Similar overall throughput can therefore hide different
critical paths. Host launch starvation, scheduling, allocator contention and
actual GPU execution need distinguishing; the event totals alone do not prove a
fourfold kernel slowdown beyond the three-view multiplier.

## V2 versus V3: where the work moved

The inspected V2 `mini_trainer/deploy.py:Predictor` uses the reader in
`mini_trainer/utils/io.py:make_read_and_resize_fn` and native classifier prediction.

| Responsibility | V2 deployment | Current V3 deployment |
|---|---|---|
| Reading | torchvision decode, CPU resize to 512 square, uint8 transfer per image | PIL decode; portable NumPy preparation; optional independent reading pool |
| Model preprocessing | Batched checkpoint transform on the device for path inputs; final input 224 square | Per-image CPU interpolation, normalization and CHW float32 materialization; final input 384 square |
| Batching | Caller supplies one batch; device tensors stacked | Request batching plus a separate streaming scheduler and host assembly pool |
| Precision | Existing CUDA autocast | Initially FP32; later restored backbone AMP, with FP32 head |
| Hierarchy | Existing batched native reduction | Initially discarded/recomputed on CPU; now native Torch ranks or batched NumPy for ONNX |
| Selection/confidence | Torch top-k and softmax on device | All rank logits downloaded; NumPy top-k/softmax; eager Python result objects |
| Optional features | Masks and embeddings already available | Expanded preset provenance, custom lists, portable ONNX, configurable TTA, bounded streaming |

V2 was not an ideal asynchronous engine: request reading is serial, transfers
are per image, and label translation uses GPU scalar `.item()` operations. Its
full evaluation harness also differs from the public API, using four reading
threads and shared backbone features. Do not conflate that full-run rate with
the public request benchmark.

Nevertheless, V2 kept batched work in the native runtime. V3's portability layer
moved substantial numerical work back to CPU and imposed CPU-return boundaries.
Portability requires matching semantics, not matching every backend's physical
execution location. Restoring AMP and native hierarchy were regression repairs,
not novel HPC optimizations. TTA, audited presets, portable ONNX and standalone
CPU integration are useful features worth retaining independently of speed.

## Concrete architectural problems

### 1. Preparation performs avoidable work before concurrency can help

[preprocessing.py](../deployment/mambo_deploy/preprocessing.py) reconstructs fixed
interpolation coordinates and normalization constants for every image. More
significantly, subtracting int64 `lo` indices from float32 coordinates produces
float64 fractions; both bilinear passes consequently use float64 intermediate
arrays. The chained indexing `image[:, yy][:, :, xx]` also materializes a
source-width intermediate before selecting columns. Several full image arrays
are allocated for arithmetic and layout conversions.

This is verified directly with the installed NumPy. A small isolated laptop
probe on an already-decoded synthetic 2000x3000 image compared the current
function with cached float32 coefficients, direct two-axis gathering and
in-place normalization. At four threads the probe increased from 235 to 444
images/s and traced single-call peak allocation fell from 16.0 to 8.9 MB.
At 48 threads the corresponding rates were 209 and 395 images/s. This is a
cost/mechanism experiment, not a deployment speed or quality qualification;
32 calls per observation also do not fully occupy 48 workers. The local probe
and output are retained in `local-evidence/pipeline-review/`. It is not a proposed
second production implementation. Its lesson is that fewer allocations and
operations can improve both work and contention without another scheduler.

Torchvision explicitly recommends tensor transforms, uint8 resizing and attention
to memory layout. Reuse native transforms when they implement the release recipe;
do not silently substitute PIL/torchvision interpolation, antialiasing or transform
order and assume the model input is unchanged. Small rounding differences are not
the gate; unintended geometry/normalization changes are. [Torchvision guidance](https://docs.pytorch.org/vision/stable/transforms.html#performance-considerations).

TTA additionally copies and rotates/pads full-resolution originals before model
resizing, even for the identity transform. This is much more expensive for the
large in-domain photos than Flemming crops. Reordering resize and rotation is a
recipe change and should not be smuggled into a pipeline refactor.

### 2. Too many ownership transitions, despite bounded queues

[streaming.py](../deployment/mambo_deploy/streaming.py) maintains in-flight reads,
encoded buffers, decode futures, per-image results, assembling batches and completed
batches. A coordinator scans futures, sorts ready indices and wakes every 5 ms.
Filesystem `stat()` runs on that coordinator, so slow metadata can also delay
handling completions. There is a single stacking executor after per-image
preparation, followed by another transfer executor and a result executor.

At batch 256, one V3 float32 input is **432 MiB**. The recorded nine allocated host
batch buffers represent about **3.8 GiB**; three-view TTA's 27 allocations represent
about **11.4 GiB**, before all decoded originals, intermediates and other results.
The 2 GiB encoded-byte budget does not bound those allocations. A per-image
prepared count obscures the actual batch and view byte footprint.

Background assembly takes roughly 262 ms per 256-image batch over the Torch
single-view collection. That is about 1.6 GiB/s of payload copying, not a measurement
of the node's DRAM bandwidth. Allocation, descheduling, GIL reacquisition and
contention may be included. This makes it important but does not justify declaring
`np.stack` intrinsically slow. Preparation should own final batch storage rather
than hand off thousands of independently allocated arrays to a serial stacker.

### 3. Transfers are partly overlapped; completion still gates submission

The Torch H2D worker uses a separate copy stream and pinned source buffers, but
host-synchronizes each staged batch before returning it. This protects buffer
lifetimes and can overlap the previous model; it is not wholly serialized. The
larger issue is [download_tensors](../deployment/mambo_deploy/transfers.py): pack
all ranks, allocate pinned host output, copy on the current compute stream, then
synchronize in the main inference caller. The next model batch is not submitted
until that finishes. A background result thread only starts after this barrier.

The ONNX path binds inputs on CUDA but outputs on CPU and calls synchronous
`run_with_iobinding`; each TTA view returns its logits to CPU. It has not established
an asynchronous output pipeline. Simply adding `non_blocking` or disabling ORT
synchronization would be incorrect without completion events and explicit buffer
lifetimes. PyTorch documents the pinned-memory/separate-stream requirements;
ORT documents device I/O binding and the caller's synchronization responsibility.
[PyTorch transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[ORT CUDA performance guidance](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning).

The current Torch H2D payload is also 2.25 times V2's uint8 512-square payload per
image (384-square float32 versus 512-square uint8). Its model sees 2.94 times the
pixel count (384 versus 224 square), though architecture differs and pixel count
is not a compute-time ratio. Float32 staging is convenient, not inevitable.

### 4. Device results are converted too early and too broadly

[results.py](../deployment/mambo_deploy/results.py) performs full softmax and
selection on CPU, rebuilds class dictionaries for every prediction object, and
eagerly creates nested labels/items. Torch copies 12,632 species, 4,476 genus and
104 family scores per image even when the consumer only needs one label/confidence
per rank. For regional selectors it can additionally download all global leaves
for the finite-value check. V2 selected on device.

The public `raw_logits` property is an existing contract: do not remove it silently.
A compact-result path must be explicit, or lazily materialize raw outputs with
clearly bounded device ownership. Preserve the existing raw mode while avoiding
its cost in a collector that only saves top-1 predictions. Optional embeddings
likewise should incur their cost only when requested.

For masked/TTA Torch inference, `head(features)` computes full native hierarchy
for every view, which is discarded before masked/averaged hierarchy is recomputed.
Embeddings requested separately call `preclassification` again. These are concrete
redundancies, but their speed impact has not been isolated. Use an existing leaf/
embedding boundary where possible; any necessary shared-core API belongs on a
feature branch and must be merged into the release branch.

### 5. Request, stream and evaluator have diverged

`predict()` creates a preparation pool per request, alternates preparation and
inference, retains whole-request outputs and concatenates them. `predict_stream()`
uses the custom scheduler and result worker. The collector calls private methods
and manages another result loop. Thus improving the latter does not necessarily
improve the interface an integration developer uses. Benchmarks currently measure
these materially different paths. One batch primitive and one streaming executor
should serve request collection, streaming and evaluation. Already-device raw tensors
also pass through `_rgb().detach().cpu().numpy()` today; the public API has no
explicit prepared-device-batch boundary for integration with an existing GPU
pipeline. A typed/explicit prepared-input interface would avoid that round trip
without ambiguously treating raw images as normalized model inputs.

## First-principles resource model

For a batch, let service demands be loading/preparation, H2D, backend work, D2H and
postprocessing. A fully serial loop pays approximately their sum. With adequate
buffers and independent resources, steady-state batch time approaches the largest
stage service time. In practice preparation/postprocessing share CPU, RAM bandwidth
and sometimes the GIL; transfers and kernels share memory fabrics. Their resource
demands must be combined where resources are shared. Concurrency cannot remove work.

For a target near 3,000 images/s, batch 256 needs a ready batch every ~85 ms. The
serial preprocessing diagnostic near 191 images/s implies about 15.7 effective CPU
seconds per wall second at that target, before accounting for scaling losses or
other work. Forty-eight Python threads do not establish that capacity. NumPy often
releases the GIL inside native operations, but surrounding Python, allocator and
scheduler activity can still interfere with the inference thread's launches.
The TTA event anomaly makes launch starvation a serious hypothesis, not a finding.
[NumPy thread behavior](https://numpy.org/doc/stable/reference/thread_safety.html).

Reading concurrency addresses latency: outstanding requests are approximately
target images/s multiplied by mean request latency. At 3,000 images/s and 100 ms,
roughly 300 requests may be justified before bandwidth and memory constraints.
That does **not** imply 300 preprocessing workers. Once the encoded queue remains
full, increasing read concurrency cannot address a downstream capacity deficit.
This also agrees with the project's [UCloud training experience](training-workflow-postmortem.md).

| Environment/regime | Likely limiting resource | Appropriate adaptation |
|---|---|---|
| Cold network filesystem/object store | Metadata/read latency, then storage throughput | High bounded read concurrency; keep metadata off the compute scheduler; optional local staging for repeated runs |
| Warm cache/local NVMe plus fast GPU | Decode, resize, memory traffic, host kernel submission | Batch-oriented preparation, native kernels, few allocations; isolate Python-heavy preparation if needed |
| Multi-socket enterprise node | CPU quota, NUMA placement, cross-socket memory/PCIe path | Budget against effective cpuset/quota, place workers and memory near the GPU; don't infer 384 usable CPUs from affinity alone |
| CPU deployment | Competition between backend native threads and preprocessing | Share one CPU budget; avoid multiplying preprocessing workers by backend threads |
| Small images/fast accelerator | Python dispatch, per-batch barriers, result materialization | Coarse tasks, compact outputs, backend event dependencies; compilation/graphs only if launch overhead remains limiting |
| Multi-GPU node | Per-GPU host supply and aggregate storage/CPU bandwidth | One ownership domain per GPU, partition sources; no assumption that one device's reader budget scales independently |
| Restricted/shared environment | Process/shared-memory limits, RAM and installation constraints | Portable bounded thread/synchronous execution remains functional; acceleration remains optional |

NVIDIA recommends considering transfer minimization, overlap and NUMA locality as
separate resource issues. These are placement policies, not reasons to add more
per-image machinery. [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

## Simpler target design

Use **three ownership domains** for the user's five logical phases:

```text
Batch producer                 Backend executor                    Result consumer
read/decode/prepare     ->      H2D -> inference -> D2H      ->     format/save/yield
owns ready batch storage       owns in-flight buffers/events       owns completed CPU results
```

Transfers are backend dependencies, not independent application schedulers. A
small ring of batch slots supplies backpressure and lifetime accounting. Enqueue
the next batch before retiring the previous completed output; only the consumer
waits for output readiness. The exact number of copy streams is backend-dependent,
not automatically one thread/stream for each logical phase.

The producer should accept an external prepared-batch iterable as well as paths.
Path loading remains a convenience adapter. A CPU worker prepares into an assigned
batch slice or owns a complete batch, rather than returning standalone image arrays
for serial stacking. Read latency can retain a bounded IO service without another
prepared-image state machine. Keep logical sample IDs/order explicit; a slow image
must not corrupt alignment. Out-of-order internal completion is optional and needs
bounded reordering, not a changed public output order.

Keep arrays compact until conversion is needed. Prefer optimized CPU uint8
geometry and late normalization; on Torch CUDA, batch tensor transforms can live
in the backend using existing torchvision operators. ONNX-only installations must
not acquire a mandatory Torch dependency. A portable CPU preprocessing path is a
legitimate backend implementation, not a reason to route Torch through NumPy too.
A graph-integrated ONNX preprocessing/postprocessing wrapper is a later artifact
choice if profiling justifies it, not required for the first simplification.

Result reduction should remain native where practical. TTA aggregates leaves once
and reduces the final hierarchy once; requested embeddings are computed once per
view. Provide a compact prediction mode for the evaluator while preserving raw
outputs for developers who use them. Reuse vocabulary metadata. Ordinary request
prediction can collect the same batch execution results, with a lightweight path
for one small batch rather than launching every service unconditionally.

## What established implementations suggest

| Existing implementation | Relevant mechanism | Recommendation here |
|---|---|---|
| PyTorch DataLoader | Batch collation, worker processes, bounded prefetch, pinned transfer preparation | First baseline for the Torch producer; accept caller-provided batches instead of rebuilding its scheduler. Tune process and internal thread budgets together. |
| torchvision transforms | Native CPU/CUDA batch operations, uint8/layout-aware paths | Reuse compatible operators; remove accidental float64 and redundant passes. |
| NVIDIA DALI | CPU/mixed/GPU pipeline execution, managed prefetch, accelerated decode and fused transforms | Architectural reference and optional HPC candidate if native preparation remains limiting; not a mandatory dependency for this portable release. |
| ONNX Runtime I/O binding | Device input/output buffers, execution stream controls | Use for an explicit asynchronous backend contract, not as a claim that CPU-bound outputs already overlap. |
| Triton Inference Server | Dynamic batching of independent requests and model-instance scheduling | Appropriate optional serving layer for many clients; not the fix for a single offline producer that cannot supply batches. |

DataLoader documents multiprocessing to avoid blocking inference with Python
loading, batch-aware fetching, and the costs/constraints of worker processes.
Its automatic batching still involves collation and pinning copies: it is a
maintained baseline, not a zero-copy guarantee. [DataLoader documentation](https://docs.pytorch.org/docs/main/data.html).

DALI's pipeline exposes bounded CPU/GPU queues and explicit asynchronous output
completion; its crop/mirror/normalize operator combines work rather than inserting
another Python stage. Those are the useful design lessons even without adopting
DALI. [DALI executor](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html),
[fused transform](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.crop_mirror_normalize.html).
Triton's dynamic batcher solves combining independent requests, which our already
batched offline campaign does not require. [Triton batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

## Bounded implementation and validation sequence

1. Establish one backend batch boundary and preserve public behavior: preprocessing
   geometry, ordering, presets/custom masks, TTA logit averaging, embeddings, partial
   batches, failure/early-close behavior and standalone ONNX. Keep existing evidence
   immutable. No new runtime or format is necessary for this boundary.
2. Replace per-image NumPy/assembly handoffs with batch-oriented preparation.
   Remove redundant copies/constants/float64 work, the serial stacker and polling
   where a standard producer suffices. Use one explicit resource budget, not another
   layer of automatically multiplying pools. This is the strongest first increment.
3. Make backend submission/completion explicit and move waiting to retirement.
   Retain pinned source/output ownership until completion; do not just delete
   synchronization. Fold the current transfer worker into that ownership model.
   Keep a correct synchronous capability path for CPU and restrictive runtimes.
4. Reduce results where they already reside, with an explicit compact mode, and
   unify request/stream/evaluation execution. Remove per-view hierarchy and duplicate
   embedding work through existing interfaces or a separately reviewed core change.
5. Validate with a short representative run, not another full quality campaign:
   identical image bank/config for V2/V3, request versus stream clearly separated;
   both small Flemming crops and larger photographs; cold/warm IO distinguished.
   A resident-batch replay isolates runtime from producer contention, and one short
   CPU/GPU timeline checks the launch-starvation hypothesis. Measure steady-state
   images/s plus latency/memory and buffer occupancy; synchronize at measurement
   boundaries. Use a few justified settings, not a combinatorial worker sweep.

The first success criterion is less work and fewer independently coordinated stages,
with a measured throughput improvement and unchanged meaningful outputs. GPU
utilization need not become flat or reach 100%. A claimed 2–3k application rate
would need an actual sustained end-to-end measurement; the prepared-input numbers
only show that the current hundreds-of-images/s plateau is not an established
model ceiling. Existing CUDA compilation/graph and decoder accelerators can follow
if the simplified pipeline exposes those as the next material limit.


## Implementation follow-up

Implemented on 25 September 2026 against baseline `7a8f53a`, confined to deployment
and the release harness. This is a bounded replacement of the preparation and
completion boundaries, not a claim that all proposed architecture work is finished.

- Preparation workers now normalize directly into disjoint reusable batch slices.
  Removed the serial assembly executor, per-image output/stack copies, repeated
  buffer sorting and 5 ms polling. Completion callbacks wake the coordinator.
  Request prediction, TTA and release diagnostic helpers use the same direct-fill
  preparation operations. Separate read/prepare concurrency remains useful for
  high storage latency and is still independently bounded.
- Cached the fixed interpolation geometry and normalization constants, and replaced
  source-width intermediate indexing with direct two-axis selection. Retained the
  original interpolation precision and release pixel hashes. The earlier float32
  coefficient probe remains experimental; its larger speedup is not claimed here.
- Torch output copies now run on a dedicated CUDA stream. The existing bounded
  result worker owns completion waits and CPU processing, allowing the submitting
  thread to continue. Device buffer reuse remains protected by CUDA events; output
  allocations retain their lifetime until completion. Standalone ONNX retains its
  synchronous output boundary without acquiring a Torch dependency.
- Top-k confidence construction normalizes only selected entries, avoiding a full
  normalized probability matrix. The full raw-logit public contract is preserved.
- Backend imports and hierarchy helpers are cached on first use. Transfer setup
  imports remain at stream initialization; repeated batch execution reuses loaded
  backend references. No new dependency or configuration control was introduced.

The remaining scheduler, two device slots and result worker retain explicit bounded
ownership. This removes one execution stage rather than adding another executor.
The request and streaming APIs still have different scheduling because requests
accept in-memory images and return accumulated results. Torch still computes
parent ranks per TTA view and repeats embedding preparation when requested; those
are not solved by this change. Moving resize work to GPU or compacting the public
result contract also remains separate work. The current core embedding context is
process-global, so using it across independent predictors would require a core
concurrency change through the prescribed feature-branch workflow.

### Local evidence

RTX 3080 Ti Laptop GPU, 256 real Flemming images, warm filesystem, batch 16,
global vocabulary, Torch auto precision, four preparation/runtime threads for
inference; median of three process-local trials after model warmup:

| Measurement | Before, images/s | After, images/s |
|---|---:|---:|
| Host preparation, 4 workers | 204.2 | 290.4 |
| Host preparation, 16 workers | 248.2 | 340.2 |
| GPU request, including preparation/results | 162.5 | 163.9 |
| GPU streaming, including preparation/results | 183.7 | 202.3 |

Preparation includes reading, decode, transforms and batch delivery, with buffer
reuse. The request result is essentially unchanged. Streaming trial ranges overlap
(before 160–198, after 189–214 images/s), so its roughly 10% median increase is
preliminary. This short, sequential local comparison is neither a B200 projection
nor a replacement for the published campaign benchmarks. It tests smaller cropped
images, not large in-domain photographs or cold WEKA storage.

All before/after predicted classes matched at all three ranks. Scripts and raw
measurements are retained, uncommitted, in `local-evidence/pipeline-review/`
(`compare_pipeline.py`, `before.json`, `after-final.json` and prediction arrays).
Focused checks cover release pixels, custom transforms, ordering, bounds, errors,
early close, CUDA slot reuse and deferred download ownership. Real-model CUDA
checks cover global/northern-Europe lists, default TTA, embeddings and partial
batches, for Torch and standalone ONNX (without importing Torch). A 65-image
Torch release collection additionally verified both preset CSVs, embeddings and
final timing totals. Static checks passed; the affected suite passed 95 tests with
two metric-environment tests skipped. No metric code changed. B200 throughput and
a full quality campaign have not been rerun.


## Compact preparation follow-up

The full-B200 smoke result (873 Torch / 826 ONNX images/s without TTA) and the
user-reported 1/7 MIG result (493 / 333 images/s) support targeting preparation
cost before adding more concurrency. This increment implements that target:

- Portable preprocessing uses FP32 interpolation instead of accidentally promoted
  FP64 intermediates. Frozen legacy hashes remain in tests as reference geometry;
  tests explicitly permit rare one-level uint8 rounding changes and bound their
  mean error. This is an intentional numerical implementation change, not a
  relaxation of the image framing, transform order or normalization contract.
- CUDA Torch request and streaming paths prepare nearest-square uint8 images on
  CPU. Existing batch buffers, pinned storage and device slots preserve uint8.
  Native batched Torch interpolation, center crop, rounding and normalization run
  on the device before the existing FP16-backbone/FP32-head inference boundary.
  A 256-image RGB 384-square staging buffer is 108 MiB instead of 432 MiB.
  This fourfold reduction describes staging storage, not total GPU or process RAM;
  device interpolation also needs temporary floating-point storage.
- TTA transforms still operate before nearest-square preparation. The default
  rotation/padding recipe, custom transform isolation, averaging, class selection
  and embedding semantics are unchanged. Full-resolution TTA materialization,
  redundant head work and optional ONNX reduced precision remain separate targets.
- ONNX and CPU execution retain portable CPU preparation and do not import Torch
  for preprocessing. No new dependencies, worker pools, user flags or model files.

A fresh laptop comparison against `dcb00d0` used the same 256 Flemming images,
batch 16, global list, default Torch CUDA precision and three warmed repetitions:

| Measurement | Before, images/s | After, images/s |
|---|---:|---:|
| Portable CPU preparation, 4 workers | 308.8 | 435.2 |
| Portable CPU preparation, 16 workers | 322.4 | 438.7 |
| Torch CUDA request | 164.0 | 257.7 |
| Torch CUDA streaming | 217.6 | 305.8 |

All 256 predicted species/genus/family labels matched before and after for both
request and streaming. This is a prediction-stability check, not a new accuracy
estimate. These results establish a useful local improvement, not an assumed
B200 speedup. Script outputs and prediction arrays are retained uncommitted in
`local-evidence/compact-preparation/`; the comparison script is
`local-evidence/pipeline-review/compare_pipeline.py`.

Static checks passed. The focused suite passed 107 tests, with two metric-environment
checks skipped (metric code unchanged). Real Torch and standalone ONNX CUDA checks
covered global/northern-Europe lists, default TTA, embeddings and partial batches.
Large-image geometry is also covered in CPU/CUDA preprocessing tests. A 65-image
release collection with default TTA produced complete three-rank CSVs for both
lists and finite 1280-dimensional embeddings; predicted labels matched the prior
implementation throughout.

Run the same [four-variant UCloud check](../dev/releases/mambo_v3/speed-smoke.md)
with fresh output folders to compare this implementation on full B200 and MIG.
Existing model caches and environments are reusable. Full quality evaluations
and the published deployment figures have not been regenerated for this change.

## Streaming ownership and virtual padding

The initial implementation below was delivered in `0c6ace2`; its B200 regression
and the subsequent admission correction are recorded below.

The full-B200 resident experiment establishes a useful reference for the existing
GPU execution: 3,667 images/s at batch 256, 3,781 at 512 and 3,818 at 1,024. The
trace has almost continuous kernel execution; increasing batch size is not the
main answer to the 1,232 images/s streaming result. This increment preserves GPU
execution and addresses preparation and handoffs instead.

- The reading stage owns metadata lookup and encoded-byte reservations. Metadata
  lookup runs in the existing reading pool, so a slow `stat()` cannot block the
  preparation owner. Reservations are granted in input order to prevent later
  reads from occupying the whole byte budget ahead of a required earlier image.
- The preparation owner receives completion messages instead of rescanning future
  dictionaries and buffered images. A priority heap selects the earliest ready
  image; unavailable earlier reads do not prevent later ready work from proceeding.
- That owner alone allocates/recycles preparation buffers. The consumer returns a
  leased batch after use; workers fill assigned disjoint slices. This replaces the
  old buffer-pool lock, shared condition/generation state and repeated progress scans.
  The message queues are bounded indirectly by read admission and batch capacity.
- Built-in non-mutating TTA transforms avoid an unconditional input copy. Custom
  callables, including subclasses of built-ins, retain copy isolation. Edge padding
  is represented in nearest-square sampling coordinates rather than materialized
  as a full-size padded image. Rotation geometry and interpolation are unchanged.
- Portable FP32 preparation converts selected pixels directly to its required
  contiguous format, without an intermediate contiguous uint8 copy. Torch/ONNX
  input contracts and the transfer/result stages are unchanged.

The streaming module is slightly shorter (238 to 231 lines). Across the three
production files, the functional additions produce a net increase of 17 lines;
this is a reduction in coordination state and interactions, not a large net code
reduction. No dependencies, worker pools, user settings or model assets were added.

Validation: the affected suite passed 117 tests with two metric-environment tests
skipped. After separating byte accounting from telemetry, the 17 streaming tests
passed again. Static checks passed. Tests cover slow reads and metadata, earliest
ready work, bounded storage, buffer leases, partial batches, source/worker errors,
early close and blocked-reader shutdown. Materialized versus virtual TTA padding
matches prepared pixels exactly, including on large images. Real Torch and
standalone ONNX CUDA checks cover global/northern-Europe lists, TTA, embeddings
and partial batches.

An initial laptop comparison of the completion/virtual-padding changes showed no
clear throughput shift on 256 small Flemming images: about 269 versus 270 images/s
without TTA and 98 versus 96 with TTA, with overlapping repetition ranges. Predicted
labels matched at all three ranks. This timing preceded moving metadata lookup into
readers; it therefore did not validate the implementation subsequently tested on
B200. It does not establish a speedup. Local evidence is retained under `local-evidence/stream-owner/`.

Use the existing [full-B200 smoke command](../dev/releases/mambo_v3/speed-smoke.md#current-preparation-update)
with a fresh output directory, keeping the same batch and worker settings. Reuse
the resident reference and existing environments; no MIG or quality campaign is
needed for this bounded pipeline comparison.


### B200 regression and read admission correction

The subsequent `b200-full-streaming` run regressed against `b200-full-compact`:

| Variant | Compact streaming, images/s | `0c6ace2` streaming, images/s |
|---|---:|---:|
| Torch | 1,232.4 | 604.8 |
| ONNX | 697.8 | 400.9 |
| Torch + TTA | 412.1 | 296.9 |
| ONNX + TTA | 254.6 | 213.9 |

These are the supplied B200 summaries, not local benchmark estimates. Local
correctness tests did not establish performance at the B200's concurrency.
Inspection found that `EncodedReads` made reader workers wait for their input-order
turn and byte capacity, with `notify_all()` on every admission/release. This created
avoidable contention and occupied IO workers with scheduling waits. Its exact share
of the measured slowdown has not been isolated.

The correction removes that class and its condition variable. The existing owner
receives asynchronous metadata completions, reserves bytes in input order, then
submits only admitted reads. Metadata and reads share the existing IO pool, with
reads dispatched first when slots are available. Readers perform filesystem work;
capacity waits consume no reader slots. Metadata lookahead remains bounded by
`read_window`; encoded bytes remain reserved through preparation. Ordered admission
prevents later images from exhausting the budget ahead of required earlier images;
actual reads and preparation still complete concurrently and out of order.

This removes 17 production lines without adding pools, dependencies or settings.
Virtual padding and reduced image copies are retained. Static checks and deployment
Ruff checks passed; the focused streaming suite passed 15 tests with three unchanged
CUDA transfer tests skipped. It covers ordering, budgets, buffer ownership, failures,
shutdown and continued metadata progress under byte-budget backpressure. GPU transfer
and model code did not change; their existing validation is reused.

The `b200-full-admission` run subsequently measured streaming throughput of 1,226.8,
701.6, 473.5 and 312.9 images/s for Torch, ONNX, Torch + TTA and ONNX + TTA. This
restored the compact baseline for inference without TTA; TTA improved by about 15%
and 23%. Request throughput was 853.2, 420.7, 287.2 and 161.8 images/s, respectively;
peak host memory was 4.47, 5.20, 6.25 and 7.70 GiB. This is recovery from the admission
regression, not resolution of the Torch streaming gap to the resident reference.


### Decode once; sample only the rotation pixels used

The `0ac0422` increment changed preparation work, leaving scheduling, buffers, transfers
and GPU inference unchanged:

- Torch JPEG/PNG inputs use the existing torchvision native CPU decoder, as the core
  loader does. Its decoded tensor shares storage with NumPy; there is no full-size
  Pillow RGB copy/raw-byte export. The decoder is resolved once per predictor before
  dispatching preparation. Other formats, PIL/array inputs and high-bit-depth PNG
  conversion retain the portable path. Standalone ONNX does not import Torch.
- Portable RGB decoding skips `convert("RGB")` when the input is already RGB. Pillow's
  [conversion implementation](https://github.com/python-pillow/Pillow/blob/main/src/PIL/Image.py)
  otherwise copies even same-mode images, and its array interface exports raw bytes.
- Built-in arbitrary-angle TTA operates directly on NumPy arrays. It maps the final
  nearest-square coordinates through the expanded rotation and interpolates only
  those pixels, preserving the existing rotate-to-uint8, edge-pad, nearest-sample
  ordering. Repeated positions from padding/upscaling are evaluated once. Rotation
  no longer converts arrays to Pillow and back or builds a full-resolution rotated
  canvas during preparation. Its expansion, fill and pixel-center conventions match
  the previous [Pillow geometry](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Geometry.c).
  Custom transforms retain their original full-resolution inputs and copy isolation.
- Already square 384-pixel inputs bypass redundant nearest gathering; CPU FP32 and
  Torch batched finishing retain their existing interpolation/normalization.

This is a modest net production-code increase for a native decoder adapter and an
array sampler, not a claimed code-count reduction. It removes representation round
trips and discarded image work without new dependencies, pools, flags or model assets.

Validation was limited to the affected deployment/streaming suite (83 passed, four
CUDA checks skipped) and focused checks for the subsequent high-bit-depth fallback
and repeated-pixel sampling. Pixel fixtures compare against the prior Pillow rotation,
including expanded non-square canvases, thin/large images and cardinal rotations.
No local throughput sweep, model-quality campaign or GPU-reference rerun was performed.
The subsequent `b200-full-preparation` results were:

| Variant | Streaming images/s | Request images/s | Peak host GiB |
|---|---:|---:|---:|
| Torch | 1,826.6 | 995.7 | 3.94 |
| ONNX | 720.5 | 456.6 | 5.02 |
| Torch + TTA | 375.2 | 215.8 | 5.93 |
| ONNX + TTA | 237.1 | 120.1 | 7.05 |

Native decoding improved no-TTA Torch streaming by 49% over admission. Both TTA
variants regressed by 21–24%, strongly implicating the shared NumPy sampler. It
reduced pixel work but introduced multiple array passes, gathers and temporary
allocations in place of compiled interpolation. The sampler was therefore removed
and the previous Pillow rotation restored; native decoding, virtual padding and
square-input shortcuts remain. The focused virtual-padding checks passed after
rollback. Post-rollback TTA throughput was subsequently measured with the combined stack below.


### Profile-guided stack validated on a fresh full B200

The `b200-full-gather` archive records commit `503de96` with the native decoder,
restored Pillow rotation, RGB pixel gathers, cheaper hierarchy lookup/lazy vocabulary
maps, reused CPU interpolation scratch, fewer top-1 score scans, and fused Torch
normalization. The later `f9cbd81` commit changes experiment setup only.

All four reports completed. Imported baseline and compact reports match the new
run's metadata/sample hashes, 4,096 ordered image identities, bundle and class-list
hashes, benchmark settings and recorded runtime versions. The new allocation has a
different GPU UUID but the same full B200 model, 48-CPU quota, 48 preparation workers,
AMD EPYC 9655 CPU, driver 610.57.04, Torch 2.14.0+cu132 and ORT 1.22.0. Batch is 256;
Torch uses FP16 and ONNX TF32. Both ONNX reports select the optimized session profile
with no failed compatibility attempts; the logs contain no failure warnings.

Comparison with the preceding `b200-full-preparation` summary supplied by the user:

| Variant | Previous streaming images/s | New streaming images/s | Change | New request images/s | Request change | Peak host GiB |
|---|---:|---:|---:|---:|---:|---:|
| Torch | 1,826.6 | 1,975.7 | +8.2% | 1,469.3 | +47.6% | 4.01 |
| ONNX | 720.5 | 996.5 | +38.3% | 761.9 | +66.9% | 4.75 |
| Torch + TTA | 375.2 | 623.9 | +66.3% | 391.4 | +81.3% | 5.84 |
| ONNX + TTA | 237.1 | 396.1 | +67.0% | 211.8 | +76.3% | 7.35 |

The immediate prior TTA run included the regressed NumPy rotation. Against the
stronger admission-run TTA figures (473.5 and 312.9 images/s), the new rates are
still +31.8% and +26.6%. Against the original imported full-B200 smoke, all four
streaming variants improve: approximately +126%, +21%, +97% and +31% respectively.
Memory is not uniformly lower: ONNX + TTA rises from 7.05 to 7.35 GiB versus the
preceding preparation summary. Torch allocator peaks are 3.42 GiB without TTA and
3.88 GiB with TTA; these are not total device-memory usage or ORT memory estimates.

This supports the combined structural changes across deployment environments.
Three short passes on one new allocation do not identify each patch's individual
contribution. Torch's observed streaming rates span 1,795–2,109 images/s; its smaller
+8% median change deserves less weight than the larger ONNX/TTA and request gains.
The no-TTA prepared-input diagnostic remains about 92 ms for Torch and 114 ms for
ONNX per batch. It excludes decode/reduction and uses already normalized FP32 input,
so it does not measure the newly optimized compact GPU preprocessing path.

The remaining targets differ by backend. Counters below accumulate all three
passes (48 batches); worker elapsed times overlap and cannot be added to runtime:

| Variant | Background input wait, ms/batch | Transfer-worker elapsed, ms/batch | Measured H2D, ms/batch |
|---|---:|---:|---:|
| Torch | 16.5 | 52.4 | 2.6 |
| ONNX | 227.7 | 21.3 | Not instrumented |
| Torch + TTA | 57.8 | 72.3 | 8.9 |
| ONNX + TTA | 549.6 | 66.5 | Not instrumented |

For ONNX, input wait remains 10.93 s across 12.50 s elapsed without TTA, and 26.38 s
across 30.71 s with TTA. Together with the preparation-worker totals, this prioritizes
CPU decoding/preparation throughput and contention over changes to inference kernels.
These are background waits, not GPU-idle percentages or proof of storage latency.

For Torch, the physical H2D copy is small, while transfer-worker elapsed also includes
slot availability and host dispatch. Summed preparation-worker elapsed is 88.8 s
across 6.30 s wall time, far below 48 workers continuously active. Increasing reader
or preparation counts alone is unlikely to resolve the remaining gap. The next
useful boundary is host submission, safe slot reuse and result completion, including
contention while preparation runs; a timeline should distinguish these rather than
calling all non-DMA transfer-worker time overhead. Existing counters do not resolve
that distinction.

Torch streaming now reaches 53.9% of the earlier batch-256 resident reference of
3,667 images/s. This is a throughput ratio, not GPU utilization. The resident path
excludes transfers and CPU results and predates the normalization improvement;
it remains a useful reference rather than a new measured ceiling. No further run
was requested simply to confirm the gains.

Evidence is retained uncommitted under
`local-evidence/ucloud-speed-smoke-2026-09-25/b200-full-gather/`, with the source
archive alongside it and derived `gather-analysis.json`. Archive SHA-256:
`d51ee3105af4e37aa048bb16fed4c94a9cd87dfefd2a574bbe9e924aec7e6947`.
The immediately preceding preparation-run comparison uses the user's pasted summary;
its complete archive was not supplied. This speed check adds no quality metrics.
