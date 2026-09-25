# V2/V3 inference pipeline review

Review date: 25 September 2026. Code baseline: `0eb90b2`; V2 source:
`32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`. The review below records the original design baseline; the
[implementation follow-up](#implementation-follow-up) records subsequent changes. Evidence is the completed UCloud campaign and source
inspection. No new B200 measurements have been made.

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
