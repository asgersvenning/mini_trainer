# V2/V3 inference pipeline: decisions and remaining limits

Review and implementation campaign: 25 September 2026. Initial review baseline
`0eb90b2`; V2 source `32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`.
Final measured stack: `503de96`. Further throughput work is **deferred for the
release freeze**. Current numbers and their provenance live in
[HPC evidence](mambo-hpc-evidence.md); this page explains the architectural decisions.

## What the evidence established

V3 initially moved work that V2 kept in PyTorch—AMP, hierarchy reduction, selection
and preprocessing—into an expensive portable CPU/NumPy path. Restoring AMP and native
hierarchy repaired regressions. Portable ONNX, audited regional lists, TTA and
standalone CPU integration remain useful additions independent of speed.

V2 was not an ideal asynchronous reference: it read requests serially, transferred
per image and translated labels through scalar device reads. Its full-evaluation
harness also differed from the public API. Backbones, input resolution and precision
differ, so neither model-compute speedup nor implementation overhead can be inferred
from a single V2/V3 end-to-end ratio.

The original UCloud campaign measured V2/V3 Torch/V3 ONNX CPU requests at 2.0/28.7/35.1
images/s (batch 8, four runtime threads), and B200 requests at 539.5/480.7/442.2
(batch 32). V3 initially trailed V2 on GPU while substantially improving measured CPU
performance. [Original request](assets/mambo-indomain-speed.csv) and
[streaming](assets/mambo-indomain-streaming-speed.csv) observations retain that
baseline; they are not current V3 timings.

The 632,913-image Torch collection spent 549.3 s waiting for staged inputs across
875.7 s total. Background assembly was 647.2 s, model-stream intervals 212.9 s,
H2D 27.5 s and D2H 1.36 s. **These times overlap.** They target ready-batch delivery,
not bulk output bandwidth or CSV writing. Utilization samples are neither phase
timings nor SM occupancy; event intervals can include gaps in host submission.

A resident-batch reference reached 3,667 / 3,781 / 3,818 images/s at batches
256 / 512 / 1,024 with almost continuous kernels. This justified prioritizing the
host pipeline over ever-larger batches. It excludes transfer and CPU results,
predates final normalization changes and is not a measured application ceiling.
See the [resident probe](../dev/releases/mambo_v3/gpu-ceiling.md).

## Current implementation and ownership

```text
Batch producer                  Backend executor                 Result consumer
read → decode → prepare   →     H2D → inference → D2H       →     format / save / yield
owns bounded batch storage      owns device buffers/events       owns completed results
```

- **Input admission:** one owner reserves bytes in input order, dispatches admitted
  reads and receives completion events. IO workers perform metadata/filesystem
  work, not capacity waits. Read lookahead, encoded bytes and prepared batches are
  bounded independently; slow earlier images cannot lose output alignment.
- **Preparation:** workers fill disjoint reusable batch slices. Serial stacking,
  repeated future scans and polling were removed. Native Torch JPEG/PNG decoding
  avoids full-resolution Pillow/NumPy round trips where supported; other formats
  retain the portable fallback.
- **Backend-specific finishing:** Torch CUDA stages nearest-square uint8 and performs
  batched interpolation/crop/normalization on device; its batch-256 input storage
  is 108 MiB rather than 432 MiB float32. That is staging storage, not total memory.
  CPU/standalone ONNX use portable FP32 preparation without requiring Torch.
- **TTA:** decode once, preserve geometry and logit-averaging semantics. Virtual
  edge padding avoids materializing padded originals; custom transforms retain
  full-resolution inputs and copy isolation. Default arbitrary rotation uses the
  restored Pillow path; the attempted NumPy sampler was slower and was removed.
- **Completion:** Torch downloads use a dedicated stream; the bounded result worker
  waits for completion while submission can proceed. CUDA events protect device
  slot reuse and pinned-output lifetime. ONNX retains its synchronous output
  boundary; I/O binding alone does not make it asynchronous.
- **Results:** reuse hierarchy/vocabulary plans, avoid repeated top-1 scans and full
  normalized-probability matrices, retain public raw logits and optional embeddings.
  Backend imports resolve at initialization/first use rather than per hot operation.

Current contracts are in [streaming](../deployment/mambo_deploy/streaming.py),
[preparation](../deployment/mambo_deploy/preprocessing.py),
[transfers](../deployment/mambo_deploy/transfers.py),
[result completion](../deployment/mambo_deploy/result_worker.py) and
[predictor](../deployment/mambo_deploy/predictor.py).
Request and streaming still have distinct scheduling/collection costs. The public
raw-logit contract prevents simply dropping all large outputs. Per-view head work
and the shared core's embedding-context concurrency boundary also limit further
simplification; core changes require the separate feature-branch route.

## Consequential negative results

| Attempt | Observation | Durable lesson |
| --- | --- | --- |
| Transfer overlap added to the early pipeline | No demonstrated B200 end-to-end gain; input delivery still dominated. | Asynchronous work can remain the limiting producer. Optimize service demands and ownership, not the number of queues. |
| Read workers waited for ordered admission/capacity (`0c6ace2`) | Torch streaming fell from 1,232.4 to 604.8 images/s. Admission-owner correction recovered 1,226.8. | Capacity waits must not occupy IO worker slots; local tests did not establish high-concurrency throughput. |
| NumPy rotation sampled only needed pixels (`0ac0422`) | Native decode improved non-TTA Torch to 1,826.6 images/s, but Torch/ONNX TTA fell 21–24% from the preceding run. | Fewer mathematical pixels can still mean more array passes, gathers and allocations than compiled interpolation. Restore the faster operator. |
| Increasing batch beyond 256 in the resident reference | Only ~4% gain by batch 1,024. | Model batch size did not explain the much larger streaming gap. |

The final stack combines native decode, restored Pillow rotation, RGB gathers,
reused interpolation scratch, cheaper hierarchy/lazy maps and fused Torch
normalization. The [final measured table](mambo-hpc-evidence.md) reports
Torch/ONNX streaming of 1,975.7/996.5 images/s, and 623.9/396.1 with TTA.
Against the initial full-B200 smoke, gains were approximately 126%/21%/97%/31%.
They are combined-stack improvements, not isolated attribution to each patch.
Peak memory is not uniformly lower; ONNX+TTA reached 7.35 GiB host RSS.

Matched imported baseline/compact reports agree on ordered sample identities,
bundle/class lists, benchmark settings and runtimes. The final run used a different
GPU UUID but the same full B200 model and 48-vCPU quota. Torch's three streaming
passes ranged 1,795–2,109 images/s; small median changes deserve less weight than
large request/ONNX/TTA gains. The prior preparation-run comparison came from the
operator's summary, not a complete imported archive.

## If performance work resumes

Overlapped steady-state throughput approaches the slowest service stage only when
its resources are sufficiently independent. Decode/postprocessing share CPU/GIL/
memory bandwidth; transfers and kernels share memory fabrics. Outstanding IO
requests hide latency; they do not supply downstream compute capacity. Use the
actual CPU quota/NUMA placement and per-GPU demand, not host-wide counts.

The final counters give different next targets:

- **ONNX:** background input wait was 10.93 s across 12.50 s elapsed without TTA,
  and 26.38 s across 30.71 s with TTA. Prioritize CPU preparation throughput and
  contention before inference kernels; these are background waits, not GPU-idle
  percentages or proof of storage latency.
- **Torch:** measured H2D averaged 2.6 ms/batch versus 52.4 ms transfer-worker elapsed,
  which also includes slot availability/dispatch. Preparation-worker time was
  88.8 s across 6.30 s wall time, far from 48 workers continuously active. Examine
  host submission, slot reuse and result completion under preparation load before
  adding workers. The counters alone do not isolate those causes.

Torch streaming is 53.9% of the earlier resident throughput; that is a throughput
ratio, not GPU utilization. GPU saturation/general HPC scalability is unproven.
Use the [mocked pipeline probe](../dev/releases/mambo_v3/pipeline-probe.md) and one
representative timeline to distinguish non-model costs, then the existing
[four-variant smoke](../dev/releases/mambo_v3/speed-smoke.md). Preserve ordering,
geometry, partial batches, embeddings, custom transforms, early close and bounded
buffer lifetimes. Do not reopen a quality campaign for unchanged numerical behavior.

Use maintained backend mechanisms where they simplify responsibility:
[DataLoader](https://docs.pytorch.org/docs/main/data.html) for batch loading,
[torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html#performance-considerations)
for suitable native operations, and
[ORT I/O binding](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning)
with explicit completion ownership. [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html)
is an optional acceleration reference, not a release dependency.
[Triton batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
combines independent requests; it does not fix an offline producer that cannot feed
the GPU. Preserve a bounded portable path for restricted installations.

## Retained evidence

Final archive SHA-256:
`d51ee3105af4e37aa048bb16fed4c94a9cd87dfefd2a574bbe9e924aec7e6947`.
Local reports/archive and `gather-analysis.json` are under
`local-evidence/ucloud-speed-smoke-2026-09-25/`, with final reports in
`b200-full-gather/`. These ignored files are not a remote evidence service.
Earlier local mechanisms/prediction checks remain in `local-evidence/pipeline-review/`,
`compact-preparation/` and `stream-owner/`; they are not B200 timing substitutes.

No new quality metrics came from the speed smoke. Public timing projections retain
raw repetitions, source hashes and execution scopes in the linked HPC evidence.
Installed release qualification is recorded [separately](../dev/releases/mambo_v3/final-qualification.md).
