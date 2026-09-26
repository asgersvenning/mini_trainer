# V2/V3 inference pipeline: decisions and remaining limits

The 25 September 2026 review compared V3 `0eb90b2` with V2
`32b3cd661778356b2e8c4cff5b10fa9061aa6f5d`; the final measured stack was `503de96`.
Further throughput work is **deferred for the release freeze**.
[HPC evidence](mambo-hpc-evidence.md) owns current timings, historical comparisons
and measurement provenance. This page retains implementation decisions and limits
needed for future performance work.

## What the evidence established

V3 initially lacked V2's AMP use and shifted hierarchy reduction, selection and
preprocessing into costly portable CPU/NumPy work. Restoring AMP and native
hierarchy repaired regressions. V2 itself read requests serially, transferred per
image and translated labels through scalar device reads; its evaluation harness
also differed from its public API. Different backbones, resolutions and precision
prevent attributing a V2/V3 end-to-end ratio solely to model or adapter efficiency.

The original 632,913-image Torch collection spent 549.3 s waiting for staged inputs
across 875.7 s total. Background assembly took 647.2 s, model-stream intervals
212.9 s, H2D 27.5 s and D2H 1.36 s. **These times overlap:** ready-batch delivery,
rather than bulk output transfer or CSV writing, was the consequential target.
Utilization samples are neither phase timings nor SM occupancy, and event
intervals can include host-submission gaps.

A [resident-batch reference](../dev/releases/mambo_v3/pipeline-probe.md#measure-resident-gpu-throughput)
reached 3,667 / 3,781 / 3,818 images/s at batches 256 / 512 / 1,024 with almost
continuous kernels. The small gain justified prioritizing the host pipeline over
larger batches. This reference excludes transfers and CPU results and predates the
final normalization changes; it is not an application throughput ceiling.

## Current implementation and ownership

```text
Batch producer                  Backend executor                 Result consumer
read → decode → prepare   →     H2D → inference → D2H       →     format / save / yield
owns bounded batch storage      owns device buffers/events       owns completed results
```

- [Input admission](../deployment/mambo_deploy/streaming.py) has one owner reserving
  bytes in input order and dispatching reads. IO workers perform filesystem work,
  never capacity waits. Read lookahead, encoded bytes and prepared batches have
  independent bounds; workers fill disjoint reusable batch slices and completion
  events preserve output order without serial stacking or future polling.
- [Preparation](../deployment/mambo_deploy/preprocessing.py) uses native Torch
  JPEG/PNG decoding where supported, with a portable fallback. Torch CUDA stages
  nearest-square uint8 inputs and finishes interpolation/crop/normalization on
  device: batch-256 input storage is 108 MiB versus 432 MiB float32, not total
  memory. CPU and standalone ONNX prepare FP32 without requiring Torch.
- [TTA](../deployment/mambo_deploy/augmentation.py) decodes once and preserves
  geometry and logit averaging. Virtual edge padding avoids allocating padded
  originals; custom transforms retain full-resolution copy isolation. Arbitrary
  rotation uses Pillow after the slower NumPy sampler was removed.
- [Transfers](../deployment/mambo_deploy/transfers.py) and
  [result completion](../deployment/mambo_deploy/result_worker.py) separate Torch
  submission from waiting: a dedicated download stream and bounded result worker
  overlap completion with inference. Events protect device-slot reuse and pinned
  output lifetime. ONNX retains a synchronous output boundary; I/O binding alone
  does not make it asynchronous.
- [Prediction](../deployment/mambo_deploy/predictor.py) reuses hierarchy/vocabulary
  plans, avoids repeated top-1 scans and normalized-probability matrices, and
  retains public raw logits and optional embeddings. Backend imports resolve at
  initialization/first use rather than per hot operation.

Request and streaming retain different scheduling/collection costs. Raw-logit
outputs, per-view head work and the shared core's embedding-context concurrency
boundary constrain further simplification. Core changes require the separate
feature-branch route.

## Consequential negative results

| Attempt | Observation | Durable lesson |
| --- | --- | --- |
| Early transfer overlap | No demonstrated B200 end-to-end gain; input delivery still dominated. | An asynchronous producer can remain the bottleneck. Improve service demands and ownership before adding queues. |
| IO workers waiting for ordered admission (`0c6ace2`) | Torch streaming fell from 1,232.4 to 604.8 images/s; moving admission to its owner recovered 1,226.8. | Capacity waits must not occupy IO worker slots. Local tests did not establish high-concurrency throughput. |
| NumPy rotation sampling only needed pixels (`0ac0422`) | Non-TTA Torch improved to 1,826.6 images/s with native decode, but Torch/ONNX TTA fell 21–24% from the preceding run. | Extra array passes, gathers and allocations can outweigh fewer sampled pixels. Restore the faster compiled operator. |

Final gains reflect the combined stack, not isolated patch attribution. Imported
baseline/compact reports agree on sample order, bundle/lists, settings and runtimes;
the final run used another GPU UUID with the same full B200 model and 48-vCPU quota.
Torch's three streaming passes ranged 1,795–2,109 images/s. Small median changes
warrant less weight than large gains. The preparation-run comparison above came
from the operator's summary, not a complete imported archive.

## If performance work resumes

Throughput approaches the slowest stage's capacity only when stage resources are
sufficiently independent. Preparation and postprocessing share CPU/GIL/memory
bandwidth; transfers and kernels share memory fabrics. Outstanding IO hides
latency but supplies no downstream compute. Use actual CPU quota/NUMA placement
and per-GPU demand rather than host-wide CPU counts.

The final counters identify two investigation priorities:

- **ONNX preparation and contention:** background input wait was 10.93 s across
  12.50 s elapsed without TTA, and 26.38 s across 30.71 s with TTA. This points to the producer before
  inference kernels; background waits are not GPU-idle percentages or proof of
  storage latency.
- **Torch submission and buffer reuse under load:** H2D averaged 2.6 ms/batch
  versus 52.4 ms transfer-worker elapsed, including slot availability/dispatch.
  Preparation consumed 88.8 worker-seconds across 6.30 s wall time, far below 48
  continuously active workers. Inspect submission, slot reuse and result completion
  before adding workers; these counters do not isolate their individual costs.

Final Torch streaming was 53.9% of the earlier resident throughput, **not GPU
utilization**. Saturation and general HPC scalability remain unproven. Use the
[mocked pipeline probe](../dev/releases/mambo_v3/pipeline-probe.md) and one
representative timeline, then the existing
[four-variant smoke](../dev/releases/mambo_v3/speed-smoke.md). Preserve ordering,
geometry, partial batches, embeddings, custom transforms, early close and buffer
lifetimes. Unchanged numerical behavior does not require another quality campaign.

Prefer maintained backend mechanisms when they simplify ownership:
[DataLoader](https://docs.pytorch.org/docs/main/data.html),
[torchvision native operations](https://docs.pytorch.org/vision/stable/transforms.html#performance-considerations)
and [ORT I/O binding](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#performance-tuning).
[DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline.html) is an
optional acceleration reference, not a dependency.
[Triton batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
combines independent requests; it cannot repair an underfed offline pipeline.
Keep a bounded portable path for restricted installations.

## Retained evidence

Final archive SHA-256:
`d51ee3105af4e37aa048bb16fed4c94a9cd87dfefd2a574bbe9e924aec7e6947`.
Ignored local reports and `gather-analysis.json` are under
`local-evidence/ucloud-speed-smoke-2026-09-25/`, final reports in `b200-full-gather/`.
Earlier mechanism checks are in `local-evidence/pipeline-review/`,
`compact-preparation/` and `stream-owner/`; they are not B200 timing substitutes or
remotely available evidence.

The linked HPC page retains public timing repetitions, hashes and execution scopes.
The smoke produced no new quality metrics;
[installed release qualification](../dev/releases/mambo_v3/final-qualification.md)
is separate evidence.
