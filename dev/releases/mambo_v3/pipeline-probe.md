# Profile pipeline overhead without model execution

`pipeline_probe.py` substitutes fixed synthetic global species/genus/family scores
for backbone/head execution; it loads no model weights. Actual GPU preprocessing,
transfers, score validation and public result construction remain.

| Mode | Input boundary | Remaining work |
| --- | --- | --- |
| `resident` | Prepared uint8 batch already on GPU | GPU preprocessing and complete result path |
| `host` | Prepared pinned host batch | Above plus two-slot H2D staging |
| `stream` | Real image paths | Full preparation/transfer/result pipeline |

All modes use the same scores/vocabulary. Resident/host reuse one prepared image
batch; stream reads the selected files. Each has one excluded warmup and one timed
pass, verifies returned count including the partial tail, and asserts no real
model was instantiated.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.releases.mambo_v3.pipeline_probe \
  --bundle local-evidence/mambo-bundle-presets-v2 \
  --manifest local-evidence/mambo-v3/flemming-manifest.json \
  --root /home/asger/data/flemming \
  --output local-evidence/pipeline-probe-new \
  --count 1025 --batch-size 64 --workers 4
```

Use a fresh output and actual local paths. The report retains sample hashes,
settings, GPU identity, throughput, transfer/input counters, submission and
result-worker durations and caller waits. These overlap; do not sum them or equate
host waits with GPU idle time.

This is a diagnostic, **not a B200 emulator**: removing model latency changes
overlap/backpressure and omits real forward dispatch. Laptop crops and four
workers do not reproduce large photos and 48 workers. Use the probe to identify
cost mechanisms, then qualify the combined change with the existing
[four-variant smoke](speed-smoke.md), not another full campaign.

## What the completed profiling changed

Native sampling located repeated NumPy iterator work in pixel gathering. The
200 Hz native-stack profile perturbed execution and was used for localization
only. Unprofiled comparisons drove the implementation decisions:

| Change | Evidence and interpretation |
| --- | --- |
| Gather whole RGB pixels with native `take` | Original 1,025-image mocked stream: 699 → 957 images/s at batch 64/four workers. Large contiguous and small/strided sources avoid different copying costs. |
| Reuse hierarchy plans, lazy vocabulary maps and score scratch | Resident submission 77 → 31 ms; result work 128 → 79 ms. File streaming did not improve further in that pass (957 → 928 images/s). |
| Reuse interpolation scratch, reduce top-1 scans, fuse Torch normalization | Isolated CPU preparation for 32 small/large decoded images: 60.6/65.6 → 35.0/37.7 ms; batch-256 result construction 31.8 → 18.2 ms; GPU finishing at batch 64: 3.95 → 2.32 ms. |
| Consider output-copy pooling | Packing ~44 µs versus ~0.66 ms D2H for 8.4 MiB at batch 64 did not justify further changes. Preserve ownership and stream synchronization. |

These are local mechanism timings, not projected HPC gains. Final short
resident/host/stream observations were 12,110/15,055/1,068 images/s; the inverted
host/resident order illustrates noise in tiny timings, not a benefit from transfers.
[Current B200 evidence](../../../docs/mambo-hpc-evidence.md) subsequently qualified
the combined stack. The [pipeline review](../../../docs/mambo-inference-pipeline-review.md)
owns current architecture, failed approaches and remaining targets; no new speed
experiment is required by this document.

## Isolate preparation and result stages

For work on these specific boundaries, `pipeline_stages.py` uses decoded
256-square/2048-square RGB and synthetic score matrices, excluding filesystem,
decode and model execution. Unprofiled CPU/CUDA measurements supply timings; a
separate CUDA trace records operators, allocations and strides.

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.releases.mambo_v3.pipeline_stages \
  --output local-evidence/pipeline-stages-new
```

Inspect useful work before adding queues or workers: `events.get()` in an owner
thread is a blocking wait, not itself a CPU hotspot. Preserve geometry, rounding,
custom transforms, tie/NaN semantics, immutable raw scores and buffer lifetimes.
Native `take(mode="clip", out=...)` avoids the buffered output of `raise` mode
when indices have already been clamped; see
[NumPy's contract](https://numpy.org/doc/stable/reference/generated/numpy.take.html).
Cross-stream transfers require completion and lifetime handling, not simply
`non_blocking=True`; see [PyTorch stream semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams).

Raw ignored evidence: `local-evidence/pipeline-probe-initial/`,
`local-evidence/pipeline-profile/` (native profile and sequential probe reports),
and `local-evidence/pipeline-stages/` (stage traces and `gpu-unprofiled.json`).
The stage baseline was `c497a9d` on RTX 3080 Ti Laptop. These local paths are not
guarantees of availability elsewhere; retain inputs/provenance when transferring.
Use existing preprocessing/result/streaming tests for changed contracts and reserve
real target measurements for changes whose performance remains unresolved.
