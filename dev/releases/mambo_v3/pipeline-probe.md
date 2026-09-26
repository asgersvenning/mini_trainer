# Diagnose deployment pipeline costs

These optional tools isolate costs before changing the pipeline. Use the existing
CUDA environment from the [repository setup](../../../README.md#local-installation)
or [UCloud speed workflow](speed-smoke.md), run from the repository root, and choose
fresh output directories. They do not require another quality evaluation.

| Tool | Measured boundary | Use when |
| --- | --- | --- |
| `pipeline_probe` | Synthetic scores replace model execution; real preprocessing, transfers and public results remain | Locating non-model overhead and contention |
| `pipeline_stages` | Isolated preparation, result construction and CUDA operator trace | Inspecting a specific operation or allocation |
| `gpu_ceiling` | Resident inputs through the deployed Torch model; outputs stay on-device | Comparing streaming throughput with sustained model execution |

Current decisions, failed approaches and remaining targets belong in the
[pipeline review](../../../docs/mambo-inference-pipeline-review.md); published
measurements and provenance belong in [HPC evidence](../../../docs/mambo-hpc-evidence.md).
Further throughput work is deferred for the release freeze.

## Profile without model execution

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.releases.mambo_v3.pipeline_probe \
  --bundle local-evidence/mambo-bundle-presets-v2 \
  --manifest local-evidence/mambo-v3/flemming-manifest.json \
  --root /home/asger/data/flemming \
  --output local-evidence/pipeline-probe-new \
  --count 1025 --batch-size 64 --workers 4
```

Substitute actual local paths. No weights are loaded. All modes use the same fixed
synthetic species/genus/family scores and vocabulary:

- `resident`: a prepared uint8 batch is already on the GPU.
- `host`: the same batch starts in pinned host memory and uses two-slot H2D staging.
- `stream`: real image paths exercise the complete preparation pipeline.

Each mode has one excluded warmup and one timed pass. The probe verifies result
count, including the partial tail, and that no model was instantiated. `report.json`
retains sample hashes, settings, GPU identity, throughput, transfer/input counters,
submission/result-worker durations and caller waits.

Removing model latency changes overlap and backpressure. This is **not an HPC
emulator**: laptop crops and four workers do not reproduce large photos and 48
workers. Durations overlap; do not sum them or interpret host waits as GPU idle time.
Use a profile to locate expensive work and an unprofiled run to measure it.

## Isolate preparation and result stages

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.releases.mambo_v3.pipeline_stages \
  --output local-evidence/pipeline-stages-new
```

The probe uses decoded 256-square/2048-square RGB images and synthetic scores,
excluding filesystem, decoding and model execution. CPU stage timings go to
`report.json`; `trace.json` and `operators.txt` record CUDA operators, allocations
and strides. **CUDA trace timings are diagnostic, not unprofiled throughput.**

Prior profiling justified native RGB gathering, reused interpolation/score scratch,
lazy vocabulary maps and fused normalization. Output-copy pooling did not justify
its added complexity. Native sampling perturbed execution and was used only to
locate costs. Detailed local evidence is ignored under
`local-evidence/pipeline-profile/` and `local-evidence/pipeline-stages/` (stage baseline
`c497a9d`, RTX 3080 Ti Laptop); these paths are not guaranteed on another checkout.
The linked HPC evidence qualifies the combined implementation.

## Measure resident GPU throughput

Reuse a completed Torch/no-TTA speed-smoke report and its working environment:

```sh
.venv-mambo-runtime/bin/python -m dev.releases.mambo_v3.gpu_ceiling \
  --baseline /work/mambo-speed/b200-full-compact/torch/report.json \
  --output /work/mambo-speed/b200-resident
```

The probe verifies the bundle and first 1,024 sample images, prepares them once,
and keeps uint8 inputs on the GPU. Calls include deployed GPU preprocessing,
backbone, classifier and global hierarchy logits. Loading, H2D/D2H, CPU results,
embeddings and TTA are excluded; precision follows the baseline report.

Batches 256, 512 and 1,024 each receive warmup and approximately 20 seconds of
sustained inference, synchronized at block boundaries. Smaller-batch results survive
an out-of-memory failure. Expect roughly 2–4 minutes plus cold-storage delays;
keep other GPU workloads idle. For a small local check, use `--batches 8 16 --seconds 1`.

| Output | Interpretation |
| --- | --- |
| `summary.csv`, `report.json` | Throughput, timing windows, sample/bundle hashes, runtime/device and Torch allocated/reserved peaks, including the resident bank |
| `gpu.csv` | 200 ms utilization/power/memory/clock samples; match device identity because nvidia-smi may see other GPUs |
| `trace.json.gz` | Separate eight-call trace at the fastest batch; profiler failure is recorded without discarding timings |

Return the complete folder. A throughput plateau plus continuously occupied kernels
supports a practical reference for this implementation, not a hardware maximum.
Compare matching batch sizes; the prepared-input smoke diagnostic has a different
boundary. Neither these timings nor the mocked modes are additive pipeline phases.

## Qualify a change

Inspect service demand and ownership before adding queues or workers. An owner
blocked in `events.get()` is waiting, not necessarily consuming CPU. Preserve
geometry, rounding, tie/NaN behavior, raw scores, custom transforms, ordering,
partial batches and buffer lifetimes. Cross-stream copies need completion and
lifetime handling as well as `non_blocking=True`.

Use existing preprocessing/result/streaming tests for changed contracts, then the
[four-variant speed smoke](speed-smoke.md) when target throughput remains unresolved.
Do not infer HPC gains from isolated local timings or rerun a full evaluation
campaign for unchanged prediction behavior.
