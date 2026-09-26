# Current HPC deployment timings

These measurements support the [deployment guide](../deployment/README.md).
A fresh full NVIDIA B200 / AMD EPYC 9655 allocation with a 48-vCPU quota ran commit
`503de96` on 25 September 2026. Runtime versions were PyTorch 2.14.0+cu132 and
ONNX Runtime 1.22.0; automatic precision selected FP16 and TF32 respectively.

Global vocabulary; batch 256; no embeddings; four runtime threads. Streaming uses
48 preparation workers, 128 readers, a 4,096-image window, two prefetched batches
and a 1 GiB encoded-byte budget. Each variant runs in a separate process with three
warmed repetitions. The error bars show their range, not a confidence interval.
Updated request timing processes 256 images per call. Streaming processes the same 4,096
images per pass, including pipeline startup and final result completion. These
modes must not be pooled. The figure retains the earlier CPU and smaller-batch
GPU request curves, replacing the batch-256 request points and streaming bars.
Updated points are not joined to older curves. Circles/lines are earlier evidence;
diamonds/bars are updated. This is warm-storage throughput, not cold WEKA performance.

![CPU and GPU request throughput with updated B200 streaming](assets/mambo-hpc-current-speed.svg)

| Variant | Request images/s | Streaming images/s | Peak host GiB |
|---|---:|---:|---:|
| V3 PyTorch | 1,469.3 | 1,975.7 | 4.01 |
| V3 ONNX | 761.9 | 996.5 | 4.75 |
| V3 PyTorch + TTA | 391.4 | 623.9 | 5.84 |
| V3 ONNX + TTA | 211.8 | 396.1 | 7.35 |

Host memory is the high-water mark of each complete benchmark process, not a
per-mode or GPU-memory measurement. TTA uses `rotation30_pad25_3`. Both ONNX sessions
used the optimized graph profile with no failed compatibility attempts.

[Raw repetition timings](assets/mambo-hpc-current-speed.csv) retain both execution
modes. [Provenance](assets/mambo-hpc-current-provenance.json) records configuration,
runtime versions, source report hashes, sample/bundle identities and hardware.
These improvements do not establish GPU saturation or general HPC scalability.
No new quality evaluation was performed by this speed check.

## Earlier CPU and V2 comparisons

CPU and V2 have not been rerun; their values in the combined figure are unchanged.
The [earlier campaign figure](assets/mambo-indomain-speed.svg),
[request observations](assets/mambo-indomain-speed.csv),
[streaming observations](assets/mambo-indomain-streaming-speed.csv), and
[campaign provenance](assets/mambo-indomain-campaign.json) retain the EPYC CPU,
V2 and batch-size comparisons. They use an earlier V3 adapter and a different
streaming protocol; do not present them as current implementation timings or join
them to the new batch-256 points as one scaling curve. The laptop results in the
main guide are also retained as historical consumer-device evidence.

## Reproduce the published figure

With the extracted B200 archive available locally:

```sh
.venv/bin/python -m dev.releases.mambo_v3.hpc_speed_report \
  --source local-evidence/ucloud-speed-smoke-2026-09-25/b200-full-gather \
  --output docs/assets \
  --baseline docs/assets/mambo-indomain-speed.csv
```

The [speed workflow](../dev/releases/mambo_v3/speed-smoke.md) records how to run the
experiment. Further throughput work is deferred for the deployment freeze.
