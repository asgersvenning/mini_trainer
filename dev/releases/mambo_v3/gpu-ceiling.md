# Small full-B200 GPU throughput reference

Run **Torch, no TTA** on the full B200. Reuse the working Torch environment and
completed compact speed-smoke report. No environment rebuild, parquet scan,
ONNX setup, MIG run or quality evaluation.

After pulling the release branch, from the repository root:

```sh
.venv-mambo-runtime/bin/python -m dev.releases.mambo_v3.gpu_ceiling \
  --baseline /work/mambo-speed/b200-full-compact/torch/report.json \
  --output /work/mambo-speed/b200-resident
```

The script checks and prepares the first 1,024 images from the prior streaming
sample once, then keeps compact uint8 images on the GPU. Each call includes the
actual deployed GPU interpolation/normalization, FP16 backbone, FP32 classifier
and global hierarchy logits. Outputs stay on-device. Image loading, H2D/D2H,
CPU prediction objects, embeddings and TTA are outside this timing boundary.

It warms batches **256, 512 and 1,024**, then times approximately **20 seconds per
batch size**, synchronizing at block boundaries rather than after each inference.
It cycles slices of the resident bank. If a batch exhausts memory, smaller-batch
results are retained. Expect roughly **2–4 minutes** including model startup and
one short profiler capture; cold image storage can add time. Do not run another
GPU workload alongside it.

The new output directory contains:

- `summary.csv`: images/s and peak Torch allocated/reserved GPU memory per batch.
- `report.json`: exact timing windows, sample/weight provenance, runtime, device
  and the fastest tested batch size. Memory peaks include the resident image bank.
- `gpu.csv`: utilization, power, GPU memory and SM clock samples every 200 ms.
  Includes all visible-to-nvidia-smi devices; match the device identity in the report.
- `trace.json.gz`: eight inferences at the fastest batch size, captured separately
  from throughput timing. Inspect with Perfetto or a Chrome trace viewer. If the
  profiler is unavailable, its error is recorded without discarding timing results.

Return this **one folder**. Read throughput together with the sustained telemetry
and kernel timeline: a throughput plateau and continuously occupied GPU with few
launch gaps support a practical bound for this implementation. High utilization
alone is insufficient, and this is not a theoretical hardware maximum. If 1,024
is still markedly faster, the experiment establishes headroom, not a plateau.
Profiler timings must not replace the unprofiled throughput result.

Compare batch 256 directly with the existing streaming batch 256; a faster larger
batch is an additional opportunity, not a like-for-like pipeline speedup. The old
prepared-input diagnostic still includes transfers and omits new GPU preprocessing,
so keep it separate. Do not subtract these timings as if they were serial phases.

For a small local correctness check only, override `--batches 8 16 --seconds 1`.
No additional settings or campaign configuration are required.
