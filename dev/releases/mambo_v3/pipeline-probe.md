# Isolate pipeline overhead with model execution mocked

`pipeline_probe.py` runs three short Torch CUDA cases. It creates synthetic global
species/genus/family scores once and substitutes them for backbone/head execution.
No model weights are loaded. Actual batched GPU preprocessing, output transfers,
score validation and `Prediction` construction remain in the path.

| Mode | Input boundary | What remains |
|---|---|---|
| `resident` | One prepared uint8 batch already on GPU | GPU preprocessing and complete result path |
| `host` | One prepared pinned host batch | Above plus existing two-slot H2D staging |
| `stream` | Real image paths | Complete deployed preparation/transfer/result pipeline |

All cases use the same score tensors and vocabulary. The first two deliberately
reuse one real prepared batch; streaming consumes the selected files. Each case
has one excluded warmup and one timed pass. The script verifies the returned count,
including the partial final batch, and that no real model was instantiated.

Example using existing local assets, without changing the environment:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.releases.mambo_v3.pipeline_probe \
  --bundle local-evidence/mambo-bundle-presets-v2 \
  --manifest local-evidence/mambo-v3/flemming-manifest.json \
  --root /home/asger/data/flemming \
  --output local-evidence/pipeline-probe-initial \
  --count 1025 --batch-size 64 --workers 4
```

Use a fresh output directory. The report includes sample paths/hashes, settings,
GPU identity, throughput, input/transfer counters, submission elapsed time, result
worker elapsed time and caller time blocked retrieving results. These counters
**overlap**; do not sum them or equate host waits with GPU idle time.

The first local run (1,025 Flemming images, batch 64, four preparation workers):

| Mode | Images/s | Elapsed s | Caller result wait s |
|---|---:|---:|---:|
| Resident | 7,771 | 0.132 | 0.036 |
| Host | 7,627 | 0.134 | 0.042 |
| Stream | 699 | 1.466 | 0.002 |

Streaming accumulated 1.391 s waiting for prepared host inputs in the background
transfer worker. Host-side inference/result submission increased from 0.078 s in
resident mode to 0.452 s with preparation active, consistent with host contention.
Results did not throttle this local streaming case. The raw report is retained
uncommitted at `local-evidence/pipeline-probe-initial/report.json`.

This is a diagnostic, not a deployment benchmark or B200 emulation: removing model
latency changes overlap, contention and backpressure, and synthetic scores omit
real forward dispatch. Small Flemming images, the laptop CPU and four workers do
not reproduce UCloud's large images and 48-worker setup. The short prepared-input
cases establish a large local separation, not a precise throughput difference
between their two modes. Use one same-environment probe when that distinction
would change the next implementation; do not introduce a worker sweep or campaign.
