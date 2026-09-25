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


## Profile-driven pixel gathering change

A native `py-spy` profile located the preparation hotspot in `_square`: the
broadcast three-axis NumPy expression repeatedly entered `mapiter_get` and buffered
iterator code. Sampling at 200 Hz with native stacks fell behind and substantially
perturbed execution; its timings are **not** performance evidence. It was used only
to locate the hot operation. The profile and a small selector comparison are under
`local-evidence/pipeline-profile/`.

Deployment now gathers complete RGB pixels with `take`. Large contiguous decoded
images use flat pixel indices, avoiding a full-width row intermediate. Small or
strided images gather rows and then columns, avoiding a source-sized flattening
copy. Coordinates, padding, output layout and caller-owned buffers are preserved.
This changes both Torch and ONNX preparation, without new configuration or queues.

Unprofiled probe comparison with the original 1,025-image run, same batch/workers:

| Measurement | Before | After |
|---|---:|---:|
| Streaming images/s | 699 | 957 |
| Preparation worker elapsed seconds (summed) | 4.883 | 3.423 |
| Background input wait seconds | 1.391 | 0.987 |
| Caller result wait seconds | 0.0024 | 0.0026 |
| Prepared resident images/s | 7,771 | 7,557 |
| Prepared host images/s | 7,627 | 7,221 |

The final report is `local-evidence/pipeline-profile/after-layout-gather/report.json`.
This is a useful local +37% end-to-end diagnostic improvement, not an expected B200
speedup. Preparation/submission contention remains; this does not establish GPU
saturation. Static checks and the affected deployment/streaming tests cover the
change. Use the existing four-variant speed smoke for the next B200 measurement.


## Stack submission and result work before the next B200 test

The same captured profile exposed additional work beyond pixel gathering:

- `hierarchy_plan` repeatedly converted the entire selected vocabulary into Python
  integers and hashed that tuple on the submission thread. Lookup now hashes native
  contiguous index bytes and reuses each resolved plan within `_ranked_views`.
  Selection order and content still determine the cache key.
- `Prediction` eagerly rebuilt all class-name dictionaries for each batch. It now
  snapshots names and constructs `cls2idx` only on access, including serialization.
  Each result retains its own mutable dictionary, independent of later selections.
- Confidence normalization allocated separate shifted-logit and exponential arrays.
  Floating-point scores now use one scratch array; raw logits remain untouched.

The unprofiled combined probe used the same 1,025 images, batch 64 and four workers:

| Measurement | Pixel gather only | Plus submission/result changes |
|---|---:|---:|
| Resident images/s | 7,557 | 11,806 |
| Host images/s | 7,221 | 9,598 |
| Stream images/s | 957 | 928 |
| Resident submission seconds | 0.077 | 0.031 |
| Resident result-worker seconds | 0.128 | 0.079 |
| Stream preparation-worker seconds (summed) | 3.423 | 3.492 |

Raw report: `local-evidence/pipeline-profile/after-host-overhead/report.json`.
These short single passes show reduced overhead with prepared inputs, but **no
additional local file-streaming improvement**. Streaming still waits on preparation;
submission elapsed time there also includes contention with preparation workers.
The combined streaming rate remains above the original 699 images/s baseline.
Do not convert these diagnostic differences into projected B200 gains.

Other sampled work includes copying gathered pixels into batch storage, GPU
preprocessing, packing/downloading rank scores, and required score validation.
These remain possible limits after preparation improves. The profile does not
establish transfer-bandwidth saturation or a need for more queues: its prominent
owner-thread `events.get()` frame is a blocking wait. No additional scheduler,
transfer pool or result API is introduced for this stack.

Validation: static/import checks, deployment Ruff checks, and affected deployment,
streaming and evaluation tests (92 passed, six optional tests skipped). The CUDA
probe retained actual transfers/preprocessing but mocked model execution. The next
HPC check is the existing four-variant full-B200 smoke, once for the complete stack.


## Further stage simplification

A bounded synthetic stage check removes filesystem/cache latency, decoding and
model execution from attribution. It uses decoded interleaved RGB at 256×256 and
2048×2048, and three score matrices with 30,000/4,000/500 classes at batches 64 and
256. CPU timings are collected outside the profiler. A separate CUDA trace records
actual preprocessing and asynchronous result download, with operator counts,
allocations and output strides. Reproduce it only when investigating those stages:

```sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m dev.releases.mambo_v3.pipeline_stages \
  --output local-evidence/pipeline-stages-check
```

The implementation changes are:

- CPU bilinear preparation uses native `take` operations and reuses interpolation
  scratch and caller-owned FP32 output storage. It avoids separate multiplication,
  sum and final-output temporaries. Pre-clamped indices permit `mode="clip"`;
  NumPy documents that the default `raise` mode always buffers `out`
  ([reference](https://numpy.org/doc/stable/reference/generated/numpy.take.html)).
  This benefits ONNX preparation and Torch CPU preparation, including each TTA view.
- Top-1 result construction reuses the maximum selected by `argmax`. NaN detection
  checks that one selected score per image instead of allocating and scanning a
  full score-sized boolean array, while retaining the stable-sort fallback.
  Confidence normalization reuses the same maximum, removing another full scan.
- Torch finishing uses `addcmul` for broadcast normalization. Rounding remains
  unchanged; rounding plus normalization now requires two kernels instead of four.
  Output is contiguous 384×384 storage instead of a view retaining 438×438 storage.
  This applies to every CUDA Torch view without compilation or a new dependency.

Local stage comparison (before = `c497a9d`, RTX 3080 Ti Laptop GPU):

| Stage | Before | After |
|---|---:|---:|
| CPU preparation, 32 small images / four workers | 60.6 ms | 35.0 ms |
| CPU preparation, 32 large images / four workers | 65.6 ms | 37.7 ms |
| Result construction, batch 64 | 6.69 ms | 4.32 ms |
| Result construction, batch 256 | 31.8 ms | 18.2 ms |
| GPU preparation, batch 64, unprofiled paired median | 3.95 ms | 2.32 ms |

Raw stage reports/traces are in `local-evidence/pipeline-stages/`; the paired CUDA
event timings are in `gpu-unprofiled.json`. Profiler durations are used to locate
work, not as the timing comparison. Fewer full-array passes and compact output
storage are structural improvements; the percentages above are local measurements,
not predicted B200 gains. They also do not establish pipeline GPU saturation.

The final short mocked-model pipeline pass (`after-stage-simplification/report.json`
under `local-evidence/pipeline-profile/`) measured resident/host/stream at
12,110/15,055/1,068 images/s, versus 11,806/9,598/928 before this increment. Its very
short prepared-input runs remain sensitive to scheduling; the host-versus-resident
ordering must not be read as a benefit from transferring inputs. It is an
integration check, not the basis for choosing the changes.

Output packing remained about 44 microseconds for batch 64 in the CUDA traces,
versus roughly 0.66 milliseconds for the resulting 8.4 MiB D2H copy. The snapshot
also protects deferred results against later device-slot writes. Direct-copy or
buffer-pooling changes are not justified by this evidence. Transfer leases and
stream synchronization remain intact; PyTorch requires explicit synchronization
and lifetime handling across streams
([reference](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-streams)).
The compact CPU path still copies selected RGB pixels into planar batch slots;
that is bounded to 384×384 pixels, rather than another source-sized image copy.

Validation covers output storage, stable ties/NaNs, immutable raw scores, exact CPU
interpolation against the prior equation, the original FP64 fixture hashes and
CUDA image geometry. The FP64 fixture now evaluates its frozen equation directly
instead of monkeypatching the optimized production function's scratch dtype;
expected hashes and tolerances are unchanged. Initial failures of two such fixture
cases were resolved by separating that reference. All 96 focused CPU cases pass
across the initial run and targeted rerun; six optional cases were skipped. The
intentional CUDA geometry case passed separately. Static/import checks pass.

Run the existing four-variant full-B200 smoke once for the whole stack. This adds
no campaign, environment setup, scheduler or tuning option.
