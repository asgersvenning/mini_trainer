# UCloud release comparison

Run this workflow **inside a manually allocated UCloud SSH node** with the original
global-lepi dataset mounted. Clone this release branch with its Git history there;
the scripts live in `dev/releases/mambo_v3` and are not included in the deployment
wheel. No allocation or remote submission is performed by these commands.

## Setup with uv

From the checkout root, resolve runtime dependencies afresh for this machine.
If already prepared, run the environment creation, activation and installation commands, then follow
[the existing-campaign instructions](#existing-prepared-campaign-reuse-models-and-image-hashes):

```sh
uv venv --python 3.13 .venv-mambo-runtime
source .venv-mambo-runtime/bin/activate
uv pip install --torch-backend=auto \
  -r dev/releases/mambo_v3/runtime-requirements.in
python -m dev.releases.mambo_v3.setup_ucloud_release \
  --metadata /work/datasets/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet
```

This uses the deployment/training packages' dependency ranges, without reading
repository lockfiles or the evaluation project's exact runtime pins. Only the
`mini_metrics` revision remains fixed, to preserve metric semantics. uv selects the
PyTorch backend from the driver; ONNX Runtime's upstream CUDA/cuDNN extras supply
its runtime dependencies. GPU execution still needs the qualification below.
The separate environment preserves the checkout's existing `.venv`.

The Parquet path is the only required dataset argument. Images are expected at
`images/<species>/<filename>` below its parent; use `--root` if mounted elsewhere.
Preparation verifies the metadata snapshot, recovers original `set == "0"`
membership and species/genus/family labels, and hashes all 632,913 test images.
It does not resplit the data. Concurrent readers overlap cold WEKA reads to warm
the cache while hashing: by default twice the available CPU affinity (96 readers
for 48 CPUs), bounded to 8–256 readers. Override with `--hash-workers` if needed.
Progress prints every five seconds, including while reads are waiting. Completed
hashes are checkpointed to `image-hashes.sqlite3` and reused on retry when the
input identity and file size/modification time match; changed files are rehashed.
The final manifest is published only after every image completes. This first
preparation pass reads the entire test set; run only one preparation process per cache.

The old `ucloud_env/uv.lock` remains available for reproducing earlier installs;
it is not the default installation path. Record the resolved environment after
qualification (`uv pip freeze`), and keep it unchanged during the campaign. Each phase records installed package versions
and source metadata and refuses to continue from qualification if they change.
The first fresh local resolution (24 September 2026, RTX 3080 Ti Laptop GPU)
passed PyTorch and ONNX inference on CPU and CUDA with default TTA, both
prediction/embedding modes, regional and custom lists. Both CUDA ONNX graphs used
full optimization without fallback. This is a four-image contract check, not a
quality or speed comparison. The original V2 pipeline also passed a four-image
CUDA qualification in the same environment. B200 qualification subsequently passed with the separate ONNX runtime described below.

| Dependency | Previous evaluation lock | Fresh laptop resolution |
| --- | --- | --- |
| PyTorch | 2.12.0+cu130 | 2.14.0+cu130 |
| torchvision | 0.27.0+cu130 | 0.29.0+cu130 |
| ONNX Runtime GPU | 1.30.0 | 1.30.0 |
| cuDNN | 9.20.0.48 | 9.24.0.43 |
| timm | 1.0.25 | 1.0.30 |

These are recorded results, not new installation pins. A fresh resolution on
UCloud may differ; retain its qualification evidence before drawing conclusions.

### Existing prepared campaign: reuse models and image hashes

With the new environment activated, **skip preparation** and run:

```sh
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/ucloud-release.json \
  --new-campaign ~/.cache/mambo-ucloud/runs-fresh-runtime
```

`--new-campaign` uses the active Python for all five variants and metrics, reuses
prepared assets, and writes `runs-fresh-runtime/config.json`. It requires a new
results directory, preserving previous evidence. For subsequent phases below,
use that config, `runs-fresh-runtime` and a separate summary directory. To retry,
use the saved config without `--new-campaign` and follow the resume rules below.
No model downloads or image rehashing are needed.

Keep this environment activated and unchanged throughout the campaign. Run
`python` directly; if using `uv run`, always add `--no-sync` to avoid an implicit
synchronization with the checkout's project environment.

Models, V2 heads and archived split provenance download automatically from public
ERDA storage with size/SHA-256 verification. The V2 BioCLIP backbone comes from its
pinned Hugging Face revision. Original V2 source is extracted from the pinned Git
commit, and run in a separate process with that source first on its import path.
The comparison therefore measures the released pipelines, not just their weights.

Preparation writes `~/.cache/mambo-ucloud/ucloud-release.json`. Use `--cache` to
choose a writable volume with room for models, image manifests and prediction CSVs.
Completed downloads and manifests are reused; `--offline` requires all downloads
to be cached. A checksum mismatch fails rather than silently replacing evidence.

## Qualify, collect, measure

Use the generated configuration path below (change it if using `--cache`). Review
its environment label, visible GPU, threads and batch sizes **before qualification**.
A MIG slice is one visible CUDA device; the default selects device `0`.

```sh
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/ucloud-release.json
```

This runs 256 identical test images through V2, V3 PyTorch, V3 ONNX, and each V3
backend with `rotation30_pad25_3` TTA. Inspect each log/report for successful
inference, runtime versions, placement and memory use. ONNX reports include
`onnx_session_info`: the graph optimization profile, initialization time and any
failed compatibility-probe attempts. A CUDA kernel-image/device-function failure
retries once with graph optimizations disabled; unrelated failures are not retried.
Both profiles retain CUDA, with individual CPU operators still allowed. A failure
of the unoptimized graph is reported as baseline incompatibility. This is a compatibility
qualification, not a reliable quality estimate or a requirement for numerical
identity between backends. Confirm the assigned GPU/MIG profile (`nvidia-smi -L`),
CPU allocation and storage mount alongside the generated environment label.

After qualification:

```sh
python -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/ucloud-release.json
python -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs/full
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/ucloud-release.json
python -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs \
  --output ~/.cache/mambo-ucloud/summary
```

`--dry-run` prints the planned commands without accessing images. `--resume`
verifies and reuses completed jobs; preserve/move partial job directories before
retrying. Changed code, inputs or configuration require a new output campaign and
qualification. A process starting is not completion: check `plan.json` status.

## Watch an existing collection

The standalone [monitor](../../monitor_mambo_release.py) reads only the phase's
plan, reports and the last 64 KiB of each log. It needs no model packages and works
with logs from runs started before the monitor was added:

```sh
python dev/monitor_mambo_release.py ~/.cache/mambo-ucloud/runs-ptx/full
```

It shows job completion, image counts, percentages, recent images/second and
estimated time remaining for the current job at its last checkpoint. Leave it
running: ETA requires two observed progress checkpoints (the collectors log every
50 batches in older collectors; the concurrent V3 collector logs approximately every five seconds). Initialization has no image ETA. Stale
checkpoints suppress ETA; finalization is not complete until the report confirms
it. Different variants have different throughput, so no whole-campaign ETA is
inferred from the current model. This monitors collection/qualification, not metrics
reduction or benchmark timing cells. Ctrl-C stops only the monitor.

**For a campaign already running, keep its checkout and environments unchanged.**
After the monitor commit has been pushed, fetch and extract just this standalone
file on UCloud; do not pull the new revision into the running campaign checkout:

```sh
git fetch origin release/mambo-v3
git show FETCH_HEAD:dev/monitor_mambo_release.py > /tmp/monitor_mambo_release.py
python /tmp/monitor_mambo_release.py ~/.cache/mambo-ucloud/runs-ptx/full
```

Fetching leaves the checked-out revision unchanged. Subsequent full/benchmark
phases can continue using the already qualified checkout. `--once` prints a single
status snapshot; it cannot infer throughput from old logs without timestamps.

For result transfer after completion, retain the campaign configuration, each
phase's `plan.json`, per-job `report.json` and `samples.json`, generated
`metrics.json` files, logs and summary outputs. Keep the prediction
`mini_metric.csv` files too: compressed copies permit additional mini_metrics
analyses and reproduction locally. Dataset images and downloaded model archives
are not required for documentation integration. Package completed evidence only;
transfer instructions and completeness checks will follow once collection and
metrics/benchmark phases finish.

## Evidence scope

Quality uses the **global vocabulary** and every original test image. All predictive
metrics come from pinned `mini_metrics`: macro accuracy/precision/recall/F1, micro
accuracy, Theil U and coverage, at species/genus/family, for all truth and known
truth separately. Predictions are unthresholded. Do not optimize thresholds on this
test split; any later calibrated comparison must freeze thresholds from separate
data and report acceptance coverage. Regional quality can be added explicitly,
but is ancillary because geographic restriction excludes part of this global set.

Speed uses CPU and GPU, global and northern-Europe lists, three isolated process
trials, two warmups and seven observations per cell. CPU batches default to 1/8;
GPU batches to 1/8/32. Each uses the same deterministic image bank (at least 32
images, expanded to the largest requested batch). The one-time ONNX synthetic probe is included in cold first-use timing and excluded
from warmed timing cells. Do not combine throughput from different selected
optimization profiles without labelling them. Timings include image loading,
preparation, transfer and completed CPU predictions; they describe warm repeated
inference, not cold storage throughput. Do not run competing collections during
benchmarking. V2 CPU includes the documented float32 input adapter required for
that pipeline; there is no qualified V2 ONNX artifact in this comparison.

The summary exports labelled quality/speed CSVs plus the runtime evidence, raw
timings and process peak host memory. Retain the job reports for GPU memory and
placement details. Keep **UCloud and laptop results as separate environment panels**
in future charts, consistently using images/second. Default UCloud timings use
in-domain images; differences from Flemming laptop timings cannot be attributed
solely to hardware. For a like-for-like hardware comparison, set `timing_manifest`
and `timing_root` to the same Flemming inputs before qualification.

Full UCloud quality and speed results are not yet available. Preserve current laptop figures;
add UCloud quality and speed figures only after the completed evidence passes the
summary checks. Cross-OS support and clean CUDA installation remain separate
qualification tasks.

## B200 runtime candidate: PTX-enabled upstream wheel

The tested Linux CPython 3.13 ORT 1.30.0 CUDA provider has no SM 100 kernels
and no PTX; its SHA-256 is
`afd77f8d1e05544456476e244601ff08d444ff90921d6d73b2066c124f109bd2`.
The same binary failed standalone CUDA Sigmoid on B200, independently of MAMBO.
Fresh dependency resolution retained that binary and did not fix the failure.

The upstream ORT 1.22.0 wheel is a bounded compatibility candidate: inspection
with CUDA 12.9 cuobjdump found 158 generic SM 90 PTX units, including FP32
Sigmoid and QuickGelu. Generic PTX provides a path to newer architectures through
[driver compilation](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html#application-compatibility-on-blackwell-architecture).
On 25 September 2026 it passed standalone Sigmoid and the four-image MAMBO
contract check on the RTX 3080 Ti Laptop GPU: both ONNX graphs, default TTA,
embeddings and regional/custom masks, with full optimization and no retry.
The user also confirmed the standalone CUDA Sigmoid probe passes on B200.
The user subsequently confirmed full-model B200 qualification passed for all five variants. Full quality/speed results remain outstanding.

After pulling the helper, test on B200 from the checkout root in an isolated
environment; this leaves the CUDA 13 PyTorch campaign environment intact:

```sh
uv venv --python 3.13 /tmp/mambo-ort-ptx
source /tmp/mambo-ort-ptx/bin/activate
uv pip install 'onnxruntime-gpu[cuda,cudnn]==1.22.0' numpy
python dev/releases/mambo_v3/probe_onnx_cuda.py
```

These extras install the candidate's CUDA 12 libraries. The exact version selects
the inspected upstream wheel; it is not a general deployment pin. Keep it separate from the CUDA 13 PyTorch environment.

### Continue after the B200 probe passes

Install only the deployment package into the existing ONNX environment, then run
qualification from the PyTorch environment. The optional `onnx_python` setting
routes both ONNX variants to the isolated interpreter, including full collection
and CPU/GPU benchmarks. Its installed dependencies are included in the campaign
fingerprint. Existing configurations without it continue to use `v3_python`.

```sh
uv pip install --python /tmp/mambo-ort-ptx/bin/python ./deployment
source .venv-mambo-runtime/bin/activate
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/runs-fresh-runtime/config.json \
  --onnx-python /tmp/mambo-ort-ptx/bin/python \
  --new-campaign ~/.cache/mambo-ucloud/runs-ptx
```

This preserves models, manifests and previous results. After successful
qualification, keep the PyTorch environment activated and run:

```sh
python -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/runs-ptx/config.json
python -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs-ptx/full
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/runs-ptx/config.json
python -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs-ptx \
  --output ~/.cache/mambo-ucloud/summary-ptx
```

The ONNX environment is in `/tmp` for this experiment; retain it for the campaign's
lifetime. Recreate and requalify it if the node's temporary storage is discarded.

## Restart V3 collection with concurrent preparation

For the 48-vCPU B200 allocation, start with **16 preparation workers and two
prefetched batches**. V3 collection now reads each image once for both SHA-256
verification and decoding, and prepares subsequent batches (including all TTA
views) while the main thread runs inference and writes predictions. Ordering,
preprocessing and prediction aggregation are unchanged. The queue bounds prepared
image memory; it does not cache the dataset. This targets IO latency and idle GPU
time without changing model precision, batch size or postprocessing.

Stop the old collection and cancel any queued shell follow-up commands **before
pulling this change**. Keep its results and both runtime environments. No package
installation or dataset preparation is needed. From the updated checkout:

```sh
source .venv-mambo-runtime/bin/activate
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/runs-ptx/config.json \
  --new-campaign ~/.cache/mambo-ucloud/runs-prefetch \
  --decode-workers 16 --prefetch-batches 2 \
  --reuse-v2-from ~/.cache/mambo-ucloud/runs-ptx
```

This inherits the separate ONNX interpreter. Completed V2 qualification/full
results are copied only after checking the invocation, inputs, environment,
relevant code and output hashes. V3 variants are requalified and recollected;
partial old V3 results remain untouched. After qualification succeeds:

```sh
python -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/runs-prefetch/config.json
python -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs-prefetch/full
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/runs-prefetch/config.json
python -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs-prefetch \
  --output ~/.cache/mambo-ucloud/summary-prefetch
```

Monitor from another terminal:

```sh
python dev/monitor_mambo_release.py ~/.cache/mambo-ucloud/runs-prefetch/full
```

V3 logs counts and throughput/ETA approximately every five seconds. Reports
separate `input_wait_seconds`, `runtime_seconds` and `reduce_write_seconds`;
`prepare_worker_seconds` overlaps these and must not be added to them as elapsed
time. Runtime time includes first-use initialization. These counters help assess
whether preparation keeps inference supplied before changing worker or queue sizes.
In the continuous scheduler below, `--prefetch-batches 0` minimizes decoded
lookahead; encoded read-ahead remains active.

A 256-image laptop ONNX check (batch 32; four workers before, sixteen afterward)
produced byte-identical prediction CSVs, and byte-identical TTA embeddings.
Collection elapsed time decreased from 5.49 to 4.64 seconds without TTA and from
9.41 to 7.48 seconds with default TTA and embeddings. These short checks establish
local benefit and output preservation, not B200 throughput. B200 full collection
remains to be measured. The dedicated speed benchmarks still measure the deployment
API unchanged, not this prefetched evaluation collector; all benchmark variants
run afresh, including V2.

## Continuous IO and preparation campaign

This supersedes the batch-at-a-time collector above. The shared deployment
`prepared_stream` scheduler reads ahead independently of decoding, prepares ready
images across batch boundaries, and emits ordered batches. Collection and
`Predictor.predict_stream` use the same scheduler. Start aggressively on the
48-vCPU B200/WEKA allocation: **256 readers, 48 preparation workers, 1,024 outstanding
images, eight prefetched batches, 2 GiB encoded-byte budget**. These are explicit
campaign settings, not portable deployment defaults. They follow the
[prior storage evidence](../../../docs/training-workflow-postmortem.md#1-storage-behavior-invalidated-small-subset-extrapolation).
Prepared views have a separate count bound; TTA multiplies their memory cost.

Stop collection and pending shell follow-ups, push/pull the implementation, then
reuse the existing environments and original completed V2 evidence:

```sh
source .venv-mambo-runtime/bin/activate
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/runs-prefetch/config.json \
  --new-campaign ~/.cache/mambo-ucloud/runs-streaming \
  --reuse-v2-from ~/.cache/mambo-ucloud/runs-ptx \
  --read-workers 256 --decode-workers 48 --read-window 1024 \
  --prefetch-batches 8 --encoded-budget-mib 2048
python -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/runs-streaming/config.json
python -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs-streaming/full
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/runs-streaming/config.json
python -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs-streaming \
  --output ~/.cache/mambo-ucloud/summary-streaming
```

Monitor with `python dev/monitor_mambo_release.py
~/.cache/mambo-ucloud/runs-streaming/full`. Inspect the first roughly two minutes
of steady V3 collection before queuing the later phases. Logs report reading,
encoded-ready, preparing and prepared-image counts, reserved encoded bytes and
cumulative input-wait time. Compare changes in wait time over that interval; model
initialization and the initial fill are not steady-state evidence. If preparation
still starves inference with spare CPU capacity, the next bounded candidate is
512 readers, keeping the window and byte budget unchanged. A different setting
requires a new campaign; do not edit a qualified config mid-run.

The old single-request timing cells remain unchanged. Additional streaming cells
use 1,024 images, the largest requested batch per device and preset, and three
observations per fresh-process trial. They include stream startup, IO, preparation,
inference, reduction and drain; integrity verification occurs before timing.
They use repeated inputs and describe warm storage, not cold WEKA throughput.
`streaming_speed.csv` exports these separately; V2 retains its original API timings
and has no new streaming cell. Keep sample-bank and execution-mode differences
visible when presenting results. Peak process memory includes both benchmark modes.

Local validation: the aggressive settings retained byte-identical prediction CSVs
and embeddings on a 256-image ONNX CUDA/default-TTA check. That short laptop run
took about 10.8 seconds (the earlier smaller pool took 7.5 seconds); it does not
establish a speed gain on B200. The UCloud run must establish the throughput benefit.
No dependencies changed; release scripts import deployment code from the checkout.

### Move batch assembly off the consumer thread

The first streaming run on B200 showed 736 encoded and 256 prepared images queued,
while the old `input_wait_seconds` increased by about 8.6 seconds over a 25-second
interval. That counter included serial batch stacking on the inference thread;
it did not isolate storage waiting. Assembly now runs in a separate worker,
including release of per-image buffers. Ordered, contiguous batches are delivered
to inference without stacking there. Reader and preparation settings stay unchanged.

Logs now show **interval** images/s and separate interval seconds for input delivery,
runtime calls and reduction/writing, plus background assembly. Background assembly
overlaps consumer work and must not be added to its timings. The pipeline counter
`queue_wait_seconds` (also exposed as `input_wait_seconds`) measures waiting for a
complete batch, including initial fill. `prepared_batches` counts complete queued
batches, and `assembling` counts images being assembled. Initial runtime loading
is still included in the first runtime interval.

Stop the old run and queued follow-ups before pulling. Reuse the same environments,
concurrency settings and completed V2 evidence; no installation is needed:

```sh
source .venv-mambo-runtime/bin/activate
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/runs-streaming/config.json \
  --new-campaign ~/.cache/mambo-ucloud/runs-assembly
python -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/runs-assembly/config.json
```

After inspecting roughly two minutes of steady V3 progress, continue with:

```sh
python -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs-assembly/full
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/runs-assembly/config.json
python -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs-assembly \
  --output ~/.cache/mambo-ucloud/summary-assembly
```

The standalone monitor takes `~/.cache/mambo-ucloud/runs-assembly/full`. Both full
collection and deployment streaming benchmarks use the corrected assembly path.

### Batch 256 with overlapped result processing

Use this campaign for the next B200 run. All four V3 variants collect at batch
**256**, without a batch-size search. The original V2 collection invocation and
completed results remain unchanged. V3 GPU benchmarks retain their existing request
sizes and add 256; their streaming cell uses 256. All variants use the same enlarged
request timing image bank, while V2 retains its original batch sizes.

Top-1 selection now uses a maximum instead of sorting every class. Confidence
normalization and hierarchical reduction are unchanged. Collection and the public
streaming API process results in a single ordered background worker with at most
two batches outstanding. CSV output order and failure propagation are preserved;
completion counts advance only after results have been written. Final reports are
published after the output queue drains.

Keep 256 readers, 48 preparation workers and the 1,024-image read window. Use **one
prepared batch ahead** at batch 256: the preparation window holds at most 512
images plus assembly temporaries, rather than nine batches of 256. TTA multiplies
prepared-image storage. The encoded budget remains 2 GiB.

Stop the previous run and queued commands before pulling this change. No venv
update is needed. The new campaign inherits runtime paths and V2 reuse:

```sh
source .venv-mambo-runtime/bin/activate
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config ~/.cache/mambo-ucloud/runs-assembly/config.json \
  --new-campaign ~/.cache/mambo-ucloud/runs-batch256 \
  --v3-batch-size 256 --prefetch-batches 1
python -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/runs-batch256/config.json
python -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs-batch256/full
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/runs-batch256/config.json
python -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs-batch256 \
  --output ~/.cache/mambo-ucloud/summary-batch256
```

Watch `~/.cache/mambo-ucloud/runs-batch256/full/torch.log`. Interval timings now
separate `hierarchy_seconds`, `prediction_seconds` and `write_seconds` for completed
background jobs. They overlap inference and must not be added to foreground times.
`output_wait_seconds` measures consumer backpressure while retiring results.
B200 qualification exercises batch 256; local output checks use laptop-sized
batches and do not establish B200 memory use or throughput.
