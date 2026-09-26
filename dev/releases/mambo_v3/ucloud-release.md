# UCloud release comparison

Run inside a manually allocated UCloud SSH node with the original global-lepi
dataset mounted. The scripts live in this repository, not the deployment wheel;
clone the reviewed release revision with Git history. Install uv, create/activate
the environment, then run the experiment. Use `tmux` and persist results under
`/work`. These commands do not allocate a node.

The completed campaign is documented in [in-domain evidence](../../../docs/mambo-indomain-evidence.md).
For a small throughput check rather than a full evaluation, use
[speed-smoke.md](speed-smoke.md). Do not repeat the full campaign to validate
unchanged quality or a documentation edit.

## Setup with uv

From the checkout root after installing uv and selecting the reviewed revision:

```sh
uv venv --python 3.13 .venv-mambo-runtime
source .venv-mambo-runtime/bin/activate
uv pip install --torch-backend=auto -r dev/releases/mambo_v3/runtime-requirements.in
python -m dev.releases.mambo_v3.setup_ucloud_release \
  --metadata /work/datasets/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet \
  --cache /work/mambo-cache
```

Runtime dependencies resolve afresh within package ranges; only `mini_metrics` is
fixed to preserve metric semantics. The historical `ucloud_env/uv.lock` remains a
reproduction record, not the default deployment install. PyTorch backend selection
and ORT CUDA/cuDNN dependencies still require real GPU qualification. Keep the
resolved environments unchanged during a campaign; use activated `python` or
`uv run --no-sync`. No implicit checkout-environment synchronization is needed.

Preparation requires only the metadata path; images default to
`images/<species>/<filename>` beneath its parent (`--root` overrides). It verifies
the metadata snapshot and preserves original `set == "0"` membership and taxonomy.
Models, legacy heads and split provenance download from public ERDA with size/hash
verification; the BioCLIP backbone uses a pinned Hugging Face revision. V2 source
is extracted from its pinned Git commit and run in a separate process.

All 632,913 test images are hashed with concurrent readers, warming WEKA as they
read. Default is twice CPU affinity, bounded 8–256; `--hash-workers` overrides it.
That default may need adjustment to the allocation quota. Progress appears every
five seconds, including stalls. Completed hashes checkpoint to SQLite and are reused
when input identity, size and modification time agree. The final manifest appears
only when all images finish. Run one preparation process per cache; `--offline`
requires cached downloads. Checksums fail explicitly instead of replacing evidence.

The chosen cache contains models, manifest, hash checkpoint and
`ucloud-release.json`. It is separate from the persistent campaign directory below.
Inspect config paths, environment label, device, threads and batches before starting.

## Qualify, collect, measure

Create one named campaign. `--new-campaign` binds the active interpreter, reuses
prepared assets and saves `config.json` in the new output directory:

```sh
python -m dev.releases.mambo_v3.ucloud_release qualification \
  --config /work/mambo-cache/ucloud-release.json \
  --new-campaign /work/mambo-results/current
```

Qualification runs 256 identical images through V2, V3 Torch, V3 ONNX and both V3
backends with `rotation30_pad25_3` TTA. Inspect reports/logs and `nvidia-smi -L` for
the actual GPU/MIG profile, placement and memory. One MIG slice is one CUDA device.
This is runtime qualification, not a reliable quality estimate or a requirement
for micro-numerical equality.

**Use the saved campaign config for every remaining phase:**

```sh
python -m dev.releases.mambo_v3.ucloud_release full \
  --config /work/mambo-results/current/config.json &&
python -m dev.releases.mambo_v3.metrics \
  --collection /work/mambo-results/current/full &&
python -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config /work/mambo-results/current/config.json &&
python -m dev.releases.mambo_v3.ucloud_summary \
  --root /work/mambo-results/current \
  --output /work/mambo-results/summary
```

The chain stops on failure. It does not wait for a separately launched collection:
queue it in the same shell after qualification/full succeeds. `plan.json` must say
`complete`; a started process or existing output file is not completion.
`--dry-run` prints planned jobs; `--resume` verifies/reuses complete jobs.
Preserve partial job directories elsewhere before retrying. Fingerprints reject
changed source, inputs, configuration or environments within an existing campaign.

### Reuse prepared assets or change settings

On an existing job, skip setup/hashing and use the prior config with a **new**
campaign directory. Do not pull changes into an actively running checkout or alter
its environments; stop the run and queued follow-ups first, or use another checkout.
Harness-only changes need no environment rebuild; installed package/dependency
changes need an explicit install and requalification.

Relevant `--new-campaign` options:

| Option | Applies to |
| --- | --- |
| `--onnx-python PATH` | Separate runtime for both ONNX variants and their benchmarks |
| `--v3-batch-size N` | V3 collection; benchmark planning also includes this GPU batch |
| `--decode-workers N` | V3 preparation concurrency |
| `--read-workers N`, `--read-window N` | Outstanding IO capacity/lookahead |
| `--prefetch-batches N`, `--encoded-budget-mib N` | Bounded preparation/encoded storage |
| `--no-device-prefetch` | Collection and streaming transfer-staging comparison |
| `--reuse-v2-from DIRECTORY` | Verified V2 qualification/full outputs; never benchmark reuse |

The full-B200 smoke used batch 256, 48 preparation workers, 128 readers, a 4,096-image
window, two prefetched batches and 1 GiB encoded budget. Those are measured settings
for a 48-vCPU/full-GPU allocation, not defaults for MIG or arbitrary machines.
Use the saved prior config to preserve its settings; see CLI `--help` for overrides.
After changing anything, use the new config consistently for full/metrics/benchmark
and summary rather than editing paths in each historical campaign recipe.

## ONNX CUDA compatibility and the recorded B200 exception

ORT session qualification probes a synthetic batch once per graph initialization.
A kernel-image/device-function failure retries with graph optimizations disabled.
Unrelated errors are not retried; failure of that baseline graph reports runtime/
device incompatibility. Both profiles retain CUDA with individual CPU operators
allowed. Reports retain `onnx_session_info` and failed attempts. The probe is part
of cold first use, excluded from warmed timing cells.

On the recorded Linux CPython 3.13 B200 setup, ORT 1.30.0's provider binary
SHA-256 `afd77f8d1e05544456476e244601ff08d444ff90921d6d73b2066c124f109bd2`
lacked SM100 kernels and PTX. Standalone Sigmoid failed independently of the model,
including with optimization disabled. Fresh resolution retained that binary, so
lock relaxation alone did not repair it.

The inspected upstream ORT 1.22.0 wheel included generic SM90 PTX and passed the
standalone probe and full model qualification on B200, plus bounded laptop checks.
That supports this runtime candidate, not a universal version recommendation.
Keep its CUDA 12 libraries isolated from the CUDA 13 Torch environment:

```sh
uv venv --python 3.13 /work/venvs/mambo-onnx
uv pip install --python /work/venvs/mambo-onnx/bin/python \
  'onnxruntime-gpu[cuda,cudnn]==1.22.0' numpy ./deployment
/work/venvs/mambo-onnx/bin/python dev/releases/mambo_v3/probe_onnx_cuda.py
```

If needed, supply `--onnx-python /work/venvs/mambo-onnx/bin/python` when creating
the named campaign above. Run orchestration/metrics from the main activated
environment. Both interpreter inventories enter the fingerprint. The historical
separate-runtime campaign completed; it is not still awaiting results.

## Progress and timing interpretation

```sh
python dev/monitor_mambo_release.py /work/mambo-results/current/full
```

The standalone monitor reads plans/reports and log tails without model packages.
ETA needs two observed checkpoints; startup/finalization have no reliable image
ETA, and stale checkpoints suppress estimates. Different variants run at different
rates, so current-job ETA is not a whole-campaign forecast. `--once` gives status;
Ctrl-C stops only the monitor. It does not track metrics reduction or benchmark cells.

Pipeline counters describe occupancy, input waits, buffer allocations and worker
durations. `preparation_worker_seconds` sums concurrent work and can exceed wall
time. `runtime_submit_seconds` excludes Torch output completion waits but includes
ONNX's synchronous runtime; CUDA stream intervals are not kernel-only time or GPU
utilization. Do not add overlapping phases or interpret low host storage-read bytes
as proof of absent network-storage waits. Pinned storage consumes host RAM.
[Pipeline decisions](../../../docs/mambo-inference-pipeline-review.md) explain the
current ownership boundaries and remaining performance limits.

## Evidence and metric policy

Quality uses every original test image with global vocabulary. All metrics come
from pinned `mini_metrics`: macro accuracy/precision/recall/F1, micro accuracy,
Theil U and coverage at species/genus/family, for all and known truth. The base
`metrics` command reports unthresholded results. The presentation path adds the
same calibration/reporting split and full/support >5 conventions used for Flemming:

```sh
python -m dev.releases.mambo_v3.indomain_report \
  --root /work/mambo-results/current \
  --output /work/mambo-results/presentation
python -m dev.releases.mambo_v3.indomain_speed \
  --source /work/mambo-results/summary \
  --output /work/mambo-results/presentation
```

`mini_metrics` selects thresholds on a disjoint calibration portion and evaluates
them on reporting rows; do not fit on the reporting portion. Retain coverage and
class-support domains alongside thresholded metrics. See
[in-domain evidence](../../../docs/mambo-indomain-evidence.md) for exact policy and
[reusable evidence](evidence-policy.md) for identities required in future comparisons.

Benchmark modes are distinct: request includes loading through completed predictions;
streaming includes its bank/startup/drain; prepared diagnostics exclude input
preparation. Keep warmups, repetitions, batch/list, threads and optimized-session
profile with each result. V2 CPU uses the documented FP32 adapter and has no
qualified ONNX counterpart. Do not benchmark concurrently with heavy collection.

Keep UCloud and laptop panels separate and use images/s consistently. Their image
domains differ, so differences cannot be assigned solely to hardware. For matched
hardware comparisons, configure `timing_manifest` and `timing_root` identically
before qualification. Latest B200 smoke values replace only corresponding measured
points; historical CPU/V2 points remain labelled.

## Persistent storage and transfer

All example campaign/summary/presentation outputs above are under `/work`.
For an older home-cache run, copy only completed results to a mounted location;
preserve its config, all phase plans/reports/logs, samples, predictions and metrics:

```sh
mkdir -p /work/mambo-results
rsync -a ~/.cache/mambo-ucloud/runs-transfers/ /work/mambo-results/runs-transfers/
rsync -a ~/.cache/mambo-ucloud/summary-transfers/ /work/mambo-results/summary-transfers/
```

`rsync -a` permits interrupted copies to be resumed while preserving directory
structure; a verified `cp -a` transfer is also sufficient. Check phase completion,
report/prediction hashes and summary consistency before packaging. Retain
`mini_metric.csv` at every rank/list for later metrics; downloaded weights and
dataset photos are not needed to integrate the results into documentation.

```sh
tar -czf /work/mambo-results.tar.gz -C /work mambo-results
sha256sum /work/mambo-results.tar.gz
```

Transfer the archive and digest, verify after download, and retain immutable raw
results alongside derived tables/figures. Archive integrity is separate from
successful campaign completion and metric validity.
