# UCloud release comparison

Run this workflow **inside a manually allocated UCloud SSH node** with the original
global-lepi dataset mounted. Clone this release branch with its Git history there;
the scripts live in `dev/releases/mambo_v3` and are not included in the deployment
wheel. No allocation or remote submission is performed by these commands.

## Setup with uv

From the checkout root:

```sh
uv sync --project dev/releases/mambo_v3/ucloud_env --locked
uv run --project dev/releases/mambo_v3/ucloud_env --no-sync python \
  -m dev.releases.mambo_v3.setup_ucloud_release \
  --metadata /work/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet
```

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

The isolated uv project pins the metric implementation and records exact runtime
dependencies in its lockfile,
with an [explicit CUDA PyTorch index](https://docs.astral.sh/uv/guides/integration/pytorch/).
It creates its own environment without changing the checkout's `.venv`. The current
lock targets Python 3.13 and CUDA 13.0; qualify the node's driver/runtime before a
full run. ONNX Runtime requests its own matching CUDA/cuDNN dependencies. The
CUDA-13 evaluation project allows ORT 1.27 or newer within major version 1; the
consumer package also allows older CUDA runtime families. These are dependency
ranges, not a claim that every allowed version/device combination has been tested.
The locked ORT 1.30.0 build failed CUDA qualification on the allocated B200; this
dependency-policy change does not establish a fix for that failure.

Use uv’s normal targeted resolution to select another runtime version without
editing the dependency declaration by hand:

```sh
uv lock --project dev/releases/mambo_v3/ucloud_env \
  --upgrade-package onnxruntime-gpu==1.29.0
uv sync --project dev/releases/mambo_v3/ucloud_env --locked
```

Here 1.29.0 illustrates version selection, not a B200-qualified recommendation.
Omit `==1.29.0` to request the newest version allowed by the project. Keep the
resulting lockfile with the campaign, and start fresh qualification after changing
it; reports retain the runtime actually used. The resolver retains other locked
versions where constraints permit. Existing models and image hashes are reusable.

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
uv run --project dev/releases/mambo_v3/ucloud_env --no-sync python \
  -m dev.releases.mambo_v3.ucloud_release qualification \
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
uv run --project dev/releases/mambo_v3/ucloud_env --no-sync python \
  -m dev.releases.mambo_v3.ucloud_release full \
  --config ~/.cache/mambo-ucloud/ucloud-release.json
uv run --project dev/releases/mambo_v3/ucloud_env --no-sync python \
  -m dev.releases.mambo_v3.metrics \
  --collection ~/.cache/mambo-ucloud/runs/full
uv run --project dev/releases/mambo_v3/ucloud_env --no-sync python \
  -m dev.releases.mambo_v3.ucloud_release benchmark \
  --config ~/.cache/mambo-ucloud/ucloud-release.json
uv run --project dev/releases/mambo_v3/ucloud_env --no-sync python \
  -m dev.releases.mambo_v3.ucloud_summary \
  --root ~/.cache/mambo-ucloud/runs \
  --output ~/.cache/mambo-ucloud/summary
```

`--dry-run` prints the planned commands without accessing images. `--resume`
verifies and reuses completed jobs; preserve/move partial job directories before
retrying. Changed code, inputs or configuration require a new output campaign and
qualification. A process starting is not completion: check `plan.json` status.

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

No UCloud inference results are claimed yet. Preserve current laptop figures;
add UCloud quality and speed figures only after the completed evidence passes the
summary checks. Cross-OS support and clean CUDA installation remain separate
qualification tasks.
