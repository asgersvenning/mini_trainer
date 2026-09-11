# UCloud global_lepi training comparison

Run inside **one allocated node**, with 1, 2, 4 or 8 visible GPUs. More than one
GPU uses one `torchrun` process per GPU and the package's existing DDP trainer
(including SyncBatchNorm). One GPU uses ordinary training. Do not invoke the
launcher once per GPU or wrap it in a multi-task `srun`.

Allocate the job manually in the UCloud web interface, connect to that node over
SSH, install the environments below, then run the launcher from the shell. Local
Slurm inside the allocation is unnecessary and cannot allocate further nodes.
Use `tmux` or another persistent shell if an SSH disconnect would stop the process;
this does not extend the allocation's lifetime.

UCloud's [PyTorch app](https://docs.cloud.sdu.dk/Apps/pytorch.html) supports a Bash
batch script and selection of GPU machine types. Mounted volumes appear under
[`/work`](https://docs.cloud.sdu.dk/guide/submitting.html). Use the actual paths
shown in your job. No Gefion account, partition, GPU type or Slurm directives are
assumed here. The scripts perform no installation, submission or paid allocation.

## Comparison and controls

The source configuration is `publication/experiments/config_temp.yaml` (singular
`publication`), a local experiment file. This protocol keeps 10 epochs, FP16 AMP,
class weighting, 0.25 warmup epochs and eight loader workers per rank. It replaces
ViT with torchvision `efficientnet_v2_s` (EfficientNetV2-S) and fixes the
`HierarchicalClassifier` head, species/genus/family, equal loss weights, normalized
prototypes, hidden layer, dropout 0.1, MuonAuxAdamW, learning rate 0.001, weight decay
0.01, regularizer 0.1 and automatic smoothing. Resizing is explicitly 384 pixels,
the backbone's preferred crop size. The initial pretrained backbone and randomly
initialized head are saved **once per seed by master**, then loaded by both branches.
Smoothing values per hierarchy level are recorded in `dataset.json`.

**Batch size 64 is global**, so it becomes 32/16/8 per rank for 2/4/8 GPUs. Learning
rate is not scaled. Changing GPU count still changes SyncBatchNorm, sampling and
kernel behavior; compare rows within an allocation, not as identical trajectories
across GPU counts. Three paired seeds are the default. Run order is shuffled within
each seed using a reproducible schedule. RNG streams are not guaranteed identical
across implementations, but starting model tensors are identical. No resume or
checkpoint averaging is used in measured runs.

| Variant | Branch | Change from eager control |
| --- | --- | --- |
| `master_eager` | master | Baseline |
| `quant_eager` | quant | Branch changes with all optional acceleration off |
| `quant_prefetch` | quant | CUDA transfer lookahead |
| `quant_compile_model` | quant | Default model compilation |
| `quant_compile_optimizer` | quant | Optimizer compilation |
| `quant_combined` | quant | Prefetch plus both compilation options |

Optional variant names: `quant_model_graphs` (reduce-overhead model mode),
`quant_optimizer_graphs` (optimizer graph replay), and **`quant_int8` (one GPU
only)**. For a single-GPU INT8 comparison, set `gpus: 1` and include at least
`master_eager`, `quant_eager`, `quant_int8`. The runner rejects INT8 with DDP.
These are target-machine qualification experiments; a supported flag is not a
guarantee that a compiled configuration succeeds or is faster on this hierarchy.
Failures remain visible. No automatic fallback is used.

Muon, normalized heads, EMLA/class weighting, regularization and smoothing already
exist on the pinned master. They are held constant rather than mislabeled as new
quant features. Their quality/tuning ablations belong to the separate
[feature-validation protocol](../../docs/training-feature-validation.md).
EMA is disabled. CPU PTQ/QAT are inference experiments. Full dataset RAM caching
is excluded because it replicates the dataset per DDP rank and can exhaust node
memory; both branches use uncached image loading. W&B is disabled to avoid requiring
credentials; local logs and checkpoints are retained.

## Fresh job setup

For the first bounded test, select one MIG in UCloud and mount the global_lepi
dataset. Internet access is needed for the source checkout, Python dependencies
and the first pretrained-weight download. CPU and RAM come with the selected
GPU product; the host totals reported inside the container are not your budget.

Clone the branch containing the current harness, then install uv if the image
does not already provide it. Follow the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone --branch quant https://github.com/asgersvenning/mini_trainer.git /work/mini_trainer
# Only if `uv --version` is unavailable:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /work/mini_trainer
bash dev/ucloud/setup.sh /work/YOUR_DATASET/YOUR_METADATA.parquet
```

The setup script reads the pinned commits from `qualification.json`, exports the
quant commit's lockfile into `/work/requirements-mt.txt`, creates two Python 3.12
environments and checks their dependencies. It uses explicit indexes and required
hashes, stops on the first error, and generates `/work/qualification.json` with the
supplied dataset path and a new smoke-test output path. It does not start training
or touch the dataset. No editable package is installed, and the checkout stays on
the harness branch. Do not reset it to the older pinned package commits.

`MT_WORK_ROOT` overrides `/work`, `MT_CONFIG` chooses another generated config path,
and `MT_TORCH_BACKEND` selects `cu126`, `cu130` (default) or `cu132`. A failed setup can
be rerun: it synchronizes only its dedicated `venvs/mt-master` and `venvs/mt-quant`
environments, removing leftover packages there. An existing generated config is
never overwritten. The UV download cache defaults to `/work/.cache/uv`; installation
time and disk space depend on the node and network and are outside the test budget.

Once setup succeeds:

```bash
cd /work/mini_trainer/dev/ucloud
export TORCH_HOME=/work/.cache/torch
bash launch.sh /work/qualification.json --stage plan
bash launch.sh /work/qualification.json --stage prepare && \
    bash launch.sh /work/qualification.json --stage train
bash launch.sh /work/qualification.json --stage summary
```

Preparation and both eager runs share a 30-minute wall-clock budget. It does not
terminate the UCloud job; choose the allocation lifetime separately, allowing for
installation. See the qualification and resource discussion below before expanding
the matrix. The manual setup below remains available for full-dataset comparisons.

## Fresh environment setup (manual)

Use the same Python and exact dependency versions for both packages. The runner
checks installed VCS commit IDs and dependency inventories. No editable installs:
running from the harness directory must import the selected installed branch.
The node image must provide a working NVIDIA driver and a C++ compiler (`c++`, or
the executable named by `CXX`); the preflight checks both CUDA execution and compiler
availability. Python wheels do not provide the NVIDIA kernel driver.
The default commits are the local branch tips inspected when this harness was
prepared, not a claim about the latest remote branches:

```bash
MASTER_SHA=2ebbe39f2365bfea482d7ebc3d000cf4d2044918
QUANT_SHA=4ea6b9613fc29a8e33ef72f28885f4e2a0e5906b
REPO_URL=https://github.com/asgersvenning/mini_trainer.git
```

In a checkout of `QUANT_SHA`, export its lockfile with an explicit CUDA backend.
Choose `cu126`, `cu130` or `cu132` according to the allocated GPU and driver; do
not choose CPU wheels. `cu130` below is an explicit example, not GPU qualification.
Keep the NVIDIA driver information from `nvidia-smi` with your results.

```bash
uv export --locked --no-dev --no-emit-project --extra recommended --extra cu130 --extra export --extra quantization --output-file /work/requirements-mt.txt
uv venv --python 3.12 /work/venvs/mt-master
uv venv --python 3.12 /work/venvs/mt-quant
export UV_LINK_MODE=copy
unset UV_TORCH_BACKEND
uv pip install --python /work/venvs/mt-master/bin/python --index https://pypi.org/simple --default-index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-first-match --require-hashes -r /work/requirements-mt.txt
uv pip install --python /work/venvs/mt-quant/bin/python --index https://pypi.org/simple --default-index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-first-match --require-hashes -r /work/requirements-mt.txt
uv pip install --python /work/venvs/mt-master/bin/python --no-deps "mini_trainer @ git+$REPO_URL@$MASTER_SHA"
uv pip install --python /work/venvs/mt-quant/bin/python --no-deps "mini_trainer @ git+$REPO_URL@$QUANT_SHA"
uv pip check --python /work/venvs/mt-master/bin/python
uv pip check --python /work/venvs/mt-quant/bin/python
```

Both pinned commits must be available in the Git repository you install from.
At preparation time the public `quant` tip was `f5c69e7cab2bfde8a5467026b293858b93e628f9`;
the pinned local tip contains two later commits. Publish those commits before
using the HTTPS installation above, or transfer a Git bundle without publishing:

```bash
# On the development machine; transfer this bundle and dev/ucloud to UCloud.
git bundle create /tmp/mini-trainer-ucloud.bundle master quant
# On UCloud, after transferring it:
git clone /work/mini-trainer-ucloud.bundle /work/mini-trainer-source
REPO_URL=file:///work/mini-trainer-source
# Then use the same pinned uv pip install commands above with this REPO_URL.
```

Keep the harness separately: it was added after the pinned quant commit and is
not installed by `uv pip install mini_trainer`. Use a checkout containing
`dev/ucloud/launch.sh`, not a checkout reset to `QUANT_SHA`, for the copy below.
Save the exported requirements.
The export and quantization extras are installed equally in both environments so
optional follow-ups do not change the dependency comparison.
Use the same backend in the export's `--extra` and the installs' explicit CUDA
index URL. Exported requirements do not carry the custom PyTorch index.
PyPI is searched first, then the CUDA index when the pinned version is absent.
Keep hashes enabled. Avoid `--torch-backend` for this exported lock: it also
redirects `torchao`, which is locked from PyPI, and can trigger an unpinned
requirement error in a fresh environment. Stop on any installation failure before
continuing to the next command. The copy link mode avoids cross-filesystem
hardlink warnings; it does not affect training speed.

Preparation downloads torchvision's pretrained weights if absent. For offline
jobs, populate a shared writable `TORCH_HOME` ahead of time using the same locked
torchvision version. Allow CPU RAM/disk for the full hierarchical initial models,
three seeds, optimizer checkpoints and compiled caches. Do not assume INT8 makes
the convolutional backbone or all optimizer state integer.

## Configure and launch

If you cloned a branch containing this harness directly onto the node, run from
that checkout's `dev/ucloud` directory. No `/work/comparison` directory or copy is
needed. Keep the checkout there; the launcher resolves its sibling scripts itself
and runs workers outside the checkout with the configured installed interpreters.
For example:

```bash
cd /work/mini_trainer/dev/ucloud
```

The following copy instructions are an alternative for transferring just the harness
when environments already exist. `setup.sh` requires a Git checkout with the pinned
package commits and is intended for the fresh-job workflow above.
First copy the harness and example configuration onto the node. If a checkout
containing `dev/ucloud` is already on the node, run from that checkout's root:

```bash
mkdir -p /work/comparison
cp -i dev/ucloud/launch.sh dev/ucloud/compare.py dev/ucloud/worker.py \
    dev/ucloud/export_followup.py dev/ucloud/comparison.json dev/ucloud/qualification.json /work/comparison/
```

Otherwise, run this from the checkout root on your development machine, replacing
`UCLOUD_SSH_HOST` with the SSH destination you use for the allocated node (and
adding your usual SSH port/key options if needed):

```bash
ssh UCLOUD_SSH_HOST 'mkdir -p /work/comparison'
scp dev/ucloud/launch.sh dev/ucloud/compare.py dev/ucloud/worker.py \
    dev/ucloud/export_followup.py dev/ucloud/comparison.json dev/ucloud/qualification.json UCLOUD_SSH_HOST:/work/comparison/
```

Copy once before configuring; preserve an already edited node configuration when
updating scripts. Creating `/work/comparison` alone does not populate it. Keep the
three Python scripts alongside `launch.sh`, which resolves them relative to itself.

On the node, edit `/work/comparison/comparison.json`: set `parquet`, `output`, both
Python paths and `gpus`.
The Parquet must sit beside `images/<speciesKey>/<filename>` as expected by
`mini_trainer`. Preparation checks every image path, rejects duplicates and missing
labels, freezes the taxonomy/index, and retains the existing `set` mapping:
`0=test`, `1=validation`, other numeric sets=train. Non-numeric sets are excluded by
the existing parser. No random re-splitting occurs. Keep image bytes immutable;
preparation checks file existence, not hashes or complete decoding of the images.

```bash
cd /work/comparison
bash launch.sh comparison.json --stage plan
bash launch.sh comparison.json --stage prepare
bash launch.sh comparison.json --stage train
bash launch.sh comparison.json --stage summary
```

`prepare` requires a **new** output directory and verifies CUDA in both environments.
It is CPU-heavy when parsing and initializing models. Start with the bounded
qualification profile below: one epoch of the full six-million-image dataset is
still a long run. Eight workers per rank can mean 64 workers on eight GPUs; reduce
consistently if the allocated CPU or shared-memory budget requires it.

### Bounded qualification on one MIG device

Use `qualification.json` for the first setup test. Edit its dataset path and
environment entries to match your node, and choose a new output path. It selects
one visible CUDA device, batch size 32, two loader workers, one epoch and one seed,
with both eager controls. The source splits are preserved, and a fixed seed selects
2,048 training, 512 validation and 128 test rows by uniform reservoir sampling within
each split. Both branches use exactly the same saved sample and starting weights.
Test images are checked for existence but are not used in training or validation.

This gives **64 training batches and 16 validation batches per branch**. Selection
still streams the source Parquet once, but retains only 2,688 rows plus one batch
in memory. Only selected image paths are checked. The taxonomy and classifier head
are built from the selected rows, reducing the pinned master's expensive normalized
head initialization. Duplicate/path validation covers the sample, not the entire
source dataset. The selected Parquet, mappings, index and seed are retained and
hashed with the other preparation artifacts.

This is an infrastructure smoke test, **not a full-taxonomy performance or quality
comparison**. Random sampling can leave validation species absent from the training
sample. Do not interpret its accuracy, loss weighting or timing as representative
of the complete dataset. Results carry `scope=qualification_subset`, including in
the paired report. Omitting `qualification` preserves the full-dataset workflow.

```bash
# From the node's checkout, after editing qualification.json:
cd /work/mini_trainer/dev/ucloud
bash launch.sh qualification.json --stage plan
bash launch.sh qualification.json --stage prepare && \
    bash launch.sh qualification.json --stage train
bash launch.sh qualification.json --stage summary
```

The profile's `budget_seconds: 1800` sets one wall-clock deadline beginning at
`prepare`, shared by preflights, preparation and both training runs. Time between
commands also counts; chain preparation and training as above. The launcher stops
active workers when the deadline expires (cleanup may take up to 15 additional
seconds), records an interrupted/timed-out run when applicable, and preserves all
logs. This is a time bound, not a promise that both runs finish on every allocation.
`timeout_seconds` additionally caps each worker call. Summary remains available
after the deadline. The budget does not stop the UCloud allocation or include manual
environment installation; set the job lifetime in UCloud separately.

After the eager pair passes, use another new output and add `quant_prefetch` first.
Test compilation variants separately: cold compilation can consume much of a short
budget. For timing comparisons, increase to at least two epochs to distinguish
first-use costs from subsequent execution, and keep sample seed, batch size,
image size, resource allocation and thread settings fixed across variants. Full
taxonomy, full-data convergence, INT8 and multi-GPU/DDP qualification remain separate.

### Job resources and bottlenecks

[UCloud's resource guide](https://docs.cloud.sdu.dk/guide/resources-products.html)
describes one MIG as one seventh of a B200 and allows one to four MIGs per job.
The UI's `4/7` selection provides four separate `1g.23gb` devices, not a single
device with four times the memory. CPU and system RAM allocations scale with the
selected product. Use the vCPU/RAM values shown for that selection; `nproc` and
`free` can expose host totals (384 vCPUs and roughly 2.2 TiB on this node type).
Keep the existing one-MIG allocation for the initial eager test; four devices
would additionally exercise DDP and would not accelerate the serial preparation.

The profile's batch size 32 leaves more headroom than the observed full-head
batch-64 run, which used about 16 GiB of a 20.5 GiB device. Keep 384-pixel inputs to
exercise the intended preprocessing. Two loader workers are a conservative starting
point for a fractional allocation. The reported 34 GiB `/dev/shm` is not a small
default shared-memory mount; there is no evidence so far that it needs increasing.
Avoid full-dataset RAM caching. If loaders later starve the GPU, compare two and
four workers within the actual CPU quota before increasing further; keep the same
worker count for paired branches. Container CPU quotas may be lower than affinity.

The launcher defaults to one CPU math thread and one compiler worker per rank.
Changing `OMP_NUM_THREADS` and `MKL_NUM_THREADS` before launch affects both branches;
two math threads can be tested if the allocated CPU budget leaves room alongside
the loaders. Do not set them from the host CPU count. Preflight records the actual
PyTorch thread count, visible CPU affinity count and shared-memory capacity. Keep
the pretrained weight cache (`TORCH_HOME`) on persistent writable storage to avoid
repeated downloads. If file I/O remains dominant, stage only the small sample's
images on confirmed fast node-local storage for a separate, identically staged
comparison; moving the full dataset is unnecessary for a smoke test.

### Preparation progress and interrupted runs

The launcher prints the preflight and preparation log paths. `prepare.log` includes
timestamped sampling, taxonomy, path-checking, index-writing, hashing and model
construction messages, with periodic counts while scanning and checking paths.
CPU model construction can still be silent inside the pinned package's initializer.
JSON is written incrementally, and per-image metadata is released before building
starting models to reduce preparation's temporary memory requirements.

```bash
output_dir=$(python3 -c 'import json; print(json.load(open("qualification.json"))["output"])')
tail -f "$output_dir/prepare.log"
```

`prepared.json` is the completion marker. Its absence can mean preparation is still
running; check the original launcher and worker processes before starting another
attempt. An existing output cannot be prepared again, and training now reports an
explicit incomplete-preparation error. Preserve an interrupted attempt and use a
new output path. Ctrl-C stops the worker process group and records `interrupted`
for an active training run; it exits with code 130 without a Python traceback.
Neither interrupted nor timed-out runs enter paired completed-run comparisons.

`--only quant_eager_seed42` selects a planned run. Run `train` again to skip
completed, checksum-verified runs and continue pending ones. Failed/interrupted
runs are preserved and block reuse of their names: use a new output directory for
a repeated comparison. Do not mix partial/resumed timings into a complete-run row.
The timeout is per run (default 32 hours), not a node reservation. Set the UCloud
job duration for preparation plus all runs. Set `CUDA_VISIBLE_DEVICES` to the
allocated devices if needed; the launcher does not overwrite it.

The launcher defaults to one CPU math thread and one compiler worker per rank.
Set `TORCH_HOME` and, if
needed, `TMPDIR` to sufficiently large writable volumes before launching. Compiler
caches are isolated per run/rank so first-use compilation is charged to that run;
filesystem page cache and hardware temperature are not reset between runs.
Compiler caches default to unique directories in `/tmp`, recorded per rank in the
run directory. Set `MT_COMPILER_CACHE_ROOT` to a larger writable directory if
needed; it must contain no whitespace because the C++ compiler toolchain can
misparse cache paths. Dataset and result paths may contain spaces. Even eager
model runs can compile existing augmentation kernels. Remove recorded cache
directories after the experiment if their disk space is needed.
Training output is redirected to the printed `console.log` path; use `tail -f`
from a second SSH session to follow it.

## Results and interpretation

`comparison.csv` contains wall time, total training-phase time, first and later
epoch times, maximum per-rank allocated CUDA memory and best validation selection
metric. `paired.json` gives wall speedup and validation differences against the
same-seed master control. Retain each seed; three seeds support an initial paired
comparison, not a precise general performance claim. Compare acceleration rows
against `quant_eager` as well to isolate optional-feature effects.

Each run retains `console.log`, `launch.json`, `result.json`, per-rank phase JSONL,
and `model/` with the trainer's configuration, summaries and `weights/best.pt` plus
training checkpoints. Wall time includes startup, loading, compilation, evaluation,
logging and saving; preparation is outside that measurement. Synchronized phase
timings include logger work. The harness disables the logger's per-step CUDA peak
reset so phase peaks cover all steps. These peaks exclude earlier model/optimizer
construction; use UCloud's resource report or `nvidia-smi` for whole-job/process
memory. CUDA allocator bytes are not total device memory or node RAM.

The built-in selection metric is the trainer's leaf-level validation accuracy;
DDP's padded validation sampler can repeat a few examples. It is not a full
macro/per-class quality study. Test data is frozen but never used by this training
runner. Choose configurations using validation first, then perform a separate,
non-distributed held-out evaluation for per-level accuracy, rare-class recall,
probability quality and uncertainty. Preserve failed and slower configurations.

## ONNX follow-up using these checkpoints

After training, use the **quant environment** to export a selected floating-point
checkpoint (including a master-trained checkpoint). This keeps the deployment
toolchain fixed while comparing learned weights:

```bash
/work/venvs/mt-quant/bin/python /work/comparison/export_followup.py /work/results/global-lepi-ddp quant_eager_seed42
```

This exports the actual hierarchical evaluation forward, runs dynamic-batch parity,
and compares CPU ONNX Runtime with reloaded FP32 PyTorch on four real validation
images through the checkpoint preprocessor. It retains the complete ONNX bundle,
source checkpoint hash, preprocessing description, image hashes, and
`validation-example.npz`. The destination must be new. It does not rerun training
or access test images.

A standalone environment with NumPy and ONNX Runtime can replay the saved inputs:

```python
import json
import numpy as np
import onnxruntime as ort

bundle = "/work/results/global-lepi-ddp/runs/quant_eager_seed42/onnx"
with open(f"{bundle}/manifest.json") as handle:
    manifest = json.load(handle)
session = ort.InferenceSession(f"{bundle}/model.onnx", providers=["CPUExecutionProvider"])
with np.load(f"{bundle}/validation-example.npz") as example:
    outputs = session.run(None, {manifest["input"]["name"]: example["images"]})
    for i, output in enumerate(outputs):
        np.testing.assert_allclose(output, example[f"output_{i}"], rtol=1e-4, atol=1e-5)
```

This establishes a small real-image export/inference parity check, not target-GPU
latency, dataset-wide quality or a standalone image decoder. Follow the
[ONNX guide](../../docs/onnx.md) and existing
[deployment benchmark workflow](../benchmarks/inference.md) for provider placement,
warmup/repeated timing, held-out inference and INT8 export. Native INT8 checkpoints
require their explicit CUDA-reference export path and are rejected by this FP32
follow-up. Calibrate PTQ on training images only.

## Local validation and limits

The bounded qualification increment has 16 focused passing cases covering the
bootstrap's success/failure flow with a fake installer, reproducible split sampling,
CPU preparation/training/reload with real EfficientNetV2-S, artifact checks,
interruption records, and deadline termination of a real worker. The core harness
tests also passed against the pinned master source. The bootstrap's real locked
export and dependency-sync dry run resolved the Python 3.12 CUDA 13.0 environment;
a complete fresh GPU environment was not installed locally.

Static checks passed. The repository-wide run completed its benchmark group, then
stalled in a spawned loader test in the sandbox; the remaining suite was rerun
outside the sandbox with one CPU math thread: 377 passed, 149 skipped, and the known
EMA expected failure. Skips include unavailable GPU/optional/slow coverage. Warnings
include upstream deprecations and missing optional dendrogram plotting dependencies.
There is no new B200/MIG runtime measurement for the bounded profile yet.

Initial harness validation, before the bounded profile:

The launch planner, input freezing, completed-run skipping, shell syntax, Ruff and
import contracts were checked. A small Parquet fixture ran through preparation,
one CPU training epoch and checkpoint reload with the real EfficientNetV2-S
hierarchical model on both the current quant code and pinned master source,
without downloading pretrained weights. The ONNX follow-up passed dynamic-batch
and four-image CPU Runtime parity using the resulting quant checkpoint.

Focused harness, training-state, integration and ONNX checks passed: 108 tests
across the validation runs, 43 hardware/optional-dependency skips and the known EMA
expected failure. CPU DDP passed when rerun outside the sandbox's socket
restriction. The repository-wide suite was interrupted in an unrelated long-running
benchmark; it was not completed. No GPU was available locally. These checks do not
establish CUDA/DDP compilation, INT8 correctness or global_lepi convergence and
throughput on UCloud; run the qualification configuration there first.

## Expanded single-GPU experiment and validation-loss diagnosis

`experiment.json` increases the qualification to 4,096 train / 1,024 validation /
128 test images and four epochs. Existing split assignments remain intact; the
sample builds its own taxonomy, so this still does not measure full-head quality.
It keeps batch 32, two loader workers, 384 pixels, seed 42 and the same pinned
packages. The three variants are master eager, quant eager and quant prefetch.
Prefetch remains diagnostic after an observed first-epoch validation NaN; finite
training losses do not establish that validation is numerically sound.

There are 128 training batches and 32 validation batches per epoch. Compare the
mean of epochs 2–4 (`later_epoch_mean_seconds`), and retain first-epoch and total
wall times separately. The expanded run measured about 31 seconds for the first
training epoch and 25.5 seconds for later epochs, but its first validation phase
(including figures) took 151 seconds and the run exceeded five minutes.
The template now sets `figures: false` for every variant, bypassing confusion,
class-distance and dendrogram figures (including species-name lookups). Scalar
validation metrics, losses, checkpoint selection and saving still run. Existing
configs default to figures enabled. This setting only changes the dedicated
harness worker; no package reinstall is required. Allow roughly 2–3 minutes per
variant based on those training times, with additional time possible for cold
image reads and startup. Whole-run and validation-phase timings are not directly
comparable to older runs with figures enabled.
Each worker has a 300-second limit; timeout terminates the experiment rather than
silently shortening its epochs. Process cleanup can take another 15 seconds.
The overall prepare/train budget is 1,800 seconds, including pauses between stages.
A fresh job is unnecessary; pull the harness update and use a new output directory.

`require_finite_losses: true` audits all expected training and validation epoch
rows in `model/logs/summary.csv`. Missing, malformed or non-finite total/per-level
losses produce `status=invalid_metrics` and `loss_check=failed`, even with exit code
zero. `result.json` identifies the affected epoch and phase. Checkpoints and timing
artifacts remain available, but such runs are excluded from successful paired
comparisons. This is a recorded-loss check, not a claim that the optimizer failed,
nor a guarantee that every intermediate tensor was finite. Existing configs retain
their previous behavior unless the flag is enabled.

First replay the affected checkpoint in four fresh processes. This uses the frozen
validation images, training class frequencies, smoothing and saved preprocessing.
FP16 means FP32 model parameters with FP16 autocast; FP32 disables autocast and
builds the criterion in FP32. No optimizer, augmentation or training runs. Each
case records parameter finiteness, per-batch input/output/loss finiteness and input
and label hashes under a new output directory. It never rewrites the old run.
Use the pinned quant interpreter, with the same thread settings as the launcher:

```bash
cd /work/mini_trainer
git pull --ff-only
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for precision in fp16 fp32; do
    for loader in eager prefetch; do
        replay_args=()
        if [[ "$loader" == prefetch ]]; then replay_args+=(--prefetch); fi
        timeout --kill-after=15s 300s /work/venvs/mt-quant/bin/python \
            dev/ucloud/replay_validation.py \
            /work/results/global-lepi-prefetch-1 quant_prefetch_seed42 \
            --precision "$precision" "${replay_args[@]}" \
            --output "/work/results/prefetch-replay-1-$precision-$loader" || break 2
    done
done
```

A completed diagnostic writes `result.json` even when it finds non-finite values;
inspect all four reports and `batches.jsonl`. Matching input/target hashes help
check that cases consumed identical batches. FP16-only failure points toward a
precision-dependent evaluation issue; prefetch-only failure warrants transfer-path
investigation. Neither outcome alone proves the root cause. Reloading clears
in-memory model caches, and per-batch checks synchronize CUDA, so a clean replay
cannot rule out a timing-sensitive failure in the original process. This is not a
speed benchmark. The original NaN occurred during evaluation, not recorded training
loss, and does not by itself demonstrate training divergence.

Then generate the expanded config, preserving this job's dataset and environment
paths. It intentionally creates a fresh preparation and starting weights:

```bash
python3 - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('dev/ucloud/experiment.json').read_text())
previous = json.loads(Path('/work/qualification-prefetch.json').read_text())
for key in ('parquet', 'environments'):
    config[key] = previous[key]
with Path('/work/qualification-expanded.json').open('x') as handle:
    json.dump(config, handle, indent=2)
    handle.write('\n')
PY
bash dev/ucloud/launch.sh /work/qualification-expanded.json --stage plan
bash dev/ucloud/launch.sh /work/qualification-expanded.json --stage prepare && \
    bash dev/ucloud/launch.sh /work/qualification-expanded.json --stage train
bash dev/ucloud/launch.sh /work/qualification-expanded.json --stage summary
cat /work/results/global-lepi-expanded-nofigures-1/comparison.csv
```

Run in tmux and retain all artifacts, including failed validation checks. A warm
first epoch is kept in the results, not discarded from model training; the later
three epochs provide the timing comparison. This is a larger exploratory test
with one paired seed, not a statistical quality comparison. Do not interpret
small accuracy or timing differences as established improvements. If the replay
finds non-finite values, keep prefetch results diagnostic while investigating them;
eager controls remain useful. Compiler variants are a subsequent experiment.

## Model compilation comparison in the existing job

Use `compilation.json` after the expanded eager/prefetch experiment. It keeps
4,096 training / 1,024 validation / 128 test images, four epochs, seed 42,
batch 32, two workers, 384 pixels, figures disabled and the same pinned venvs.
It replaces prefetch with `quant_compile_model` (model compilation only).
The deterministic run order is quant eager, master eager, then quant model
compilation, so both controls finish before a possible compilation timeout.
No dedicated NaN tracing or replay is required for this experiment. The existing
recorded-loss audit remains enabled; `invalid_metrics` retains timing evidence
but excludes that run from the automatic successful-pair report.

Pull into the current job and run in tmux. The template uses the dataset and venv
paths from the existing job and a fresh output directory; no installation is
needed. If those paths differ, copy the JSON and edit it before preparation.
Do not reuse or modify an already prepared output directory.

```bash
cd /work/mini_trainer
git pull --ff-only
bash dev/ucloud/launch.sh dev/ucloud/compilation.json --stage plan
bash dev/ucloud/launch.sh dev/ucloud/compilation.json --stage prepare && \
    bash dev/ucloud/launch.sh dev/ucloud/compilation.json --stage train
# Run summary even if training reports a timeout or invalid metrics.
bash dev/ucloud/launch.sh dev/ucloud/compilation.json --stage summary
cat /work/results/global-lepi-compile-model-1/comparison.csv
cat /work/results/global-lepi-compile-model-1/paired.json
```

Each worker has a 300-second limit, with up to 15 seconds for termination cleanup.
The 1,800-second overall budget includes preparation and gaps between commands.
Based on the existing eager runs, allow about five minutes for the two controls
combined, up to five minutes for compilation/training, plus preparation. This is
an estimate, not a promise that the compiled variant will finish.

The harness creates fresh compiler cache directories for each run. No separate
compiled warmup is performed: cold compilation cost belongs in `wall_seconds`
and the first epoch. Compare those with `later_epoch_mean_seconds` (epochs 2–4),
`train_seconds`, and `peak_allocated_bytes`. Later epochs may still include
recompilation; retain the console log when interpreting them. Compare primarily
against quant eager to isolate the effect of compilation, with master eager as
the branch control. A cold-start timeout establishes that this configuration
does not fit the short-job budget; it does not rule out longer-run benefits.
Small single-seed timing differences are exploratory, not established speedups.

## Optimizer compilation comparison in the existing job

`optimizer-compilation.json` tests optimizer compilation alone. It keeps the model
eager, prefetch disabled and optimizer CUDA graphs disabled. The workload remains
4,096 train / 1,024 validation / 128 test images, four epochs, seed 42, batch 32,
two loader workers and 384 pixels on one GPU. Figures are disabled and the
recorded-loss audit stays enabled. Installed master and quant revisions are unchanged.

The seeded order is quant eager, master eager, then `quant_compile_optimizer`.
Both controls therefore run before a possible compiler timeout. Each worker is
limited to 300 seconds (plus up to 15 seconds cleanup), and the overall budget is
1,800 seconds including preparation and pauses between stages. Allow roughly
10–12 minutes including preparation, based on the previous eager timings and
the compiled worker's limit; this does not guarantee compilation will finish.

Run these commands in tmux in the current allocated job. No venv rebuild is
needed. The new output directory preserves all earlier experiment artifacts.

```bash
cd /work/mini_trainer
git pull --ff-only
bash dev/ucloud/launch.sh dev/ucloud/optimizer-compilation.json --stage plan
bash dev/ucloud/launch.sh dev/ucloud/optimizer-compilation.json --stage prepare && \
    bash dev/ucloud/launch.sh dev/ucloud/optimizer-compilation.json --stage train
# Run summary even after a timeout or an invalid-metrics report.
bash dev/ucloud/launch.sh dev/ucloud/optimizer-compilation.json --stage summary
cat /work/results/global-lepi-compile-optimizer-1/comparison.csv
cat /work/results/global-lepi-compile-optimizer-1/paired.json
```

Monitor the compiled run from a second terminal:

```bash
tail -F /work/results/global-lepi-compile-optimizer-1/runs/quant_compile_optimizer_seed42/console.log
```

Compare the compiled optimizer primarily against quant eager within this run:
`wall_seconds`, `first_epoch_seconds`, `later_epoch_mean_seconds` and
`peak_allocated_bytes`. Compiler caches start fresh; no unmeasured compilation
warmup is added. Later epochs can still incur recompilation, so keep warnings
with the timing results. The existing optimizer, learning-rate schedule and AMP
settings remain in effect. This experiment does not combine model compilation
with optimizer compilation; it measures their effects separately first.

If master has `invalid_metrics`, `paired.json` will be empty because that report
requires a completed master baseline. The CSV still contains the direct quant
comparison. Preserve the loss-audit flag alongside exploratory timing conclusions;
no dedicated numerical tracing is a prerequisite for this experiment.

## Combined compilation, prefetch and native INT8

`combined-int8.json` compares the following runs in this seeded execution order:

| Variant | Installed branch | Native INT8 | Model compile | Optimizer compile | Prefetch |
| --- | --- | --- | --- | --- | --- |
| `quant_eager` | quant | No | No | No | No |
| `master_eager` | master | No | No | No | No |
| `quant_compile_model` | quant | No | Yes | No | No |
| `quant_compile_both` | quant | No | Yes | Yes | No |
| `quant_float_combined` | quant | No | Yes | Yes | Yes |
| `quant_int8_combined` | quant | Yes | Yes | Yes | Yes |

Both combined variants use the same pinned quant package. Here “float” means
ordinary FP32 parameters with FP16 AMP, not a switch to FP32-only training.
Native INT8 applies to eligible linear modules; convolutions, normalization,
biases and other unsupported operations remain floating point. Check the
`INT8 training coverage:` entry in the INT8 console log for the actual coverage.
This is not an entirely INT8 EfficientNet. Optimizer CUDA graphs remain disabled.
The older `quant_combined` name remains a floating-point variant for compatibility.

The four-epoch, 4,096/1,024/128-image workload, batch 32, two workers, 384 pixels,
seed 42, figures disabled and loss audit enabled are unchanged. Each worker has
a 300-second timeout (plus up to 15 seconds cleanup). The entire experiment has
a 1,800-second budget including preparation and gaps between stages. Six workers
all reaching their limits would exceed that budget, so the controller may stop
before the last run in that case. Based on previous timings, allow approximately
20–25 minutes; INT8 and combined compilation are unverified on this allocation.
Fresh compiler caches keep compilation cost in each run. No numerical tracing
or extra warmup is required.

Pull into the current job and run in tmux; existing venvs already include the
quantization extra. Preparation checks the dependency and single-GPU restriction
for any variant requesting INT8, including the new combined variant.

```bash
cd /work/mini_trainer
git pull --ff-only
bash dev/ucloud/launch.sh dev/ucloud/combined-int8.json --stage plan
bash dev/ucloud/launch.sh dev/ucloud/combined-int8.json --stage prepare && \
    bash dev/ucloud/launch.sh dev/ucloud/combined-int8.json --stage train
# Run even if training reports a failure, timeout or invalid metrics.
bash dev/ucloud/launch.sh dev/ucloud/combined-int8.json --stage summary
cat /work/results/global-lepi-combined-int8-1/comparison.csv
cat /work/results/global-lepi-combined-int8-1/paired.json
```

The controller continues after ordinary nonzero worker exits or invalid metrics,
but stops on a timeout. Preserve the logs. If the floating combined worker times
out and the overall budget has time left, launch just the not-yet-started INT8 run:

```bash
bash dev/ucloud/launch.sh dev/ucloud/combined-int8.json --stage train \
    --only quant_int8_combined_seed42
bash dev/ucloud/launch.sh dev/ucloud/combined-int8.json --stage summary
```

Do not rerun an existing failed directory or reset the experiment deadline.
Compare model-only against both compilations to test their interaction, then
both compilations against float combined to test prefetch, then float combined
against INT8 combined to test quantized training. Use total wall time, first-epoch
time, later-epoch mean and peak allocated memory, retaining loss-audit status.
The automatic paired report uses master; the CSV supports these direct quant
comparisons even when master is excluded by its loss audit. One seed does not
establish quality equivalence or a small performance improvement.

## Figures-enabled qualification

`figures.json` exercises the simplified dendrogram renderer, compact SVG export,
and bounded whole-matrix confusion reporting for two
epochs on the existing 4,096/1,024/128-image subset. Use only
`quant_compile_model_seed42`: floating-point training with model compilation,
without optimizer compilation or prefetch. The eager variants remain in the
configuration to satisfy harness validation; they are not selected below.
The worker retains its 300-second timeout and the experiment its 1,800-second
budget. Cold compilation and taxonomy lookups are included; completion within
five minutes needs confirmation on the allocated job.

This fix changes the installed package, so pull the harness and explicitly
upgrade the quant venv to the code revision pinned in this config. Existing
benchmark configs retain their older pins and will need a matching environment
if reused. Run in tmux:

```bash
cd /work/mini_trainer
git pull --ff-only
FIGURES_SHA=$(python3 -c 'import json; print(json.load(open("dev/ucloud/figures.json"))["environments"]["quant"]["commit"])')
uv pip install --python /work/venvs/mt-quant/bin/python --no-deps --link-mode=copy \
    "mini_trainer @ git+https://github.com/asgersvenning/mini_trainer.git@$FIGURES_SHA"
uv pip check --python /work/venvs/mt-quant/bin/python
bash dev/ucloud/launch.sh dev/ucloud/figures.json --stage plan
bash dev/ucloud/launch.sh dev/ucloud/figures.json --stage prepare && \
    bash dev/ucloud/launch.sh dev/ucloud/figures.json --stage train \
    --only quant_compile_model_seed42
# Summarize even if training reports a timeout or invalid metrics.
bash dev/ucloud/launch.sh dev/ucloud/figures.json --stage summary
cat /work/results/global-lepi-figures-3/comparison.csv
```

Inspect the console log's rendering/export timings and the saved figures under
`/work/results/global-lepi-figures-3/runs/quant_compile_model_seed42/model/logs/figures/`.
Both `epoch-0001` and `epoch-0002` should contain readable dendrogram SVGs,
captioned confusion overview PNGs and class-distance PNGs. The per-level
`Confusion_matrix_lvlL/` and `Soft_confusion_matrix_lvlL/` subdirectories retain
full-resolution PNGs, numerical data, class indices and color scales. Dashboard
overviews cover every class in the original order. See
[confusion reporting details and measurements](../benchmarks/reporting/confusion.md).
Compare first and second epoch reporting costs to check label-cache reuse. `paired.json` will be empty because no master run was selected; this is a
figure qualification, not a new branch comparison. Preserve any failed output
and choose a new output path for a retry.

See [renderer measurements and limitations](../benchmarks/reporting/dendrogram.md).

## Scaling the selected configuration

For four full GPUs with figures, W&B, a bounded batch sweep and checkpoint
continuation, use [the DDP qualification and production handoff](ddp.md).
Production training uses `mt_htrain` under `torchrun`; the Python API harness
remains a qualification tool.

For the post-training expert benchmark, start with the
[bounded RAM-staging inference trial](expert-trial.md). It generates its minimal
configuration and runs the standard prediction CLI without rebuilding an index.

## Calibrate filesystem read concurrency

`calibrate_io.py` is a standalone Python file; copying that one file is sufficient.
It uses standard-library threads and subprocesses, with Pillow needed only for
`--mode decode`. It does not import mini_trainer or alter its environment.

```bash
/work/venvs/mt-quant/bin/python dev/ucloud/calibrate_io.py \
  /work/flemming_helsing/restructured/valid/referenced \
  --mode stage --destination /dev/shm --output /work/expert-io-calibration.json
```

The default sweep tests 1–1,024 concurrent reads with eight-second trial limits,
then retests the three strongest candidates twice in shuffled order. It recommends
the smallest concurrency within 5% of the best median confirmation throughput.
The default overall budget is 180 seconds; blocked filesystem metadata calls may
outlast the budget. Each trial consumes only paths actually submitted to its reader pool; untouched
paths remain available after a timeout. Submitted selections are disjoint and
shuffled. Shared cache state remains unknown. Run away from other heavy read jobs
when selecting a baseline. Existing reports are preserved and a numeric suffix is
chosen automatically for subsequent runs.

`--mode read` (the default) measures concurrent encoded-byte reads without retaining
a second copy. `--mode stage --destination PATH` also measures writes to the intended
staging filesystem; its own temporary copies are removed between trials.
`--mode decode --resize 384` includes RGB decoding and optional CPU resize, for a
loader-oriented measurement. The resulting thread count is for concurrent I/O or
read/decode tasks, **not** a recommendation to create that many DataLoader processes.
This file calibrates concurrency; it does not install read-ahead into training.

`--workers`, `--files-per-trial`, `--trial-seconds`, `--budget-seconds`, and
`--confirmation-rounds` are adjustable. Each trial selects at most 2,048 files and
16 GiB of encoded data by default. Selection shrinks automatically to fit the byte
cap and staging destination; actual and requested concurrency are both recorded
when fewer files fit. Small trials (including ten-file trials) run normally, and
completed reads are ranked without an arbitrary 32-image minimum. Linux child RSS
is monitored against a 4 GiB stop threshold. Failed settings are excluded. The JSON report contains per-trial
throughput, latency, errors, sampled paths and the recommendation. If samples or
time are exhausted, inspect the confirmation coverage before treating the result
as repeatable. Very fast RAM trials benefit from increasing `--files-per-trial`.
