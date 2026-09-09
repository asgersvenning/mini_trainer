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
uv pip install --python /work/venvs/mt-master/bin/python --torch-backend=cu130 --require-hashes -r /work/requirements-mt.txt
uv pip install --python /work/venvs/mt-quant/bin/python --torch-backend=cu130 --require-hashes -r /work/requirements-mt.txt
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

Keep the
harness separately (it was added after the pinned quant commit), e.g. copy this
whole `dev/ucloud` directory to `/work/comparison`. Save the exported requirements.
The export and quantization extras are installed equally in both environments so
optional follow-ups do not change the dependency comparison.
Use the same backend in **both** the export's `--extra` and the installs'
`--torch-backend`; exported requirements do not carry the custom PyTorch index.

Preparation downloads torchvision's pretrained weights if absent. For offline
jobs, populate a shared writable `TORCH_HOME` ahead of time using the same locked
torchvision version. Allow CPU RAM/disk for the full hierarchical initial models,
three seeds, optimizer checkpoints and compiled caches. Do not assume INT8 makes
the convolutional backbone or all optimizer state integer.

## Configure and launch

Copy `comparison.json` and edit `parquet`, `output`, both Python paths and `gpus`.
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
It is CPU-heavy when parsing and initializing models. Start with a separate
qualification config/output: one seed, one epoch, and the two eager controls plus
the acceleration settings you intend to test. Use the same global batch and image
size as the full study. This is a real-data qualification run, not a substitute for
the full 18-run comparison. Eight workers per rank can mean 64 workers on eight
GPUs; reduce consistently if the allocated CPU or shared-memory budget requires it.

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
