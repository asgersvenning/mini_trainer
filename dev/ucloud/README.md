# UCloud training comparison

Run inside one manually allocated UCloud node, with mounted data and results under
`/work`. Connect by SSH and use `tmux` for long commands; it does not extend the
allocation. These helpers do not submit jobs or install a scheduler.

This harness retains historical master/quant comparisons. Quantization is merged;
the labels identify **pinned packages**, not current branch tips. The completed
production recipe and lessons are in the
[training post-mortem](../../docs/training-workflow-postmortem.md). For a new
production run use the [DDP/production guide](ddp.md); for MAMBO deployment testing
use the [release runbook](../releases/mambo_v3/ucloud-release.md).

## Fresh job setup

Install uv, clone the repository, select the reviewed harness revision, then let
the setup helper create the dedicated environments. Keep the harness checkout
separate from the pinned packages it installs.

```bash
# Install uv only if the node does not already provide it.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/asgersvenning/mini_trainer.git /work/mini_trainer
cd /work/mini_trainer
git checkout YOUR_REVIEWED_REF
bash dev/ucloud/setup.sh /work/YOUR_DATASET/YOUR_METADATA.parquet
source /work/venvs/mt-quant/bin/activate
export TORCH_HOME=/work/.cache/torch
```

Replace the revision and metadata placeholders. The node needs a working NVIDIA
driver and C++ compiler; preflight checks CUDA execution and compiler availability.
Internet or prepared caches are needed for packages and pretrained weights.
Mounted paths/allocation products are described in the
[UCloud guide](https://docs.cloud.sdu.dk/guide/submitting.html).

`setup.sh` reads the template's immutable package commits, exports the quant pin's
lock, installs matching dependencies into dedicated Python 3.12 environments, and
writes `/work/qualification.json`. It does not train or modify the dataset. It
requires full Git history containing those pins and does not install editable
packages. Never reset the harness checkout to an older package pin.

| Setup override | Purpose |
| --- | --- |
| `MT_TEMPLATE` | Profile JSON; defaults to the adjacent `qualification.json` |
| `MT_CONFIG` | New generated node configuration; existing files are not overwritten |
| `MT_WORK_ROOT` | Persistent environment/result root; default `/work` |
| `MT_TORCH_BACKEND` | `cu126`, `cu130` (default), or `cu132`; choose for the node |
| `MT_REPO_URL` | Package Git source, including a transferred `file:///work/...` clone |
| `UV_CACHE_DIR` | Package cache; default `/work/.cache/uv` |

Setup can retry partial dependency installation in its dedicated environments.
It uses required hashes, explicit PyPI/CUDA indexes and copy link mode. Do not add
`--torch-backend` to the exported requirements install: that can redirect the
PyPI-locked TorchAO dependency. The helper is the maintained installation recipe;
inspect `requirements-mt.txt` and installed inventories rather than duplicating
its commands. For offline source transfer, create a Git bundle containing the
template's pinned commits and set `MT_REPO_URL` to the resulting node-side clone.

## Configure and launch

From the repository root:

```bash
bash dev/ucloud/launch.sh /work/qualification.json --stage plan
bash dev/ucloud/launch.sh /work/qualification.json --stage prepare &&
bash dev/ucloud/launch.sh /work/qualification.json --stage train
# Keep failure evidence: summary can run even after an unsuccessful train stage.
bash dev/ucloud/launch.sh /work/qualification.json --stage summary
```

Review `parquet`, new `output`, environment paths/commits, GPU count, batch and
worker settings in the resolved JSON. The launcher runs installed packages outside
the checkout. Use one launcher per node: multiple visible GPUs use one torchrun
process per GPU; a single device uses ordinary training. Do not wrap the launcher
in another multi-task launch. Native INT8 variants reject DDP.

Metadata must be beside `images/<speciesKey>/<filename>`. Preparation preserves
existing `set` values: 0=test, 1=validation, other numeric sets=train; the current
parser excludes nonnumeric sets. It freezes taxonomy/index and checks selected
paths, duplicates and labels. It does not hash/decode all image bytes; keep inputs
immutable. Test records are not used for training or validation.

**The shipped qualification is an infrastructure smoke.** It uses one visible
CUDA device (including one MIG slice), batch 32, two loader workers, one epoch and
seed 42, with both eager controls. Reservoir sampling preserves splits and selects
2,048 train / 512 validation / 128 test records. Only these paths are inspected;
taxonomy is sample-derived. Missing training species in validation and the smaller
head make its quality/throughput unrepresentative of full-data training. Reports
identify `scope=qualification_subset`.

Preparation and both runs share a 1,800-second deadline from `prepare`; pauses
between commands count. Per-worker limits and up to 15 seconds termination cleanup
are separate. Installation and the UCloud reservation lifetime are outside that
budget. One full-data epoch is not a substitute for this small qualification.

## Profiles and comparison controls

Choose a profile for a decision, copy it to a new node config, and review its
pins/paths before preparation. Use `MT_TEMPLATE` when its packages differ from the
installed environments. Do not treat the following as a mandatory experiment sequence.

| Profile | Scope |
| --- | --- |
| [qualification.json](qualification.json) | Small eager-pair environment and lifecycle smoke |
| [comparison.json](comparison.json) | Full-data, three-seed eager/prefetch/compilation comparison |
| [experiment.json](experiment.json) | Four-epoch 4,096/1,024/128 subset; eager and prefetch |
| [compilation.json](compilation.json) | Same subset, model compilation |
| [optimizer-compilation.json](optimizer-compilation.json) | Same subset, optimizer compilation alone |
| [combined-int8.json](combined-int8.json) | Staged model compile → optimizer compile → prefetch → native INT8, one GPU |
| [figures.json](figures.json) | Two epochs with required diagnostics; select `quant_compile_model_seed42` |
| [ddp.json](ddp.json) | Separate full-head DDP and production handoff |

The comparison keeps EfficientNetV2-S at 384 pixels, normalized hierarchical
species/genus/family heads, equal loss weights, hidden layer/dropout, MuonAuxAdamW,
class weighting, regularization and smoothing fixed. The pinned master already has
those features; they are not quantization innovations. Master creates one initial
pretrained-backbone/random-head checkpoint per seed for both packages. Starting
tensors match; implementation RNG trajectories need not.

`global_batch_size` is **global**: 64 becomes 32/16/8 per rank on 2/4/8 GPUs.
Learning rate is not automatically scaled. GPU count changes sampling/SyncBatchNorm
and kernels, so compare paired variants within an allocation. Run order is
reproducibly shuffled per seed; no checkpoint averaging or resumed timing is used.

The main controls are `master_eager` and `quant_eager`. Optional variants add
`quant_prefetch`, `quant_compile_model`, `quant_compile_optimizer`, `quant_combined`,
model/optimizer graphs, or single-GPU `quant_int8`. In the combined profile,
`quant_compile_both` adds optimizer compilation, `quant_float_combined` adds prefetch,
and `quant_int8_combined` adds native INT8. “Float” retains FP16 AMP over ordinary
floating parameters. INT8 covers eligible Linear modules, not the whole backbone;
inspect its coverage report. No automatic fallback hides failed variants.

Templates disable EMA and full-dataset RAM caching. Diagnostic subset profiles may
disable figures; that makes their timings incomparable with figures-enabled runs.
Preserve full diagnostics and W&B when qualifying production. Feature/quality
ablations belong in the [separate protocol](../../docs/training-feature-validation.md).

## Resources, progress and recovery

Use the allocation's CPU/RAM limits, not host-wide `nproc`/`free` totals. Inspect
`nvidia-smi` for actual GPU/MIG topology. Read concurrency, DataLoader processes,
math threads and compiler workers consume different resources. The launcher defaults
to one CPU math thread and compiler worker per rank; explicit OMP/MKL settings
affect both branches. Size decode workers to the allocation; latency-bound shared
storage can warrant far more outstanding reads, as the production post-mortem
demonstrates. Avoid replicating a full decoded dataset per DDP rank.

Keep caches on sufficiently large writable storage. Compiler caches are isolated
per run/rank and their paths are recorded; `MT_COMPILER_CACHE_ROOT` must contain
**no whitespace** because of compiler tooling. Dataset/result paths may contain
spaces. Cold compilation counts toward each run; shared filesystem caches and
device temperature are not reset.

The launcher prints `prepare.log` and run `console.log` locations. Preparation logs
sampling, taxonomy, checking/hashing and model construction; CPU initialization may
still be silent. `prepared.json` is the completion marker. If absent, inspect live
processes/logs before restarting. Preparation requires a fresh output directory;
preserve incomplete attempts rather than deleting evidence.

Run `train` again to skip completed, checksum-verified runs and continue pending
ones. `--only quant_eager_seed42` selects one planned run. Failed/interrupted names
cannot be reused: use a new comparison output. Ctrl-C terminates the process group
and records interruption; timeout stops the controller. Do not reset deadlines or
mix partial/resumed timing into complete-run rows. Summary remains available.

With `require_finite_losses: true`, missing/malformed/nonfinite epoch loss rows
produce `invalid_metrics` even after exit zero. Logs/checkpoints/timings remain,
but automatic successful-pair reports exclude the run. A saved-loss audit does not
prove every intermediate tensor was finite. A missing valid master control can
leave `paired.json` empty while direct quant comparisons remain in the CSV.

## Results and interpretation

Retain the config/prepared inputs plus `comparison.csv`, `paired.json` and each
run's launch/result/console logs, per-rank phase JSONL and trainer `model/` outputs.
Comparison rows expose wall, training, first/later-epoch time, allocated memory and
validation selection metric. Master pairing is per seed; compare acceleration
against `quant_eager` too. Few seeds and short runs do not establish convergence.

Wall time includes startup, IO, compilation, evaluation, logging and saving but
excludes preparation. Phase peaks exclude earlier model/optimizer construction;
allocator memory is not whole-process/device/RAM usage. Retain cold setup and
later epochs separately, including recompilation warnings and slower/failed rows.
The selection metric is leaf validation accuracy; padded DDP validation can repeat
samples. Full macro/per-class/parent quality needs separate held-out evaluation.

## Targeted follow-ups

**Validation-loss replay:** `replay_validation.py` evaluates a saved run without
training, using its frozen validation inputs, frequencies and preprocessing.
Compare FP16/FP32 and ordinary/prefetched loading only when that changes a diagnosis:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /work/venvs/mt-quant/bin/python \
  dev/ucloud/replay_validation.py /work/results/YOUR_COMPARISON quant_prefetch_seed42 \
  --precision fp32 --output /work/results/YOUR_NEW_REPLAY
```

Inspect `result.json` and `batches.jsonl` input/label hashes and finiteness.
Reloaded caches and synchronized diagnostics mean a clean replay cannot rule out
the original timing-sensitive failure. This is not a throughput benchmark.

**Floating ONNX:** use the quant environment to hold the export toolchain fixed,
even when exporting master-trained weights:

```bash
/work/venvs/mt-quant/bin/python dev/ucloud/export_followup.py \
  /work/results/YOUR_COMPARISON quant_eager_seed42
```

The new output contains source/input hashes, preprocessing, a complete graph plus
external data, dynamic-batch checks and `validation-example.npz` from four real
validation images. This establishes bounded FP32 CPU parity, not dataset quality
or target-GPU performance. Native INT8 checkpoints require the separate explicit
CUDA-reference export path. See [ONNX](../../docs/onnx.md) and
[inference benchmarks](../benchmarks/inference.md); calibrate PTQ on training only.

**Production inference:** use the normal prediction CLI after
[DDP qualification](ddp.md). The completed September campaign's pinned-overlay
staging helpers are retired; their
[historical source](https://github.com/asgersvenning/mini_trainer/tree/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/dev/ucloud)
and [lessons](../../docs/training-workflow-postmortem.md) remain available.
For current MAMBO comparisons use [the release runbook](../releases/mambo_v3/ucloud-release.md).

## Calibrate filesystem read concurrency

`calibrate_io.py` is a standalone standard-library helper (Pillow only for decode).
It measures concurrency; it does not install read-ahead into the trainer.

```bash
python dev/ucloud/calibrate_io.py /work/YOUR_IMAGES \
  --mode stage --destination /dev/shm --output /work/io-calibration.json
python dev/ucloud/calibrate_io.py --summary /work/io-calibration.json
```

Modes: `read` discards encoded reads, `stage` also measures destination writes and
removes its temporary copies, `decode --resize 384` includes RGB preprocessing.
Defaults sweep 1–1,024 readers, then confirm the three strongest settings twice;
recommendation is the smallest within 5% of the best median. This is an IO-reader
count, **not a DataLoader process count**.

Startup and IO each have a 30-second limit; total budget is 600 seconds. Trials
default to 2,048 files / 16 GiB encoded data and a 4 GiB child-RSS stop limit.
Byte/destination caps shrink selection; reports retain requested/actual concurrency.
Submitted paths are disjoint and shuffled, but shared-cache state is unknown.
Use `--help` for overrides; increase sample size when RAM trials are too short.

Inspect JSON, adjacent `.summary.txt` and failure `.error.log`. Completed partial
trials can inform a provisional sweep result; confirmation errors/memory failures
exclude a setting. Uninterruptible kernel IO may delay termination beyond budgets;
the calibrator stops instead of accumulating competing readers. Existing reports
receive numeric suffixes. Run away from competing heavy scans when selecting a
baseline and report both first-pass and repeated-read conditions.

## Validation boundary

Offline tests cover planning, frozen inputs/splits, real CPU prepare/train/reload,
loss audits, worker termination and staged IO. See `tests/benchmarks/test_ucloud_*`
and `test_io_calibration.py`. They do not qualify CUDA/DDP compilation, INT8 quality
or full-data throughput. Historical per-edit test counts/ETAs are not current
status; retained campaign evidence is summarized in the linked post-mortem.
