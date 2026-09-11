# Four-GPU qualification and production handoff

Use one manually allocated UCloud node with **four full B200 GPUs**, the actual
Parquet and image storage, and internet access. `ddp.json` rejects devices below
140 GiB each, so this is not a continuation on the fractional-GPU job.

The chosen configuration is floating-point training on `quant`: FP16 AMP, model
compilation, figures, W&B, loss auditing and checkpoints. Optimizer compilation,
explicit CUDA prefetch, INT8 and EMA remain off. Qualification alone uses the
Python API. Production will use `mt_htrain` under `torchrun`.

This campaign qualifies four ranks. It does not establish eight-rank scaling or
full-dataset I/O throughput. If eight GPUs become available later, qualify that
allocation separately. At the same per-GPU batch, four GPUs halve the global batch;
review the production batch and learning-rate schedule explicitly before training.

## Setup and authentication

Clone the repository on branch `quant` into `/work/mini_trainer` (or pull it there).
Use the same checkout throughout the campaign; preparation hashes the harness.
With `uv`, `git`, Python and a C++ compiler available:

```bash
cd /work/mini_trainer
git pull --ff-only
MT_TEMPLATE=dev/ucloud/ddp.json MT_CONFIG=/work/ddp-b32.json \
  bash dev/ucloud/setup.sh \
  /work/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet

# Authenticate in the foreground before redirected workers start.
/work/venvs/mt-quant/bin/wandb login
# Optional: WANDB_ENTITY selects a team; otherwise use the authenticated default.
export WANDB_PROJECT=mini-trainer-ddp
nvidia-smi -L
```

Authentication uses the existing W&B SDK, without a new login mechanism or
credentials in Git. Each trial uses one shared W&B run with rank labels and a
unique ID. Only rank zero uploads figures and controls the shared finish state.
See [W&B distributed logging](https://docs.wandb.ai/models/track/log/distributed-training).
The installed quant environment contains the pinned package and locked dependencies.

## Baseline and batch sweep

Preparation samples 32,768 training, 4,096 validation and 128 test images from the
**existing splits**, but builds the head from the **full source taxonomy**. It
preserves class order across trials. The subset is half the eight-GPU proposal,
keeping 256 training steps and 32 validation steps per rank at batch 32.
This tests production head/figure dimensions
without training on all six million rows. Read `prepare.log` for actual class
counts. The test split is reserved for later evaluation, not batch selection.

There is one **30-minute wall-clock budget** shared by baseline preparation and
all derived trials, including gaps between commands. Individual training workers
have a 600-second guard; initial preparation has a 900-second guard. Neither is
an expected duration. If the campaign expires, retain its results and review
before starting another qualification campaign in the same allocated job.
Figures and W&B stay enabled. Qualification deadlines stop child processes, not
the UCloud allocation.

Define this helper in the foreground terminal or tmux session:

```bash
cd /work/mini_trainer
run_trial() {
  local cfg="$1" root rc=0
  root=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$cfg") || return
  bash dev/ucloud/launch.sh "$cfg" --stage plan &&
  bash dev/ucloud/launch.sh "$cfg" --stage prepare &&
  bash dev/ucloud/launch.sh "$cfg" --stage train || rc=$?
  if [[ -f "$root/prepared.json" ]]; then
    bash dev/ucloud/launch.sh "$cfg" --stage summary || rc=$?
    python3 dev/ucloud/scaling.py report "$root" || rc=$?
  fi
  return "$rc"
}
run_trial /work/ddp-b32.json
```

The baseline is three epochs at 32 images per GPU (global batch 128), with four
loader workers per rank. Four GPUs are launched with `torchrun`, no Slurm.
Check all four ranks completed, losses passed, figures look correct in W&B,
and reserved GPU memory has headroom before increasing the batch:

```bash
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-b64.json \
  --output /work/results/global-lepi-ddp-4gpu-b64-1 --batch 64
run_trial /work/ddp-b64.json

# Only after the 64/GPU result passes and has memory headroom:
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-b128.json \
  --output /work/results/global-lepi-ddp-4gpu-b128-1 --batch 128
run_trial /work/ddp-b128.json
```

Derived trials reuse verified preparation and starting weights, and inherit the
original deadline. They do not scan/sample/build the taxonomy again. Do not edit
frozen configs or reuse a failed output directory. Stop increasing at an OOM,
failed loss check, unstable memory, or lack of throughput improvement. Consider
256/GPU only if 128/GPU still has substantial headroom and time remains.

Select by **warm aggregate training images/second at an acceptable memory
footprint**, reserving roughly 20% GPU memory, rather than speed at fixed batch.
`scaling.py report` uses summed sample counts divided by the slowest rank's phase
time and excludes the first epoch, which includes cold compilation. It reports
peak reserved memory as a fraction of actual device capacity. W&B also records
`qualification/*` phase metrics; rank-level evidence remains in `phases-rank*.jsonl`.

Loader wait measures host time obtaining batches, including worker startup,
loading and decoding; it is not a pure storage measurement and can overlap GPU
work. Compare it alongside throughput. If it is high across warm epochs, try
one matched trial with more loader workers instead of a broad matrix. The
repeated subset benefits from filesystem caches: these timings cannot certify
full-dataset distributed-I/O throughput for a 24-hour run.

## Stability and checkpoint continuation

Reserve time for this gate instead of chasing every batch size. Set `BATCH` to
the selected **per-GPU** batch. Request a four-epoch scheduler horizon and preserve the generated horizon for
both the source and restored trial:

```bash
BATCH=64  # replace with the measured choice
WORKERS=4  # replace with the measured worker count
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-stability.json \
  --output /work/results/global-lepi-ddp-4gpu-stability-1 --batch "$BATCH" --workers "$WORKERS" --epochs 4
run_trial /work/ddp-stability.json
HORIZON=$(python3 -c 'import json; print(json.load(open("/work/ddp-stability.json"))["epochs"])')

python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-restore.json \
  --output /work/results/global-lepi-ddp-4gpu-restore-1 --batch "$BATCH" --workers "$WORKERS" --epochs "$HORIZON" \
  --checkpoint /work/results/global-lepi-ddp-4gpu-stability-1/runs/quant_compile_model_seed42/model/weights/checkpoint_1.pth \
  --resume-epoch 2
run_trial /work/ddp-restore.json
```

The restored run starts at epoch 2 (zero-based), trains the remaining epochs,
and uses a new output and W&B run. Each rank checks model, optimizer, scheduler
and scaler state against the source checkpoint **before any updates**. Require
four successful `restore-rank*.json` records, finite train/eval losses, stable
memory, and working figures/checkpoints after restoration. This tests state
restoration and continued execution, not bitwise reproduction of augmentation RNG.

## Keep the allocation and hand off immediately

Qualification and production run in the **same allocated UCloud job**. Do not
terminate/release the job, rebuild its environment, or wait for a new allocation
between them. The separate storage campaign below means a separate harness budget
and output directory, not another UCloud job. Do not enable the continuous-benchmark
provisioning/cleanup policy for this production allocation.

Before qualification, prepare the production CLI configuration and verify its paths,
full-dataset splits, pinned environment, W&B authentication, output location and
learning-rate/epoch recipe. Leave only measured batch and worker choices to finalize.
Keep the selected environment and caches warm. Reserve qualification time **in
addition to** the intended production duration in the UCloud job lifetime, plus
checkpoint/finalization margin; verify remaining time before launch. If the job
lifetime is insufficient, resolve its extension or a shorter reviewed training
budget while qualification runs rather than releasing the allocation.

After the required gates pass, record the selected batch/workers and qualification
report, finalize the pre-reviewed production configuration, and launch `mt_htrain`
under four-rank `torchrun` immediately in the existing terminal/tmux session. Do not
spend the allocation completing optional sweep points once the choice is supported.
The production configuration is still an explicit prerequisite, not generated or
validated by these qualification commands.

## Production uses the public CLIs

Do not launch the 24-hour run from this qualification harness. After qualification,
freeze a reviewed `production.yaml` for the full dataset, selected batch and
worker count, intended learning-rate schedule, complete taxonomy and preprocessing.
Record package/dependency pins, W&B run, split provenance and checkpoint hashes.
The final epoch count and learning-rate recipe need a separate production decision;
short qualification accuracy does not select them.

The planned command interface is:

```bash
/work/venvs/mt-quant/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 --no-python \
  /work/venvs/mt-quant/bin/mt_htrain --config /work/production.yaml --wandb --compile

/work/venvs/mt-quant/bin/mt_hpredict --config /work/evaluation.yaml
uvx --from "mini_metrics @ git+https://github.com/asgersvenning/mini_metrics.git@$METRICS_SHA" \
  mm_metrics --files "$PREDICTIONS_CSV" --output-dir /work/production-metrics --output

/work/venvs/mt-quant/bin/mt_export --weights "$BEST_PT" \
  --output /work/production-onnx --input-shape 3 384 384 \
  --preprocessing /work/deployment-preprocessing.json
```

These production/evaluation files and shell variables are **handoff placeholders**,
not runnable configs yet. `METRICS_SHA` must be a reviewed immutable commit.
Evaluation must use only the original held-out test split, with matching class
order and score semantics. Use the existing `mt_hpredict` CLI for this hierarchical
model; it also provides an explicit flat-head route. Consolidating inference into
one CLI is [deferred on the roadmap](../../docs/roadmap.md#5-mini_metrics-and-continuous-model-evaluation),
not a production prerequisite. Verify the deployment preprocessing and real-image
ONNX parity following [the export guide](../../docs/onnx.md).

A later full `master` training comparison remains optional and separate. This
qualification deliberately selects and stress-tests one configuration.

## Bounded resource-utilization decisions

Run only the next informative trial; do not execute this as an unattended matrix.
Keep the initial 30-minute campaign and reserve time for checkpoint continuation.
A minimum campaign is baseline, one larger batch, one worker comparison, then
restoration. The time guards can expire before all gates finish; retain evidence
and review any additional campaign rather than silently extending its budget.

Trial generation automatically raises the epoch count when needed to plan at least
100 training steps after the first epoch. At 256/GPU this subset requires five
rather than three epochs. `scaling.py report` exposes actual `warm_steps` and
`sufficient_warm_steps`; incomplete/failed runs are not performance-qualified merely
because their planned duration was sufficient. Changing the epoch horizon also
changes the learning-rate schedule, so these are throughput trials, not matched
convergence comparisons.

After selecting a batch, compare workers while holding batch, subset and epoch
horizon fixed. Start with 8/GPU; try 16/GPU only if loading/throughput evidence
justifies it. A small timing difference is not enough to select a winner.

```bash
BATCH=64  # replace with measured choice
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-w8.json \
  --output /work/results/global-lepi-ddp-4gpu-w8-1 --batch "$BATCH" --workers 8
run_trial /work/ddp-w8.json

# Optional, after reviewing w8:
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-w16.json \
  --output /work/results/global-lepi-ddp-4gpu-w16-1 --batch "$BATCH" --workers 16
run_trial /work/ddp-w16.json
```

Use `--workers "$WORKERS"` for both stability and restoration trials once workers
are selected. Read the generated stability config's actual `epochs` and pass that
same value to the restore trial; automatic warm-step sizing may have increased it.
A restore run keeps its explicit horizon and excludes its own first epoch from warm
performance reporting because it starts a fresh compiled process.

Reports retain per-rank phase records, training rank-time ratios, allocation peaks
and 32-batch timing windows (including a final partial window). Completed windows are also appended immediately
to `windows-rank*.jsonl` so a timeout retains them. Windows synchronize
CUDA at their boundaries, adding some overhead consistently across trials. They
measure rank-local elapsed time, not pure disk throughput. Figure and checkpoint
call durations are recorded separately in `components-rank*.jsonl`; figures are
also inside validation phase time, so do not add these overlapping totals. These
component measurements are host elapsed time and include backend logging costs.

### Separate broader storage pass

This is explicitly launched with its own budget. It requires fresh preparation,
selects 262,144 training / 8,192 validation / 128 test images, and excludes all image
identities in the baseline selection. The full taxonomy remains in use. The
baseline selection hash is checked, and `selection.json` records the new Parquet
hash and exclusion provenance; both are frozen in the preparation manifest.

```bash
BATCH=64    # selected batch
WORKERS=8   # selected worker count
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-storage.json \
  --output /work/results/global-lepi-ddp-4gpu-storage-1 \
  --batch "$BATCH" --workers "$WORKERS" --storage
run_trial /work/ddp-storage.json
```

The worker runs one epoch with a 600-second guard; the separate campaign allows
1,800 seconds including preparation (up to 900 seconds) and operator gaps. These
are limits, not runtime estimates or UCloud allocation extensions. A timeout may
leave partial window/phase evidence and must not be reported as success. With only
one epoch there is no warm-epoch result: inspect successive windows and first-epoch
cost separately. Reads use actual dataset storage without explicit RAM caching;
do not flush shared caches. Disjoint images reduce reuse of the baseline sample
but do not establish cold-cache conditions or certify full-dataset production I/O.

## Generate the production CLI configuration

`production.py` verifies the starting weights and full-taxonomy preparation artifacts
and writes a new YAML file for the real `mt_htrain` CLI. It deliberately omits the
qualification `data_index`: the CLI parses the complete source Parquet and preserves
its supplied splits. The starting weights are the original pretrained initialization,
not a model trained on the qualification subset. Full metadata parsing/broadcast and
full validation have larger CPU/memory costs than qualification.

Generate this only after the epoch horizon is reviewed from the broader storage
measurement and remaining allocation time. The generator requires an explicit epoch
count; it does not extrapolate the cached subset into a production schedule. It
retains the qualified LR 0.001, weight decay 0.01 and 0.25-epoch warmup; larger global
batches are not automatically assigned a linearly scaled learning rate.

```bash
# Set these to the final measured/reviewed choices before running this block.
: "${BATCH:?selected per-GPU batch}"
: "${WORKERS:?selected per-GPU workers}"
: "${EPOCHS:?reviewed full-data epoch horizon}"
PRODUCTION_OUTPUT=/work/results  # user-confirmed persistent storage for this job

/work/venvs/mt-quant/bin/python dev/ucloud/production.py \
  /work/results/global-lepi-ddp-4gpu-b32-2 /work/production.yaml \
  --output "$PRODUCTION_OUTPUT" --name global-lepi-production-1 \
  --batch "$BATCH" --workers "$WORKERS" --epochs "$EPOCHS"
```

Before launch, verify the generated YAML, available storage and the installed package
pin against the successful qualification. Keep the same four GPUs and environment.
The W&B project and shared run identity are set explicitly in YAML; `--wandb` is still
required to enable the backend. Also pass `--compile` explicitly: the pinned CLI
parser otherwise overrides YAML compilation with its false flag default.
No UCloud credentials are involved.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export MPLBACKEND=Agg PYTHONHASHSEED=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_HOME=/work/.cache/torch
/work/venvs/mt-quant/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 --no-python -- \
  /work/venvs/mt-quant/bin/mt_htrain --config /work/production.yaml --wandb --compile
```

Run in tmux and retain stdout/stderr. The pinned CLI writes `checkpoint_last.pth`
after every completed epoch, `best.pt` after improvements, and numbered checkpoints
every five epochs (zero-based 0, 5, 10, ...). These writes are not guaranteed atomic:
if expiry interrupts a write, retain a previous intact numbered checkpoint. The
qualification harness forces more frequent numbered checkpoints; that override does
not carry over to the CLI. Do not assume a final `last.pt` exists after interruption.
The best score on resume is not persisted by the current training loop; archive prior
best artifacts before any separately reviewed resume into an existing output.

The CLI does not inherit qualification-only finite-loss auditing or timing wrappers.
Inspect the epoch summary and checkpoints during production; the full-dataset run
has no harness wall-time guard and stops at its configured epoch horizon or when the
job/process is terminated. Evaluation/export are separate post-training activities.
