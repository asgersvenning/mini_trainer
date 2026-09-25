# Four-GPU qualification and production handoff

This procedure uses the September campaign's pinned packages and four full B200s.
For measured results and next-run lessons see the
[training post-mortem](../../docs/training-workflow-postmortem.md).
Select new package pins explicitly for a new campaign; qualification on this
profile does not establish other GPU topologies or full-dataset storage throughput.

## Setup

Follow [UCloud setup](README.md#fresh-job-setup) to install uv and clone the reviewed
harness into the manually allocated node. Keep the checkout unchanged after
preparation hashes it. The node needs mounted images/Parquet, internet or prepared
caches, a C++ compiler, and four GPUs with at least 140 GiB each (`ddp.json` rejects
smaller devices). Launch one controller, not a second multi-task wrapper.

```bash
cd /work/mini_trainer
MT_TEMPLATE=dev/ucloud/ddp.json MT_CONFIG=/work/ddp-b32.json \
  bash dev/ucloud/setup.sh \
  /work/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet

# Authenticate before redirected workers start; WANDB_ENTITY optionally selects a team.
/work/venvs/mt-quant/bin/wandb login
export WANDB_PROJECT=mini-trainer-ddp
nvidia-smi -L
```

The profile enables FP16 AMP, model compilation, figures, W&B, finite-loss auditing
and checkpoints. Optimizer compilation, explicit CUDA prefetch, INT8 and EMA are
off. Each trial has one shared W&B run with rank labels; rank zero uploads figures
and finishes the run. Qualification uses the Python API; production uses `mt_htrain`.

## Baseline and resource selection

The [profile](ddp.json) samples 32,768 train / 4,096 validation / 128 test records
from their **existing splits**, with the **full source taxonomy** and fixed class
order. Test rows are reserved for later evaluation. This exercises full-head
training and diagnostics without scanning every image; short-run accuracy is not
a production selection criterion.

| Control | Baseline / rule |
| --- | --- |
| Batch | 32 per GPU, 128 global; `scaling.py --batch` always means per GPU |
| Loader workers | 4 per rank; select using allocated CPU/RAM and observed loading latency |
| Epochs | 3 initially; derived trials plan at least 100 training steps after the first epoch |
| Shared deadline | 1,800 seconds from preparation, including operator gaps and derived trials |
| Child limits | Preparation 900 seconds; training 600 seconds; termination may add cleanup time |

These are limits, not ETAs or allocation extensions. Reserve time for restoration
and production; run only trials that change a decision. Derived trials reuse
verified preparation/starting weights and inherit the deadline. They need new
config/output paths; retain failed attempts instead of overwriting evidence.

In the foreground terminal or tmux session, define:

```bash
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

Require four completed ranks, finite losses, correct W&B figures and memory
headroom. Then try a larger batch, using a fresh destination for each trial:

```bash
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-b64.json \
  --output /work/results/global-lepi-ddp-4gpu-b64-1 --batch 64
run_trial /work/ddp-b64.json
```

Continue to 128 or 256 per GPU when memory and time justify it. Select warm
aggregate images/s at acceptable memory use (roughly 20% headroom); stop at OOM,
unstable losses/memory or no meaningful throughput gain. Global batch changes the
training recipe; learning rate is not automatically scaled.

For a worker comparison, hold batch, sample and generated epoch horizon fixed.
Use a meaningful increase for the available resources: the production campaign
ultimately used 32 workers/rank, while storage probes favored much higher encoded
read concurrency. A DataLoader process and an outstanding file read are different
resources. A small cached subset cannot select production storage concurrency.

```bash
BATCH=64   # selected per-GPU batch
WORKERS=8  # candidate per-GPU worker count
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-workers.json \
  --output /work/results/global-lepi-ddp-workers-1 --batch "$BATCH" --workers "$WORKERS"
run_trial /work/ddp-workers.json
```

### Interpret measurements

- `scaling.py report` divides summed samples by the slowest rank's phase time,
  excludes the first epoch (cold compilation), and reports warm-step sufficiency,
  rank imbalance and peak reserved memory relative to device capacity. Failed or
  incomplete runs are not qualified performance results.
- Trial generation can increase the epoch horizon: 256/GPU needs five epochs for
  the warm-step target. This also changes the LR schedule; throughput trials are
  not matched convergence comparisons. Restored trials keep their explicit horizon
  and exclude their own first epoch from warm reporting.
- Loader wait includes worker startup, reading and decoding and can overlap GPU
  work. `phases-rank*.jsonl` and `windows-rank*.jsonl` retain phase and 32-batch
  windows, including partial final windows. Window-boundary CUDA synchronization
  adds overhead; these are not pure storage timings.
- `components-rank*.jsonl` records host time for figures/checkpoints, including
  logging costs. Figure time also appears inside validation time; do not add
  overlapping measurements. Cached repeat epochs do not establish cold-file or
  full-dataset throughput.

## Stability and checkpoint continuation

Use the selected `BATCH` and `WORKERS`. Preserve the **generated** scheduler horizon
for both source and restored trials; it may exceed the requested four epochs:

```bash
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

Restoration starts at zero-based epoch 2 with a new output/W&B run. Require four
successful `restore-rank*.json` records verifying model, optimizer, scheduler and
scaler **before updates**, then finite losses, stable memory and working figures
and checkpoints. This checks restoration and continued execution, not bitwise
reproduction of stochastic augmentation. Do not silently extend an expired
campaign budget to finish optional trials.

## Separate broader storage pass

This optional pass has its own 1,800-second budget and fresh preparation. It selects
262,144 train / 8,192 validation / 128 test images, excluding identities from the
baseline sample while retaining full taxonomy. The exclusion hash and selection
provenance are frozen. It runs one epoch, with the same 900-second preparation and
600-second worker guards:

```bash
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-storage.json \
  --output /work/results/global-lepi-ddp-4gpu-storage-1 \
  --batch "$BATCH" --workers "$WORKERS" --storage
run_trial /work/ddp-storage.json
```

There is no warm-epoch result for this pass: inspect successive windows and startup
separately. A timeout leaves partial evidence, not a successful run. Disjoint paths
reduce sample reuse but do not guarantee cold WEKA caches; never flush shared caches.

## Generate and launch production

Keep the qualified environment and allocation. Allow time for full metadata parsing,
validation, diagnostics, saving and finalization. Choose the full-data epoch/LR
schedule from storage evidence and remaining allocation time, not cached-subset
speed. Resolve any allocation extension before launch.

`production.py` verifies preparation hashes and writes a new YAML for the public
CLI. It uses the **original pretrained initialization**, not subset-trained weights,
and omits the qualification `data_index` so the CLI parses the full Parquet with
its supplied splits/taxonomy. It retains LR 0.001, weight decay 0.01 and 0.25-epoch
warmup; no batch-dependent LR scaling is applied.

```bash
: "${BATCH:?selected per-GPU batch}"
: "${WORKERS:?selected per-GPU workers}"
: "${EPOCHS:?reviewed full-data epoch horizon}"
PRODUCTION_OUTPUT=/work/results

/work/venvs/mt-quant/bin/python dev/ucloud/production.py \
  /work/results/global-lepi-ddp-4gpu-b32-1 /work/production.yaml \
  --output "$PRODUCTION_OUTPUT" --name global-lepi-production-1 \
  --batch "$BATCH" --workers "$WORKERS" --epochs "$EPOCHS"
```

Review the YAML, persistent storage and installed pin against qualification. Keep
the same four GPUs; the generator rejects other topologies. Pass `--wandb` to enable
the configured backend and `--compile` because this pinned parser's false flag
default otherwise overrides YAML. Run in tmux and retain stdout/stderr:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export MPLBACKEND=Agg PYTHONHASHSEED=0 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_HOME=/work/.cache/torch
/work/venvs/mt-quant/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 --no-python -- \
  /work/venvs/mt-quant/bin/mt_htrain --config /work/production.yaml --wandb --compile
```

The CLI does not inherit qualification-only finite-loss auditing, timing wrappers
or wall-time guards. Inspect epoch summaries/checkpoints during production. The
pinned CLI writes `checkpoint_last.pth` each completed epoch, `best.pt` after
improvements, and numbered checkpoints every five epochs (zero-based 0, 5, 10, …).
The harness's more frequent checkpoint override does not carry over. Writes are
not guaranteed atomic: retain an intact earlier checkpoint if expiry interrupts a
write, and do not assume `last.pt` exists. Best score is not persisted on resume;
archive previous best artifacts before resuming into an existing output.

Held-out prediction uses `mt_hpredict`; pass its CSVs to the pinned
[mini_metrics environment](../releases/mambo_v3/evaluation.md#metrics). Follow the
[ONNX guide](../../docs/onnx.md) for export, preserving class order, preprocessing
and score semantics. Evaluation/export remain separate from training.
