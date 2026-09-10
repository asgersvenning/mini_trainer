# Eight-GPU qualification and production handoff

Use one manually allocated UCloud node with **eight full B200 GPUs**, the actual
Parquet and image storage, and internet access. `ddp.json` rejects devices below
140 GiB each, so this is not a continuation on the fractional-GPU job.

The chosen configuration is floating-point training on `quant`: FP16 AMP, model
compilation, figures, W&B, loss auditing and checkpoints. Optimizer compilation,
explicit CUDA prefetch, INT8 and EMA remain off. Qualification alone uses the
Python API. Production will use `mt_htrain` under `torchrun`.

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

Preparation samples 65,536 training, 8,192 validation and 128 test images from the
**existing splits**, but builds the head from the **full source taxonomy**. It
preserves class order across trials. This tests production head/figure dimensions
without training on all six million rows. Read `prepare.log` for actual class
counts. The test split is reserved for later evaluation, not batch selection.

There is one **30-minute wall-clock budget** shared by baseline preparation and
all derived trials, including gaps between commands. Individual training workers
have a 600-second guard; initial preparation has a 900-second guard. Neither is
an expected duration. If the campaign expires, retain its results and review
before allocating another campaign. Figures and W&B stay enabled.

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

The baseline is three epochs at 32 images per GPU (global batch 256), with four
loader workers per rank. Eight GPUs are launched with `torchrun`, no Slurm.
Check all eight ranks completed, losses passed, figures look correct in W&B,
and reserved GPU memory has headroom before increasing the batch:

```bash
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-b64.json \
  --output /work/results/global-lepi-ddp-b64-1 --batch 64
run_trial /work/ddp-b64.json

# Only after the 64/GPU result passes and has memory headroom:
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-b128.json \
  --output /work/results/global-lepi-ddp-b128-1 --batch 128
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
the selected **per-GPU** batch. Keep the scheduler horizon at four epochs for
both the source and restored trial:

```bash
BATCH=64  # replace with the measured choice
python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-stability.json \
  --output /work/results/global-lepi-ddp-stability-1 --batch "$BATCH" --epochs 4
run_trial /work/ddp-stability.json

python3 dev/ucloud/scaling.py trial /work/ddp-b32.json /work/ddp-restore.json \
  --output /work/results/global-lepi-ddp-restore-1 --batch "$BATCH" --epochs 4 \
  --checkpoint /work/results/global-lepi-ddp-stability-1/runs/quant_compile_model_seed42/model/weights/checkpoint_1.pth \
  --resume-epoch 2
run_trial /work/ddp-restore.json
```

The restored run starts at epoch 2 (zero-based), trains the remaining two epochs,
and uses a new output and W&B run. Each rank checks model, optimizer, scheduler
and scaler state against the source checkpoint **before any updates**. Require
eight successful `restore-rank*.json` records, finite train/eval losses, stable
memory, and working figures/checkpoints after restoration. This tests state
restoration and continued execution, not bitwise reproduction of augmentation RNG.

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
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 --no-python \
  /work/venvs/mt-quant/bin/mt_htrain --config /work/production.yaml --wandb

/work/venvs/mt-quant/bin/mt_predict --config /work/evaluation.yaml
uvx --from "mini_metrics @ git+https://github.com/asgersvenning/mini_metrics.git@$METRICS_SHA" \
  mm_metrics --files "$PREDICTIONS_CSV" --output-dir /work/production-metrics --output

/work/venvs/mt-quant/bin/mt_export --weights "$BEST_PT" \
  --output /work/production-onnx --input-shape 3 384 384 \
  --preprocessing /work/deployment-preprocessing.json
```

These production/evaluation files and shell variables are **handoff placeholders**,
not runnable configs yet. `METRICS_SHA` must be a reviewed immutable commit.
Evaluation must use only the original held-out test split, with matching class
order and score semantics. Before finalizing `evaluation.yaml`, qualify the
requested `mt_predict` entry point on a hierarchical checkpoint: currently
`mt_hpredict` supplies the hierarchical builder/collector while `mt_predict`
selects their generic counterparts. Resolve that small CLI compatibility gap
before production; do not silently flatten hierarchy labels or substitute an
API-only evaluator. Likewise verify the deployment preprocessing and real-image
ONNX parity following [the export guide](../../docs/onnx.md).

A later full `master` training comparison remains optional and separate. This
qualification deliberately selects and stress-tests one configuration.
