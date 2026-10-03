# Flat training ablations on UCloud

Research protocol and executable harness, not measured evidence of a feature benefit.
The study uses production trainer APIs through a research builder; it changes no
package defaults. It follows the [publication workflow](../README.md) conventions:
frozen inputs, paired seeds, validation-only choices and retained individual results.

## Questions and budget

EfficientNetV2-S, ImageNet1K V1 initialization, 384 pixels; all backbone parameters
train after one epoch of head-only learning-rate warmup. `fine_tune=True` is **not**
used: that API freezes the backbone. Start with a **2×2×2 factorial** over the
normalization package, prototype regularization and EMLA versus CE. Keep projection
and MuonAuxAdamW fixed. Add one fixed-adjustment control: **nine runs, seed 42,
ten epochs each**, evaluated on validation only. Use final-epoch results, not the
best validation checkpoint. The test partition is reserved for confirmation.

| Variant | Normalized | Regularization | Loss |
| --- | --- | --- | --- |
| full | yes | 0.1 | EMLA |
| no_normalization | no | 0.1 | EMLA |
| no_regularization | yes | 0 | EMLA |
| ce | yes | 0.1 | CE |
| no_normalization_no_regularization | no | 0 | EMLA |
| no_normalization_ce | no | 0.1 | CE |
| no_regularization_ce | yes | 0 | CE |
| core_reference | no | 0 | CE |
| fixed_adjustment | yes | 0.1 | fixed adjustment |

The eight factorial cells identify average component effects, pairwise interactions
and the three-way interaction within this recipe. The additional fixed-adjustment
cell distinguishes adaptive EMLA from its constant-gate counterpart at the full
recipe only. `core_reference` retains projection and Muon: it is a factorial anchor,
not a standard bare-linear training baseline. Defer optimizer/projection comparisons.

The screening config sets `screening: true`, one seed and ten epochs; skip `tune`.
The frozen head LR is 0.003, with weight decay 0.001, supported by the bounded
LR checks below.
Batch 512 / 32 workers comes from the retained B200 capacity evidence.
The original long tuning and fixed-LR screening allocations were stopped; preserve
their partial artifacts, but do not present them as completed ablations. Prepare a
fresh output root at the revised source. Earlier protocols remain reproducible from
their pinned commits rather than by mixing old plans with this factorial design.

### Contrasts and confirmation

`factorial.json` reports each seed separately for macro recall, tail recall and NLL.
For a factor set S, the marginal contrast sums cell outcomes with alternating signs
(+ when every factor in S is on), then divides by 2^(3-|S|) to average over the
remaining factors. Main effects are average on-minus-off differences; pairwise
interactions are average differences of differences; the three-way interaction is
a difference of those pairwise interactions. Conditional versions hold the remaining
factors at each on/off setting, so averaging cannot hide opposing interactions.
Recall contrasts are in fractions (multiply by 100 for percentage points); lower
NLL is better. Interactions depend on the outcome scale and are not mechanisms.
Incomplete cubes produce no factorial estimate; never pool seeds to fill cells.
The fixed-adjustment control is excluded from the cube. `paired.json` additionally
retains full-recipe removal contrasts, including full minus fixed adjustment.

A small marginal or full-recipe removal effect is **not** a reason to discard a
factor. Review conditional effects, pairwise and three-way interactions regardless
of marginal size. As practical screening flags, use absolute contrasts of at least
0.5 macro-recall points or 1 tail-recall point, sign reversals, or optimization
failure; these are prioritization thresholds, not significance tests. One seed
cannot establish repeatability or absence of an effect.

Confirm selected contrasts using fresh paired seeds 43 and 44. Replicate **all cells
needed for the contrast**: four for a pairwise interaction at a fixed third-factor
setting, all eight for the three-way or a pairwise interaction averaged over the
third factor. Do not replicate only the best combination. Freeze the selected cells
and epoch budget first, and report confirmation separately from the exploratory
seed. Extend training only if the relevant paired curves leave convergence unresolved.
The initial budget is **90 model-epochs**, with no automatic confirmation sweep.

Backbone LR is head LR/3; one-epoch warmup then the existing cosine schedule.
Smoothing is explicitly `1/512`, projection dropout 0.1, existing augmentations,
FP16 AMP, eager execution. EMA, compilation and quantization are off. Final prototype
weights have zero ordinary weight decay in both head families; all resolved
parameter groups are retained. Sampling, augmentation and regularizer torch RNG
streams are isolated; backbone/projection initialization hashes are saved.
Dropout and backbone stochastic operations can still differ between head recipes;
this is paired initialization and controlled data randomness, not bitwise trajectories.

The normalization package includes spherical initialization, unit prototype norms,
L2 embeddings, frozen bias and nonlinear cosine-to-z-score transformation. Its
control is the existing **BatchNorm-based** `normalized=False` head, not a bare
linear classifier. Disabling projection also removes its dropout/activation.
Fixed adjustment uses EMLA's identical counts, smoothing and centered log-count
offsets with gate one. Inference uses raw classifier logits, with no added priors.

In EfficientNetV2-S, the added projection is the only trainable matrix eligible for
Muon; convolutional backbone parameters and the final classifier use auxiliary
AdamW. Keep this routing fixed across factorial cells. Defer projection removal,
optimizer choice, individual normalization operations, initialization, hierarchy,
extra backbones and additional strengths until a specific result warrants them.

## Learning-rate qualification before screening

The original tuning only exercised LR 0.0003 and did not bracket instability.
The initial screen allocation was stopped; retain its artifacts as setup evidence.
The replacement range and hold checks below support head LR 0.003 for screening.

Run one fresh process per optimizer with the same frozen cohort and sampled images:

```bash
python -m publication.experiments.training_ablations.lr_range /work/results/lr-study /work/results/lr-muon --optimizer muon
python -m publication.experiments.training_ablations.lr_range /work/results/lr-study /work/results/lr-adamw --optimizer adamw
```

Prepare `lr-study` at the probe revision using the ordinary `prepare` command.
Each probe uses 128 batches of uniformly sampled training images at batch 512,
then reuses that sample for a second epoch. The first epoch warms up the head
at a base LR of 0.0003 with backbone LR zero; the second increases head LR from
1e-5 to 1 geometrically, with backbone LR one third of head LR. This shortened
warmup tests the unfreezing transition, not the complete full-cohort schedule.
AMP-skipped updates do not advance the schedule: inspect actual LR coverage.
The head warmup is never run at the upper end of the range.

`lr-curve.jsonl` records per-batch loss, actual group LRs, unscaled gradient norms
before clipping for backbone/projection/classifier, and AMP scales/skips. The
production clipping, optimizer, augmentation, loss and regularizer stay active.
Stop on five consecutive AMP skips, persistent nonfinite losses/regularization,
or smoothed full-model loss exceeding four times its best value after twenty
batches. These are operational divergence signals; distinguish loss divergence
from numerical overflow. A completed ramp without a signal does not establish a
boundary. Unrelated errors, including OOM, fail rather than become LR evidence.

The completed ramps at revision `bb3ec42` used identical sampled images and model
initialization for both optimizers. Both showed useful descent in the broad head-LR
region 0.001–0.01 and deterioration above it. Muon reached 1 without AMP skips;
AdamW recorded two skips near 0.072 and recovered, reaching 0.834. This does not
pinpoint a numerical boundary, nor is that needed for selecting a useful LR.

Check **one** conservative candidate per optimizer: `--hold --upper 0.003`. Holds
warm up to that LR and keep it constant with the backbone active in the second
epoch. Additional checks require an observed failure or concrete ambiguity; do not
refine exact optima or instability thresholds. These short probes establish neither
optimal hyperparameters nor long-run stability.

Both fixed-LR checks completed at revision `bb3ec42` in UCloud job `12410954`,
with 128 head-warmup and 128 backbone-active batches each, zero AMP skips and
decreasing training loss. Results and curves are retained under
`/work/results/hold-09/{muon,adamw}` alongside the referenced frozen inputs in
`/work/results/lr-08/study`. Freeze head LR 0.003 (backbone 0.001) and weight decay
0.001 for the factorial. This is a qualified candidate, not an estimated optimum.

Use two independent single-B200 allocations for the factorial, with disjoint shards
of the same prepared root. Reuse the installed environment and prepared inputs for
same-revision stages; prepare once for the new scientific revision. The first
allocated node can prepare/qualify and begin its shard without waiting for the
second allocation. Check stage completion or failures rather than polling every
batch. Existing focused test evidence is sufficient for unchanged code.

## Frozen cohort

Preparation uses the supplied metadata only; no remote GBIF lookups. Species are
selected from training counts, proportionally within family × abundance quartile.
Quartile ties use species-key order. Largest-deficit allocation and seeded within-
stratum order give nested 256/512/1024 candidates, independent of source row order.
Keep all images for selected species: original sets 0=test, 1=validation, 2–9=train.
Never cap head classes, oversample tail classes, or randomly repartition images.

The selector was exercised on source SHA256
`094afa30bab25daad6583d33055bf7c61bcbade2f7f3c5e212d6b16fbf4429fe`:

| Species | Families | Genera | Training images | Abundance Gini |
| --- | --- | --- | --- | --- |
| Full: 12,632 | 104 | 4,476 | 5,063,857 | 0.5911 |
| 256 | 26 | 231 | 105,972 | 0.5919 |
| **512** | **41** | **437** | **205,557** | **0.5928** |
| 1,024 | 53 | 791 | 411,893 | 0.5918 |

The selected cohort has 25,711 validation and 25,704 test images, all 512 species
represented in each partition, and no GBIF observation crossing partitions.
Proportional sampling omits small families; this is not all-family coverage.
`selection.json`, `species.csv`, `classes.json` and `samples.parquet` retain exact
counts, taxonomy, original sets, IDs and ordering. Image bytes are not hashed:
keep the mounted image source immutable. Missing/corrupt images fail visibly.

## Manual UCloud workflow

Use the **PyTorch app, version 26.05**, with one full **B200** for initial
qualification (the screenshot's `gpu-nvidia-b200-1-gpu`: 48 vCPUs and 288 GB RAM).
Mount `datasets` in Folder #1; inside the job, inspect `/work/datasets` for the
metadata and images. The screenshot does not establish the dataset's subdirectory.
Mount a writable results folder in Folder #2 so `/work/results` is persistent.
Do not assume an arbitrary new directory under `/work` is a mounted drive.

Open the app's browser terminal/console after allocation; SSH is optional and
depends on the selected app's capabilities. The official
[PyTorch app documentation](https://docs.cloud.sdu.dk/Apps/pytorch.html) describes
interactive use, initialization scripts and batch execution.
On each fresh node, first install uv, then clone the repository:

```bash
curl -fLsS https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/asgersvenning/mini_trainer.git /work/mini_trainer
cd /work/mini_trainer
git checkout YOUR_REVIEWED_STUDY_REF
```

`YOUR_REVIEWED_STUDY_REF` must contain this study (push the implementation before
using a remote clone). Keep that revision fixed throughout the campaign. The node
has full sudo access: if base tools are missing on an Ubuntu/Debian image, install
them with `sudo apt-get update` and `sudo apt-get install -y curl git tmux build-essential`.
The scripts arrive with the clone; no separate script transfer is needed.

Copy the supplied config and point its `parquet` and `images` fields at the actual
paths beneath `/work/datasets`. The template's `/work/global_lepi` paths apply when
that dataset folder is mounted directly instead of mounting its parent:

```bash
cp publication/experiments/training_ablations/config.json /work/ablation-config.json
```

`images` is the parent of `<speciesKey>/<filename>`. Set worker count for the actual
allocation before preparation. W&B is enabled by default; authenticate after setup
as described below, before starting qualification.

```bash
cd /work/mini_trainer
source publication/experiments/training_ablations/setup.sh
python -m publication.experiments.training_ablations.study prepare /work/results/lepi-ablations --config /work/ablation-config.json
```

Setup explicitly selects CUDA 13.0 into `/work/venvs/mt-ablations`; override
`MT_TORCH_BACKEND` if necessary. It synchronizes only this dedicated environment.
It also installs uv if absent, so the API pilot works on a fresh node too.
Preparation freezes source, installed packages, configuration, manifests and a local
pretrained weight artifact. An optional `pretrained` config path supplies an offline
torchvision state dictionary; its provenance is recorded as operator-supplied.
Preparation requires a fresh output directory; retain incomplete attempts separately.

### W&B credentials

The setup's `recommended` extra installs the W&B SDK/CLI. The study config sets
`entity: asvenning` and `project: mini-trainer-ablations`, targeting
[the study project](https://forge.coreweave.com/wandb/asvenning/mini-trainer-ablations).
Workers set `WANDB_ENTITY` from that frozen configuration. After sourcing setup,
authenticate in the node's terminal using the interactive prompt:

```bash
wandb login
```

Enter the API key in that prompt, not in chat, the study config or a committed
script. This login is local to the node; repeat it on fresh nodes. Qualification
then exercises actual run creation and metric/figure logging under that project.

For unattended fresh nodes, place the API key in a file on a **private** mounted
folder, restrict access to that folder/file, and export it inside the batch script
before launching the study:

```bash
export WANDB_API_KEY="$(cat /work/private-secrets/wandb-api-key)"
```

Keep this file outside the repository, shared datasets and collected result
artifacts; do not enable shell tracing around credential loading. The controller
passes environment variables to its workers without recording their values in
launch manifests. A custom W&B deployment can additionally set `WANDB_BASE_URL`.
These are standard [W&B SDK environment variables](https://github.com/wandb/wandb/blob/main/wandb/env.py);
no separate connector is required for training uploads. For API allocation, mount
the private folder explicitly; the pilot template does not include credentials.

Then, inside `tmux`, run:

```bash
python -m publication.experiments.training_ablations.study qualify /work/results/lepi-ablations --hours 2
python -m publication.experiments.training_ablations.study run /work/results/lepi-ablations --hours 20
python -m publication.experiments.training_ablations.study summarize /work/results/lepi-ablations
```

The screenshot's one-hour allocation is a starting limit, not a budget for the
complete campaign. Set each stage's deadline to the actual time remaining after
installation and preparation, and use measured tuning runtimes to size later jobs.

### Operational timing and recovery qualification

Before scheduling the campaign, use one bounded timing run on the same allocated
GPU and mounts, after ordinary qualification. The helper arrives with the clone:

```bash
python -m publication.experiments.training_ablations.operational /work/results/lepi-ablations /work/results/lepi-operational-01
```

It uses the frozen study revision/environment and a fresh output directory outside
`runs/`. It samples 4,096 training images uniformly across the selected cohort with
seed 39, retaining the full class vocabulary/counts and original partitions. It
trains the full recipe for two epochs, measures epoch two after four warmup batches,
and verifies checkpoint reload and backbone updates on sampled validation images.
`sample.parquet`, batch timings and `profile.json` preserve the sampled IDs, rates,
GPU peak memory and provenance. W&B uses the separate `operational_profile` name.
The helper does not change qualification, tuning or publication results.

Training intervals include loading, augmentation, compute and logging; explicit
CUDA synchronization makes their completion boundaries observable. A separate
loader-only pass follows training. Both reuse sampled images and may benefit from
filesystem caches: they are not sustained cold-storage measurements. Full-epoch
projections include sampled training and validation rates. The allocation scenarios
show steady rates and a deliberately slower case (half throughput plus the entire
pilot wall time per epoch); neither is a confidence bound. Startup, storage tails,
checkpoint/figure costs and the other recipes still require headroom. Interrupted
training restarts from initialization, so a complete main run must fit an allocation.
Use a larger `--samples` value only if timing variability leaves that decision unclear.

To test larger batches and worker budgets without altering the prepared study,
run separate profile processes with explicit overrides and the same sample count:

```bash
python -m publication.experiments.training_ablations.operational /work/results/lepi-ablations /work/results/capacity-b512-w32 --samples 12288 --batch-size 512 --workers 32
python -m publication.experiments.training_ablations.operational /work/results/lepi-ablations /work/results/capacity-b768-w48 --samples 12288 --batch-size 768 --workers 48
```

An explicit batch overrides `qualified.json` only for that profile. Each output's
`resolved.json` records the settings; frozen campaign files remain unchanged.
A failed CLI run retains `failure.json` and exits nonzero. Continue with a smaller
candidate only when `cuda_oom` is true; data or other failures need investigation.
Use a fresh process/output for each candidate. These paired capacity probes change
batch and workers together to select a practical operating point, not to attribute
throughput to either parameter. Freeze the selected settings before optimizer tuning.

On a fresh qualification directory, also send SIGTERM to the **study controller**
while its first training worker is active, keeping the allocation alive. Confirm
that the worker exits, `failure.json` records interruption, and no `complete.json`
exists for that attempt. Then rerun the same `qualify` command with `--retry`.
The failed attempt must remain and a new attempt must complete. Completed training
with interrupted evaluation is reused instead; CPU recovery tests cover that path.
This is a workflow check, not evidence of checkpoint continuation after interruption.

### Unattended execution through the Web UI

The app's **Batch Mode** accepts a Bash script, executes it when the job starts,
and stops the job when it finishes. An API wrapper is therefore optional for
unattended startup and shutdown; it adds programmatic submission and queueing.

For the first allocation, use the browser terminal to verify mounted paths and
qualify the environment. For subsequent allocations, put the same uv installation,
clone/checkout, setup and chosen study-stage commands in a Bash script stored on
a mounted folder, then select it in Batch Mode. Start it with `set -euo pipefail`,
use the same reviewed revision, and retain logs/results on the mounted results
folder. Use Initialization for setup followed by interactive work; use Batch Mode
when the job should terminate after the study commands. No SLURM or tmux is needed
inside the batch script.

**Replace each `--hours` with the remaining allocation lifetime.** These example
values are limits, not runtime predictions. A minute is reserved for termination
cleanup; tmux does not extend an allocation. No GPU allocation is performed by
the study itself.

Qualification runs four tiny real-image treatments (full, no normalization, core
reference and fixed adjustment), retaining the full classifier vocabulary/counts
and exercising both head types, both loss paths and disabled regularization. It uses training/validation only, two
epochs (warmup followed by backbone updates), at most `max(2 × batch, 128)` records per partition. Only CUDA OOM permits
global batch fallback through 768 → 512 → 256 → 128 → 64 → 32, starting at the
configured batch; other failures stop. Reloaded backbone
parameters must differ from their initialization; BatchNorm buffer changes alone
do not satisfy this check. Freeze the selected batch
before screening. This is infrastructure evidence, not convergence or representative
full-dataset IO evidence. Its counts/metrics must not enter publication quality tables.

Start with one GPU (`--devices 0`). To qualify concurrent IO and run independent
lanes on equivalent GPUs, pass e.g. `--devices 0,1` consistently. There is no DDP
or automatic LR scaling. GPU model/memory must match qualification. Review actual
tuning timings and storage behavior before committing the main-run allocation;
the [existing IO calibrator](../../../dev/ucloud/README.md#calibrate-filesystem-read-concurrency)
is available if loading stalls. Do not interpret tiny warm-image timings as a
full-cohort throughput guarantee.

## Recovery and interpretation

`status ROOT` lists attempts. Failed attempts retain logs and failure records.
After inspecting them, repeat the stage with `--retry`: completed runs are checked
and skipped, successful training is reused for failed evaluation, interrupted
training starts a new numbered attempt from the original initialization. This
does not claim exact checkpoint continuation. A controller lock prevents two
launchers from duplicating the same campaign. Deadlines terminate child process
groups, including loader workers. Source/config/environment or completed-artifact
changes are rejected rather than silently reused.

Every attempt retains resolved settings, commands, initialization hashes, optimizer
groups, trainer checkpoints/figures/learning curves, timings, hardware and logs.
Evaluation reloads final weights in FP32 and saves logits, labels and sample IDs.
Screening, tuning and qualification evaluate validation; non-screening main runs
evaluate test. Metrics are
macro recall (primary), accuracy, NLL, multiclass Brier score and class-balanced
recall in training-frequency tertiles. Ties in tertiles follow frozen class order.
Missing-support recalls are null and excluded from macro means, with support saved.

`summarize` writes `results.csv`, `paired.json`, `paired-summary.json`, `summary.json`
and `macro-recall.png` when main results exist. Seed points and paired mean/SD/range
are descriptive; three seeds are not strong significance evidence. Seed variability
is distinct from finite-test-set uncertainty. Wall timing includes training-stage
construction, validation, diagnostics and saving; it excludes environment setup and
preparation. Allocated-memory peaks are phase peaks, not whole-device memory.

## Optional API pilot

The external [ucloud-api wrapper](https://github.com/GuillaumeMougeot/ucloud-api)
documents submission, persistent client-side queues, mounting and batch termination.
The pilot has submitted successfully on the configured SDU project and completed
fresh-node installation, preparation, six warmup-only GPU runs with W&B uploads,
checkpoint reload/evaluation, and automatic successful shutdown (job 12410894).
That first pilot missed post-warmup backbone updates. Corrected job **12410899**
(commit `932dfa9`, 3 October 2026) passed all six two-epoch treatments at batch 128
on one full B200, with changed backbone parameters verified after checkpoint
reload. Mounted-key authentication worked; all six W&B runs finished with metrics,
and the batch exited with code 0 and UCloud state SUCCESS.

Persistent evidence is on the member drive at
`/12348329/mini-trainer-ablations/results/pilot-authenticated-02` (mounted as
`/work/results/pilot-authenticated-02`). Retain `bootstrap.log`, `exit-code`,
`study/qualified.json`, prepared manifests, and all per-attempt artifacts. W&B run
IDs, in full/no-projection/AdamW/AdamW-no-projection/unnormalized/fixed order, are
`xe611ncd`, `lx1pmlvf`, `r12qrs5x`, `4q4qebia`, `iduoy7bz`, and `1qum5bq2` in
[the study project](https://forge.coreweave.com/wandb/asvenning/mini-trainer-ablations).

These pilots predate the peak-memory reporting correction (`8440f49`): their
`peak_allocated_bytes` fields underreport the batch peaks and must not size future
allocations. Operational job **12410903** (`2e8c752`) subsequently verified the
corrected peaks, controller SIGTERM with worker cleanup, preservation of the failed
attempt, and successful retry followed by all six qualification treatments.
Evidence is under `/12348329/mini-trainer-ablations/results/operations-03`:
`interruption.json`, `study/qualified.json`, and `profile/profile.json`.

Capacity job **12410905** (`fb74a81`) completed both settings below on one full
B200 using the same 12,288-image sample (507 species, seed 39). Both checkpoint
reloads passed without OOM. Profiles and sample hashes are retained under
`/12348329/mini-trainer-ablations/results/capacity-04`, in `b512-w32/` and `b768-w48/`.

| Batch / workers | Training images/s | Validation images/s | Peak allocated GB |
| --- | ---: | ---: | ---: |
| 512 / 32 | 741.50 | 3040.10 | 117.59 |
| 768 / 48 | 732.36 | 3147.48 | 176.15 |

Select **512 / 32** for campaign qualification: both training rates are within 5%,
and the smaller batch leaves substantially more memory headroom. These are single
sampled measurements with warm-cache effects, not evidence of statistical speed
superiority or sustained full-cohort throughput. The paired probes change batch
and workers together. Freeze the qualified batch before tuning; do not change it
between scientific treatments. Full-cohort runtime and concurrent storage load
still need verification during tuning.

Continue development on `research/training-ablations`; pin each node to a published
commit and use a fresh prepared study when source changes.

Keep the wrapper in its own tool environment and pin a reviewed Git revision.
Use interactive `ucloud login`; never copy tokens into configs, logs or this repository.

Keep `ucloud-pilot.toml` beside the `ucloud/` bootstrap directory. Fill the product,
drive paths and full Git commit before submission; select the project using
`ucloud login --project ID`. Use `ucloud products` and `ucloud apps show` to verify
account-specific values. PyTorch 26.05 supports batch scripts and a web terminal,
but its API application definition does not support SSH.

The wrapper uploads only the bootstrap. It installs uv, clones the repository,
checks out the requested commit, and sources `setup.sh` with the explicit CUDA extra.
Do not upload a working tree as the training checkout: the wrapper excludes `.git`,
which preparation needs to record provenance. Mount `global_lepi` directly for the
committed `/work/global_lepi` input paths, and a writable `results` folder. An
alternative configuration can be supplied using `MT_ABLATION_CONFIG` in the batch
command. The spec omits `setup.python="uv"` to avoid its implicit sync.

After preparation, the default pilot waits up to 20 minutes for W&B login. In the
web terminal, run the `wandb login` and `touch .../wandb-ready` commands printed in
the pilot log; create the marker only after successful login. Use `--authenticated`
in place of `--wait-for-wandb` when credentials are already available to the job.
For file-based authentication, mount a separate private credential folder read-only
and prefix the batch command with
`MT_WANDB_API_KEY_FILE=/work/mini-trainer-secrets/wandb-api-key`. The bootstrap reads
the file into `WANDB_API_KEY` with shell tracing disabled; it does not print or copy
the key into the checkout, study config or results. Upload the credential directly
from its local file using `ucloud files upload LOCAL_FILE REMOTE_FILE`; keep that
folder out of source synchronization and result collection. This also requires
`--authenticated` to bypass the interactive login marker.
The pilot stores `bootstrap.log`, `exit-code` and the prepared study on the results
mount. Each attempt requires a fresh output directory. A forced allocation stop
may prevent the exit-code marker from being written; inspect both job state and
study completion markers.

```bash
ucloud q submit publication/experiments/training_ablations/ucloud-pilot.toml --name ablation-pilot
ucloud q daemon --until-idle
ucloud q logs ablation-pilot
```

The pilot prepares and qualifies a separate campaign in a one-hour allocation,
without auto-extension. Acceptance requires persistent study artifacts/logs,
successful process exit **and** actual allocation termination, including checking
a cancelled attempt remains incomplete. The controller must stay available for
queue progression. The wrapper syncs at launch, so never point it at a changing
working tree. Manual allocation remains usable regardless of pilot outcome.

## Checks

```bash
bash dev/check.sh static
bash dev/check.sh test tests/integration/test_publication_ablations.py
.venv/bin/ruff check publication/experiments/training_ablations
.venv/bin/ruff format --check publication/experiments/training_ablations
```

`tests/integration/test_publication_ablations.py` covers cohort selection, loss
controls, RNG isolation, tuning, metrics, interrupted evaluation recovery and tiny
CPU train/reload runs. These do not establish CUDA correctness or live API behavior.
For this research-only change, those focused checks cover the affected boundaries;
the expensive architecture, deployment and quantization suites are not required.
Broaden validation if a later change alters shared trainer/package behavior.

## Dispatch across single-GPU allocations

Prepare and qualify one campaign once, then mount the **same persistent study
root** at the same path in each allocation. Pin the same revision, environment,
dataset paths and credentials. Separate single-GPU nodes can execute disjoint
strides of the frozen plan; this avoids waiting for several GPUs on one node:

```bash
# Separate allocations; each sees its own GPU as device 0.
python -m publication.experiments.training_ablations.study run /work/results/lepi-ablations --devices 0 --shard 0/2 --hours 10
python -m publication.experiments.training_ablations.study run /work/results/lepi-ablations --devices 0 --shard 1/2 --hours 10
```

Each shard exits successfully after its own work. Screening writes its shared
plan under a lock and needs no tuning prerequisite. For a legacy non-screening
campaign, dispatch `tune` first: tuning selection and the main plan appear only
after all eight tuning runs finish. Indices are zero-based; keep the
shard count fixed for each stage. Unstarted runs need no `--retry`; failed or
interrupted attempts require inspection followed by that flag as usual.

Per-run POSIX file locks protect attempt creation and execution, including ordinary
unsharded commands. Overlapping submissions fail visibly before duplicating a run.
The results filesystem must honor these locks across nodes; qualify that behavior
before concurrent dispatch. Sharded tuning does not write intermediate summaries;
run `summarize` after the stage if needed. The final main shard writes the main
summary. Existing commands without `--shard` retain their usual behavior.

Preparation accepts batches 32, 64, 128, 256, 512 and 768. Qualification descends
that ladder from the requested size **only on CUDA OOM**. Keep the default 128
until measured capacity supports a different globally frozen batch and worker
count; no automatic learning-rate scaling accompanies a batch change.
