# Training ablations on UCloud

Research protocol, executable harness and exploratory screening results.
The study uses production trainer APIs through a research builder; it changes no
package defaults. It follows the [publication workflow](../README.md) conventions:
frozen inputs, paired seeds, validation-only choices and retained individual results.

See [presentation protocol and completion workflow](presentation.md) for the
claim-to-evidence mapping, paired tables, figures and final completeness gate.

## Mechanism analysis and corrected PlantNet replication

The scientific questions are distinct: frequency adjustment should reduce learned
frequency preference; EMLA's adaptive gate should preserve early common-class
learning; HGLL plus prototype repulsion should change class geometry and rare-class
confusions; parent supervision should preserve useful taxonomic distinctions under
naturally imbalanced branching. A prototype effect alone does not establish an
embedding or predictive benefit. Equal prediction mass is evaluated under equal
true-class weighting, not demanded on naturally imbalanced images. The angular
reference distribution does not by itself establish posterior calibration.

The two nine-treatment, ten-epoch Lepidoptera studies and the six seed-42,
twenty-epoch duration checks are complete. EMLA's full-recipe tail-recall advantage
over CE is 2.43/2.53 points at ten epochs (seeds 42/43). At twenty epochs it is
2.73 points. Full-recipe macro/tail recall barely changes with the longer schedule.
The regularization interaction is schedule-dependent: its positive unnormalized
EMLA tail effect at ten epochs becomes negative at twenty epochs. Do not claim
that the ten-epoch sign reversal is invariant to training duration.

The existing warmup-epoch confusion matrices show EMLA-minus-fixed common-class
recall gains of 6.88/5.81 points, with rare-class costs of 11.91/11.84 points;
the rare-class gap nearly disappears by epoch two. This supports deferred correction,
not numerical-stability superiority or universal endpoint improvement. Compare
full trajectories, not adjusted training loss magnitudes across different objectives.

Use the existing commands to consolidate immutable artifacts:

```bash
python -m publication.experiments.training_ablations.dynamics STUDY NEW_CURVES
python -m publication.experiments.training_ablations.analysis STUDY NEW_REPORT --geometry
python -m publication.experiments.training_ablations.analysis_plots NEW_REPORT NEW_FIGURES
python -m publication.experiments.training_ablations.embeddings STUDY NEW_EMBEDDINGS --device cuda
```

Analysis now includes identical leaf-probability aggregation for flat/hierarchical
parent metrics, parent frequency/descendant associations, taxonomic error probability
and destination shares, prototype taxonomic neighbourhoods, and per-epoch frequency
recall from sparse confusion counts. Existing scalar logs retain dense gate summaries
when available. Supplementary logs are hashed independently; old completion manifests
are not rewritten. Figures include paired early-learning trajectories and geometry/
hierarchy comparisons. Embedding extraction is validation-only, with at most 32
images per species selected identically across runs; it reports centroid separation
and within-class angular spread. It does not replace full-cohort evaluation.

### Completed hierarchy comparison

All eight ten-epoch runs on the complete-family cohort are complete (1,513 species,
seeds 42/43). A final-report exception in one old controller does not invalidate
the completed, hashed training/evaluation artifacts; reporting was regenerated
with the corrected analysis code. No retraining was required.

With regularization enabled, hierarchy minus species-only supervision changes
species macro recall by -0.211/+0.075 percentage points and rare-species recall by
-0.439/+0.212 points (seeds 42/43). Genus macro recall improves by +0.682/+0.527
points and family macro recall by +0.753/+1.068, using the same leaf-probability
aggregation for both objectives. Thus coarse-rank utility is more repeatable than
species-level improvement. The loss-weight contrast includes reallocating species
supervision across ranks; it does not isolate an architectural change alone.

Regularization's rare-recall effect within the hierarchical objective is
+0.507/+0.906 points, versus +0.570/-0.401 for species-only supervision. The resulting
hierarchy × regularization contrast is -0.063/+1.307 points: a strong seed-dependent
interaction, not established synergy. Prototype effective rank rises from about
445–452 without regularization to 979–1,017 with it. However, hierarchical
rare-to-rare error probability changes from 2.378 to 2.769% in seed 42 and from
2.967 to 2.438% in seed 43. Geometric spreading is replicated; reduced rare-class
confusion is not yet a consistent consequence. Image-embedding occupancy remains
unmeasured by these prototype diagnostics.

### PlantNet normalization × regularization: first seed

All four EMLA cells at seed 42 and ten epochs are complete on the frozen validation
partition (31,097 images, 987 species). Percentages below use raw reloaded FP32
predictions; rare recall averages the least frequent third of classes.

| Normalization | Regularization | Accuracy | Macro recall | Rare recall | Equal-class ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Off | Off | 78.15% | 52.45% | 31.16% | 26.22% |
| Off | On | 78.96% | 52.39% | 29.79% | 22.08% |
| On | Off | 78.83% | 57.45% | 40.43% | 11.85% |
| On | On | 79.07% | 55.31% | 34.50% | 12.13% |

Normalization improves macro/rare recall and equal-class calibration under both
regularization settings. Regularization improves empirical accuracy but reduces
rare recall, especially with normalization (-5.93 points versus -1.37 without it).
The normalization × regularization finite difference is -2.08 macro-recall and
-4.56 rare-recall points. This interaction needs second-seed replication.

Regularization reduces rare-to-rare error probability from 11.85 to 9.88% with
normalization, and from 13.37 to 10.49% without it. With normalization, however,
rare-to-medium and rare-to-common errors increase by 3.80 and 4.10 points. This
supports a changed confusion pattern, not a net rare-class benefit. Equal-class
NLL is 1.828 without regularization versus 1.930 with it for normalized heads;
the regularized pairing does not dominate probability quality in this comparison.

The normalized full model's mean training gate rises from .31/.41 for rare/common
classes during the first epoch to .95/.91 during the last. This verifies adaptive
confidence-dependent adjustment; it is not direct detection of overfitting.
In the first three completed epoch evaluations of the still-active CE/fixed runs,
EMLA preserves more image-weighted accuracy than fixed adjustment: 72.01 versus
62.68% at epoch three, while rare recall is lower (15.50 versus 23.86%). Unlike the
small Lepidoptera cohort, the rare-class gap has not closed by epoch three. These
AMP epoch metrics are provisional trajectories, not final FP32 endpoint results.
Finish CE/fixed, hierarchy and second-seed comparisons before selecting additional
training. Historical PlantNet test results and these validation results remain
separate.

### Corrected PlantNet inputs

Reuse the previously corrected V2 `images_gbif/` and `data_index.json`, originally
referenced at `/dcai/projects/iu_0126/datasets/plantnet`. The existing converter in
`plantnet_hierarchical/general/plantnet300k/format.py` removes genus-only labels,
merges accepted-species synonyms and resizes images to 512 pixels. Reuse its frozen
outputs; do not rerun live GBIF resolution. Retain available conversion provenance
alongside the dataset. Source revision alone does not establish the historical
conversion revision.

[plantnet.json](plantnet.json) expects the dataset root mounted as `/work/plantnet`:
the index's relative paths already include `images_gbif/`. The adapter consumes
`path`, `split`, and leaf-first hierarchical `label` arrays, retains every corrected
species and supplied split, and validates paths and taxonomy without fabricating
GBIF observation IDs or numeric source splits. Class counts come from the index;
the legacy `species` selection limit does not apply. `source_metadata` points to
the original image CSV for split verification, class-merge/exclusion reporting and
an inherited observation-overlap audit. Missing source metadata must be resolved
before the replication is qualified; it is never silently reconstructed.

The transferred corrected cohort contains **987 species, 325 genera and 111
families**, with **243,744 training / 31,097 validation / 31,088 test images**.
The source audit identifies six excluded original labels and three merged
accepted-species groups, with zero cross-partition observations. Training-count
Gini is 0.8421; 19.45% of species are in singleton genera. These are properties of
the preserved corrected cohort, not additional selection criteria.

The source image metadata matches the official V2 MD5
`87e7d4b94f2b709524a7e90c7e9060ba`. The dataset's original `format.py` is retained
with SHA256 `14ef0273fb1bf153af3f14d9581cf89aa56c3eddb7ff39939c5aeb6cb0f7c0a4`;
it differs from the later repository copy in its script header and integrity-error
handling. The corrected data and retained labels were not regenerated.

On UCloud the dataset is stored at
`/12348329/mini-trainer-ablations/datasets/plantnet`. Its `_transfer/transfer-manifest.json`
records nine archive checksums and the original metadata/formatter checksums;
`source-audit/` retains the original-to-corrected class map and audit. The node
verifies the transferred archive bytes and extracted metadata before preparation.

### Frozen targeted matrix

Run seeds 42 and 43 for ten epochs: normalization × regularization under EMLA
(four species-supervised cells), CE and fixed adjustment at the full recipe,
and hierarchical supervision with regularization off/on. The two normalized
species controls are shared with the hierarchy comparison: eight treatments per
seed, sixteen runs / 160 model-epochs. No additional subsetting or rebalancing.
The hierarchy uses the same three ranks and equal rank weights as the existing
hierarchy campaign. Loss contrasts are confined to species supervision; this is
not a complete normalization × loss or hierarchy × loss factorial.

Use the established EfficientNetV2-S, 384-pixel, one-warmup-epoch recipe, head LR
0.003, backbone LR 0.001, batch 512, workers 32, FP16, regularization 0.1, and
no EMA/compilation. Gate logging observes the detached per-example uncertainty
factor by rank and frequency third, preserving the criterion's outputs, gradients
and RNG. It measures uncertainty, not a direct detector of frequency overfitting.

Prepare and qualify once, then run `study run STUDY --shared-queue --hours HOURS`
on two equivalent single-B200 nodes. Reject an unplanned qualification batch change
before main training. Freeze source/environment, W&B identity, artifacts and data
hashes. Initial wall-time planning is 8–12 hours excluding transfer/allocation;
replace this with qualification throughput and validation overhead before launch.
After healthy startup, leave the queue unattended and stop active polling.

No automatic epoch extension or third seed: a duration follow-up must address a
still-changing paired contrast, and an interaction replication must include all
four necessary cells. Keep cohorts, budgets and seeds separate. Preserve test
partitions until analysis choices are frozen, then evaluate every cell entering a
reported contrast, not only the best variant. Optimizer/backbone/strength sweeps
and individual HGLL-component attribution remain outside this increment.

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

## Initial screening results

Campaign `factorial-11` completed all nine final-checkpoint evaluations on
4 October 2026, source `d628ce5`, seed 42, using the recipe above. Both UCloud
jobs (`12410957`, `12410958`) finished successfully. Retained evidence is under
`/12348329/mini-trainer-ablations/results/factorial-11/study/runs/`, with each
variant in `<variant>_seed42/attempt-000/`. Completion manifests record artifact
hashes; all six JSON artifacts per run were independently downloaded and verified.
Checkpoint and prediction bytes were not independently downloaded for this review.
Backbone and projection initialization hashes match across all nine runs.

Final validation results (recall and accuracy in percent):

| Variant | Accuracy | Macro recall | Tail recall | NLL |
| --- | ---: | ---: | ---: | ---: |
| full | 97.9736 | 97.2469 | 96.5591 | 0.089494 |
| no_normalization | 97.7519 | 96.9388 | 96.1375 | 0.097816 |
| no_regularization | 97.8997 | 97.1650 | 96.6861 | 0.093018 |
| ce | 98.0203 | 96.3054 | 94.1300 | 0.079053 |
| no_normalization_no_regularization | 97.6314 | 96.5243 | 95.5263 | 0.097794 |
| no_normalization_ce | 97.8414 | 96.0253 | 93.9457 | 0.094592 |
| no_regularization_ce | 98.0009 | 96.2262 | 94.2670 | 0.080388 |
| core_reference | 97.7714 | 95.8936 | 93.7296 | 0.094031 |
| fixed_adjustment | 97.8803 | 97.4536 | 97.2217 | 0.097551 |

Equal-cell contrasts computed by `study.factorial_contrasts` (recall differences
in percentage points; positive NLL differences are worse):

| Contrast | Macro recall | Tail recall | NLL |
| --- | ---: | ---: | ---: |
| Normalization | +0.3904 | +0.5758 | -0.010570 |
| Regularization | +0.1768 | +0.1409 | -0.001069 |
| EMLA versus CE | +0.8561 | +2.2092 | +0.007515 |
| Normalization × regularization | -0.1926 | -0.5457 | -0.002721 |
| Normalization × EMLA | +0.1681 | +0.4299 | +0.008042 |
| Regularization × EMLA | +0.1427 | +0.2026 | -0.001364 |
| Three-way interaction | -0.2800 | -0.3851 | -0.001649 |

EMLA improves macro and tail recall in all four conditional comparisons with CE.
Regularization's small average effect conceals a tail-recall sign reversal under
EMLA: +0.6113 points without normalization versus -0.1270 with it (interaction
-0.7382 points). This is an exploratory combination effect, not evidence of a
mechanism or a repeatable effect. No averaged interaction crosses the predefined
recall-magnitude flags; that does not establish absence of interactions.

At the full recipe, adaptive EMLA minus fixed adjustment gives -0.2067 macro-recall
points, -0.6625 tail-recall points, +0.0933 accuracy points and -0.008056 NLL.
Thus this screen supports a recall benefit from adjustment versus CE, but does
not establish that the adaptive gate improves recall over constant adjustment.
All findings are validation-only, one-seed results within a ten-epoch budget.

### Frozen overnight replication

The overnight budget is approximately ten wall-clock hours on two full B200 nodes.
[overnight.json](overnight.json) repeats **all nine variants at seed 43,
ten epochs each**: nine runs / 90 model-epochs. [duration.json](duration.json)
defines the separate duration check, restricted to these six variants:
`full`, `no_normalization`, `no_regularization`,
`no_normalization_no_regularization`, `ce` and `fixed_adjustment`, seed 42,
20 epochs from the same initialization, with all other settings unchanged.
This adds 120 model-epochs, for approximately 17.5 GPU-hours total. Each node
receives three long runs and either four or five short runs. At the observed
50 minutes per ten epochs, the two queues take approximately 9 hours 10 minutes
and 8 hours 20 minutes, plus setup and evaluation. Allocate 12 hours per node as
a buffer, with automatic exit on completion; queue delay is outside this estimate.

This replaces the narrower proposed confirmation subset. Retaining complete cubes
replicates every conditional, averaged pairwise and three-way contrast, while the
fixed-adjustment control tests the adaptive gate at the full recipe. Keep the
fresh seed separate from the exploratory seed 42. Analyze the 20-epoch duration
check separately; do not pool different epoch budgets. Its matched seed-42
ten-epoch counterparts already exist. Because cosine decay spans the requested
budget, this tests a longer training schedule, not simply ten extra epochs at
the original terminal learning rate.

All nine screening curves reached 99.8–99.9% training accuracy; validation accuracy
changed by at most 0.06 points between epochs nine and ten, with flat or worsening
validation loss. This does not establish convergence: the learning rate was
approaching zero. Review of all 4,010 training-batch records per run showed
epoch-boundary changes despite smooth learning rates, without sustained loss
explosion; do not assign a cause to those changes from curves alone. Criterion
loss excludes the separately logged regularizer and differs across loss recipes.

Epoch confusion-count artifacts recover macro/tail recall despite their absence
from scalar learning logs. Full-recipe EMLA's tail advantage over CE shrinks from
about seven points at epoch two to 2.43 at epoch ten. Regularization's conditional
effects also change during training. These are AMP epoch evaluations, distinct
from the final reloaded FP32 evaluations above. The duration check tests these
contrasts under a longer schedule, including normalization × regularization under
EMLA. Its one seed cannot establish repeatability, and it does not cover the full
three-way interaction at 20 epochs.

The final class-distance figures show substantially more uniform prototype
geometry with regularization and pronounced clusters/bands without it in both
head families. These figures apply the same cosine-based transform, but do not
establish taxonomic alignment or downstream representation quality. Quantify the
geometry from retained artifacts before proposing another training sweep.

Use the same cohort, splits, initialization family, batch 512, workers 32, LR and
all other training settings. The `screening: true` execution option selects the
fixed-LR path and validation-only evaluation; it does not mean hyperparameters
will be selected again. Reserve test data for a later frozen assessment. Report
each seed's contrasts and their agreement or disagreement before interpreting a
mean. Two seeds cannot establish absence of a small effect.

Each node installs uv, clones the pinned training source, loads the existing
private W&B credential and runs its static shard unattended. The first node
prepares and qualifies the shared study; the other waits for qualification, then
runs independently. Retain node logs and exit codes, individual completion/failure
markers, checkpoints, predictions and W&B runs on mounted storage. A failed
treatment stops its shard rather than silently skipping it; the other shard can
continue. The last successful shard writes the complete study summaries. No
automatic tuning, changed batch size during the main study, retries or additional
experiments are scheduled after these 15 runs. The duration stage uses its own
prepared root and six-cell frozen plan, rather than filling out an unrequested
20-epoch factorial. Both stages retain validation-only evaluation.

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

### Reproducible mechanism analysis

Analyze existing final predictions; no training or image inference is required:

```bash
.venv/bin/python -m publication.experiments.training_ablations.analysis \
  /path/to/study /path/to/new-analysis
.venv/bin/python -m publication.experiments.training_ablations.analysis_plots \
  /path/to/new-analysis /path/to/new-figures
```

The input can be the mounted study or a local mirror containing `prepared.json`,
`classes.json`, `samples.parquet` and completed `runs/*/attempt-*` artifacts:
`complete.json`, `run.json`, `evaluation.json` and `predictions.npz`. Consumed
artifacts are hash-verified; prediction IDs, labels and order must match the
recorded split and class vocabulary. Only the latest attempt is considered;
incomplete attempts are reported, never replaced by an older successful attempt.
Outputs use a new directory outside the study. Training artifacts are unchanged.

The array-based analysis functions are independent of campaign paths and seeds.
They share frequency groups, prior definitions and output tables across treatments:

- Per-class recall, support, hard prediction mass, soft probability mass, and descriptive frequency associations
  (Spearman and slope against natural-log training count). Equal-class prediction
  mass averages the row-normalized confusion matrix over observed true classes;
  it does not require uniform predictions on an imbalanced image population.
- Tail/mid/head error flows exclude correct predictions. `error_probability` is
  the average per-class probability of that error destination; the separate
  conditional error share uses total source errors as its denominator. Pair tables
  retain counts, conditional probabilities and same-genus/family indicators. These
  support stratification, not a claim that taxonomy has already been controlled.
- NLL, multiclass Brier, fixed-bin top-label reliability and ECE under empirical
  and equal-observed-class priors. Empty reliability bins and missing class support
  remain explicit. ECE depends on bins and is not proof of calibration.
- Complete-cube contrasts for balanced NLL/Brier and rare-to-rare error use the
  existing factorial implementation and comparability guards. Partial cubes and
  undefined metric contrasts are omitted, not imputed; fixed adjustment remains
  outside the cube. Analyze different epoch budgets in separate roots.

Add `--geometry` to load verified `model/weights/last.pt` checkpoints on CPU.
Use `--variants full no_regularization no_normalization
no_normalization_no_regularization` to restrict the workload. Geometry includes
effective rank of the unit-prototype Gram matrix, mean resultant length,
nearest-prototype angles and frequency-pair mean angles. This measures prototype
coverage, not the full distribution of learned image embeddings.

The spherical-null diagnostic uses common random directions (`--seed 20261004`,
`--null-samples 8192` by default), unit prototypes, the cosine-to-z transform and
zero bias for every treatment. It is an angular reference comparison, **not**
the actual unnormalized head's BatchNorm/bias forward pass. Hard class occupancy,
mean softmax mass and plug-in Monte Carlo standard errors are retained; the default
sample size gives coarse occupancy estimates for 512 classes. Duplicate/tied
prototypes are rejected rather than assigning all tied wins to the first class.

Each run writes `summary.json`, `classes.csv`, `confusion_flows.csv`,
`reliability.csv` and `error_pairs.csv`. Top-level `report.json`, `factorial.json`
and `provenance.json` retain run identities, split, source/artifact hashes,
analysis settings and package versions. The plotting command consumes only these
tables and records their hashes. It does not refit metrics or load models.
Correlations near ceiling recall require inspection of support, ties and the
frequency-stratified plots; neither correlation nor error-flow changes alone
identify a causal mechanism. Seed variation and Monte Carlo uncertainty are
distinct from finite-validation-set uncertainty.

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

The bounded MIG trial (`12410956`, revision `bb3ec42`, exit 0) used the same
12,288-image sample hash as the capacity probe. Its artifacts are retained at
`/work/results/mig-10/profile`. UCloud reports the product
`gpu-nvidia-b200-1-mig.1g` as a 1/7 allocation with 23 GB GPU memory, six vCPUs and
36 GB host memory. Batch 64 / eight workers achieved 168.72 training images/s,
488.93 validation images/s and 15.23 GB peak allocated GPU memory.

Compared with full-B200 batch 512 / 32 workers (741.50 training images/s), this is
4.4 times slower per run but 1.59 times the training throughput per allocated GPU
fraction. Seven such slices would project to 1,181 images/s; simultaneous-slice
contention and queue availability were not measured, so this is not demonstrated
aggregate throughput. Setup and evaluation overhead are excluded from those rates.
Both configurations used the same images, but different batch sizes/step counts:
this is an operational comparison, not evidence of equivalent optimization.
MIG is promising for independent runs configured for smaller batches; do not mix
batch-64 MIG results into the frozen batch-512 factorial. A change of batch size
requires a separate scientific configuration, including its LR qualification.

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

### Complete-family hierarchical comparison

`hierarchy.json` defines a separate, validation-only cohort and eight paired runs:
normalization and Muon fixed, species-only versus species/genus/family EMLA,
crossed with prototype regularization off/on, seeds 42 and 43. Ten epochs retain
the prior screen's schedule; earlier duration evidence did not establish a
consistent benefit from twenty epochs. Existing-cohort runs are selection evidence,
not controls for this cohort.

Selection uses training metadata only. Permute sorted family keys with NumPy's
seeded generator (seed 20261004), retaining each whole family if its training
images fit the remaining 500,000-image budget. Preserve every training-observed
species in each retained family and every source sample/split for those species.
There is no class balancing, per-class cap or selection on validation performance.
Freeze and verify the resulting family list and species count in the configuration.
This yields 1,513 species, 517 genera, 30 families and 499,986 training images.
Species in singleton genera comprise 19.8%, compared with 19.3% in the source
and 75.4% in the previous 512-species subset. The largest families cannot fit this
budget; inference is conditional on the selected complete families, not a claim
that all source-family distributions are represented. `selection.json` records
source and cohort branching, abundance and split support; `classes.json` records
rank vocabularies, training counts and child-to-parent mappings. Cross-environment
cohort verification uses the ordered table-content hash in `selection.json`;
Parquet byte hashes include writer metadata and can differ across pandas versions.
Prepared-node artifact integrity continues to use byte hashes.

All treatments use the same bottom-up `HierarchicalClassifier`, initialization,
loader, rank labels and aggregated parent predictions. The flat control sets loss
weights to `[1, 0, 0]`; the hierarchical objective uses `[1/3, 1/3, 1/3]`. This
explicit fixed-total weighting choice changes how supervision is distributed
across ranks while avoiding an automatic threefold coefficient increase. It does
not guarantee equal gradient magnitudes. EMLA and the production rank-specific
smoothing rule apply at each rank; no empirical prior is installed in the head.
The treatment is **adding the weighted hierarchical EMLA objective**, not an
isolated architecture effect or isolated parent-EMLA effect. Uniform leaf and
uniform parent priors need not agree under unequal branching.

Use the existing prepare/qualify/run commands with this configuration. Qualification
covers both objectives with regularization enabled, two tiny epochs (including
backbone updates), checkpoint reload and final FP32 evaluation. Keep batch 512,
32 workers and the original LR/optimizer settings unless a revised protocol is
explicitly frozen. The general runner can reduce batch on OOM; the campaign
bootstrap must reject such a reduction before main dispatch.

For two independent single-GPU nodes mounting the same prepared root, run on each:

```sh
python -m publication.experiments.training_ablations.study run /work/results/CAMPAIGN/study \
  --shared-queue --devices 0 --hours REMAINING_HOURS
```

Both controllers scan the same frozen, seed-blocked plan. A nonblocking run lock
claims work; busy runs are skipped and completed runs verified rather than
repeated. Failures retain evidence and require explicit `--retry`. There are no
fixed node shards or implicit retries. The last controller writes the summary
once all runs have verified completion. Confirm cross-node filesystem locking
before dispatch; local concurrency tests cannot establish remote lock behavior.

`paired.json` records hierarchy-minus-species effects conditional on regularization
and their difference-in-differences, per seed. Report both seeds rather than
claiming population confidence from two replicates. Final evaluation records
species, genus and family metrics from the same leaf probabilities for every
arm. Retained leaf logits also support the existing mechanism analysis: class
frequency versus prediction mass/performance, within/across-parent and rare-class
confusion, calibration and prototype geometry. Relate these outcomes to descendant
counts as well as image frequency. Parent-EMLA decomposition and additional
seeds/durations are follow-ups motivated by the observed contrasts, not an
automatic expansion of this campaign.

### Lower unique support within the same classes

`support-sensitivity.json` caps the least-supported third (505 of the same 1,513
species) at 16 distinct training images. For each affected species, the selected
images are sampled deterministically and drawn with replacement until that species
has its original number of training examples. This preserves per-class image
exposures, total batches, optimizer updates, LR schedule, class-frequency counts
used by EMLA/regularization, and all validation/test rows. It isolates unique-image
support in the tail from changes to the learned class prior. On the frozen cohort,
it reduces distinct training images from 499,986 to 477,377 while keeping 499,986
training draws; the capped tail has original support 38–92 images (median 58).

The four new runs compare the regularized species-only and hierarchical objectives
at seeds 42/43. Pair each against the corresponding completed full-support runs in
`hierarchy-14`; interpret the difference-in-differences as whether hierarchy changes
sensitivity to reduced unique tail data. This is a targeted interaction check, not a
support dose-response or a second full factorial. Keep test evaluation reserved
until the validation contrasts and analysis choices are frozen.

The two single-GPU UCloud jobs share the prepared study root and claim runs through
the existing cross-node lock protocol. Their fresh-node bootstrap installs `uv`,
clones the pinned repository revision, prepares and qualifies once, then launches
the shared queue. Fill `REVIEWED_COMMIT` in both TOML commands after pushing the
reviewed branch revision. Submit independently so UCloud can allocate them whenever
capacity becomes available:

```bash
ucloud login --project 6a9d3c0b-52bc-4652-94a6-1411d59b958e
ucloud q submit publication/experiments/training_ablations/ucloud/support-16/node-0.toml
ucloud q submit publication/experiments/training_ablations/ucloud/support-16/node-1.toml
ucloud q daemon --until-idle
```

Retain both node logs and exit markers under
`/12348329/mini-trainer-ablations/results/support-16`. The coordinator fails before
training if the shared filesystem lock test fails, the frozen tail count/cap differs,
qualification changes batch size, or less than five hours remain for the paired run
lanes.
