# Studies

What each study id of `data/snapshots.json` is (the ids of `studies()`, `runs()`, `scores()`, ...).
The ids of the studies used in the paper's analyses are mapped to datasets in `cfg$study_dataset`
(`helpers/config.R`).

| Study | Dataset | Kind | Runs |
|---|---|---|---|
| `gefion-global-lepi` | Global Lepidoptera (12,632 species) | architecture comparison | 12 models |
| `gefion-plantnet` | Pl@ntNet300K (987 species) | architecture comparison | 12 models |
| `gefion-flemming` | Expert labelled AMI | external evaluation of the `gefion-global-lepi` models | |
| `lepi-1513` | Global Lepidoptera, 1513-species subset | ablation | 19 configurations × 3 seeds |
| `lepi-1513-capped` | same | ablation, low support | 2 configurations × 3 seeds |
| `lepi-512` | Global Lepidoptera, 512-species subset | ablation | 13 configurations × 3 seeds |
| `lepi-512-duration` | same | ablation, 20 epochs | 6 configurations × 3 seeds |
| `plantnet` | Pl@ntNet300K (987 species) | ablation | 19 configurations × 3 seeds |
| `factorial-11` | Global Lepidoptera, 512-species subset | early exploratory ablation | 9 configurations, seed 42 |
| `overnight-12` | same | seed replication of `factorial-11` | 9 configurations, seed 43 |
| `duration-12` | same | duration check, 20 epochs | 6 configurations, seed 42 |
| `hierarchy-14` | Global Lepidoptera, 1513-species subset | hierarchy study | 8 runs |
| `support-16` | same, 505 species capped | low support, paired with `hierarchy-14` | 4 runs |
| `plantnet-16` | Pl@ntNet300K (987 species) | early targeted replication | 16 runs |

The ERDA folders (`snapshots.json`) are `ablation-*` (`lepi-512`, `lepi-512-duration`, `lepi-1513`,
`lepi-1513-capped`, `plantnet`), `gefion-*` (the two architecture comparisons and `gefion-flemming`),
`evidence-*`/`replicates-*` (the earlier studies from `factorial-11` to `plantnet-16`),
`checkpoints-*` and `heads-*`.

## Descriptions

**duration-12.** The original training-duration check on the 512-species Lepidoptera subset. Six
selected configurations were trained from initialization for 20 epochs at seed 42, compared with
their ten-epoch counterparts in factorial-11. It tests whether normalization–regularization
interactions and CE/fixed-adjustment/EMLA differences persist under a longer schedule. Because the
cosine learning-rate schedule also stretches, this is a schedule-length comparison, not simply
continuing the ten-epoch models.

**factorial-11.** The initial exploratory study on 512 Lepidoptera species, with nine
configurations trained for ten epochs at seed 42. Eight form the normalization × regularization ×
CE/EMLA factorial; the ninth uses fixed adjustment with normalization and regularization enabled. It
identifies conditional effects and interactions rather than merely ranking complete recipes.
Projection and optimizer are held fixed, and the unnormalized control uses the existing
BatchNorm-based head.

**gefion-flemming.** External evaluation of the Global Lepidoptera models on the Flemming
dataset, rather than a separate training campaign. It compares transfer behavior across three
backbones and four classifier architectures. The exported evaluation contains 50,991 known-species
and 7,649 unknown-species images, making it useful for assessing domain shift and behavior outside
the training vocabulary; known- and unknown-species results require separate interpretation.

**gefion-global-lepi.** The large-scale Global Lepidoptera architecture comparison:
EfficientNet-B0, EfficientNetV2-S and ViT-L/16, each with flat, hierarchical, conditional and
independent classifiers, giving twelve models with ten logged training epochs. The retained
evaluation covers 632,913 images and 12,632 species. It provides evidence about architecture and
backbone dependence at scale, but does not isolate the individual methodological additions as
tightly as the controlled ablations.

**gefion-plantnet.** The corresponding twelve-model architecture comparison on the
corrected PlantNet300K dataset, using the same three backbones and four classifier types. Its test
partition contains 31,088 images from 987 species, with hierarchical models covering five taxonomic
ranks. It supplies cross-dataset architecture evidence, but its training recipe differs from the
later UCloud ablations, so results should be compared within the study before drawing broader
conclusions.

**hierarchy-14.** The first targeted hierarchy study, using 1,513 species across 517 genera and 30
complete families, selected within a roughly 500,000-training-image budget. Eight runs cross
species-only versus species/genus/family supervision with regularization off/on, at seeds 42 and 43
for ten epochs, while normalization and EMLA remain enabled. Preserving complete families avoids the
distorted branching of the earlier 512-species subset. The contrast tests hierarchical supervision
and its interaction with regularization, rather than architecture alone.

**lepi-1513.** The expanded ablation study on that same complete-family Lepidoptera cohort: 19
configurations × three seeds (42–44), ten epochs each. It combines a no-projection baseline, the
full normalization × regularization × CE/fixed-adjustment/EMLA matrix under species supervision, and
six normalized hierarchical configurations crossing regularization with the three losses. This is the
main controlled study for examining the methodological ladder and whether adjustment and
regularization behave differently when hierarchical supervision is present.

**lepi-1513-capped.** A targeted lower-support companion to lepi-1513, comparing the fully equipped
species-only and hierarchical configurations across three seeds, giving six ten-epoch runs. The
least-supported third of species is limited to 16 distinct training images per species, then sampled
repeatedly to preserve original class exposure counts and optimizer-step budgets. It asks whether
hierarchy changes sensitivity to reduced unique training information, while keeping the vocabulary,
evaluation images and training-frequency priors fixed.

**lepi-512.** The expanded species-only ablation on the original 512-species subset, with 13
configurations × three seeds, ten epochs each. It includes the full normalization × regularization ×
CE/fixed-adjustment/EMLA matrix plus the no-projection baseline. Compared with the initial screening,
it provides more seed replication and tests fixed adjustment throughout the
normalization–regularization matrix, rather than only at the full recipe. Its reduced hierarchy makes
it primarily evidence about species-classification components.

**lepi-512-duration.** The expanded duration comparison on the 512-species cohort: six
configurations × three seeds, twenty epochs each, paired with the corresponding ten-epoch
configurations in lepi-512. It covers all four normalization–regularization combinations under EMLA,
plus CE and fixed adjustment at the full normalized, regularized recipe. It tests schedule dependence
more robustly than the original single-seed duration-12 study.

**overnight-12.** A complete second-seed replication of factorial-11: the same nine configurations on
512 species, ten epochs each, at seed 43. Repeating the entire factorial preserves the ability to
check all conditional, pairwise and three-way effects rather than reproducing only selected winners.
It is a seed-replication study; the separately named duration-12 supplies the longer-schedule
comparison.

**plantnet.** The expanded cross-dataset ablation on corrected PlantNet300K, retaining 987 species
and the supplied train/validation/test partitions. Like lepi-1513, it contains 19 configurations ×
three seeds, ten epochs each, covering the species-level factorial, no-projection baseline and
hierarchical loss comparisons. Here hierarchical supervision spans species, genus, family, order and
class, so it tests whether the methodological findings extend to plants, while differing in hierarchy
depth from the Lepidoptera campaign.

**plantnet-16.** The earlier, targeted PlantNet replication: eight configurations × two seeds, ten
epochs each, giving sixteen runs. It tests normalization × regularization under EMLA, adds CE and
fixed-adjustment controls at the full recipe, and compares hierarchical supervision with
regularization off/on. Unlike the later plantnet campaign, it uses three taxonomic ranks and does
not contain the complete loss-by-normalization-by-regularization matrix.

**support-16.** The original lower-unique-support study paired with hierarchy-14. Four ten-epoch runs
compare regularized species-only and hierarchical EMLA models at seeds 42 and 43, after capping the
least-supported 505 species at 16 distinct training images each. Repeated sampling preserves the
original training exposure counts and schedule. Its purpose is a focused hierarchy ×
unique-data-support comparison; lepi-1513-capped carries that question into the later three-seed
campaign.

## Correspondence between the architecture comparison and the ablations

The `gefion-*` models all use the full recipe (hidden layer, normalization, regularization, EMLA loss)
and differ in backbone and head (flat, hierarchical, conditional, independent). In the ablation studies
(`plantnet`, `lepi-1513`) the same components are switched on and off. For EfficientNetV2-S on
Pl@ntNet300K (the test set is the same, 31,088 images in both):

| `gefion-plantnet` | ablation `plantnet` variant |
|---|---|
| `efficientnet_v2_s_flat` | `full` |
| `efficientnet_v2_s_hierarchical` | `hierarchy_regularized` |

The conditional and independent heads, and the EfficientNet-B0 and ViT-L/16 backbones, have no
counterpart in the ablations. The training recipes differ (`gefion-plantnet` / `plantnet`): batch size 64 /
512, learning rate 0.001 / 0.003, weight decay 0.01 / 0.001, optimizer `MuonAuxAdamW` / `muon`; both
train for 10 epochs in float16. The recipe is therefore the same in what is switched on, not in the
training hyperparameters.
