# lepinet recipe presets

These presets reproduce lepinet's EfficientNetV2-S recipe (originally fastai) with opt-in
mini_trainer components, assembled by `mini_trainer.hierarchical.recipes.LepinetBuilder`. Library
defaults and other recipes, including Nemo's, are unchanged. Each preset spells out every setting,
so a variant changes one line.

| Preset | Head | Loss weights (species, genus, family) |
| --- | --- | --- |
| `lepinet-I.yaml` | `IndependentClassifier`: one cosine head per level | 1, 1, 1 |
| `lepinet-BU.yaml` | `HierarchicalClassifier`: one species head; genus and family are its supervised marginals | 1, 1, 1 |
| `lepinet-F.yaml` | `HierarchicalClassifier`, species loss only; marginals at inference | 1, 0, 0 |

## Run

```bash
mt_train --config dev/lepinet/lepinet-I.yaml --input <gbifxdl.parquet> --output <runs> --name <run> [--seed N] [--num_workers N]
mt_hpredict --weights <runs>/<run>/weights/last.pt --input <gbifxdl.parquet> --output <predictions> --name test
PYTHONPATH=<lepinet>/data/tools/mini_metrics-61db373 <lepinet>/.venv/bin/python dev/lepinet/score.py \
    --lepinet <lepinet> --predictions <predictions>/test/mini_metric.csv --expect-instances 629742
```

Score `last.pt`: lepinet evaluates its final epoch, while mini_trainer's `best.pt` is chosen by
validation accuracy. Prediction reads its image settings from the checkpoint metadata, so
`mt_hpredict` reproduces the validation transform without the recipe builder. `score.py` keeps
only test images of species in the model's vocabulary (set 0 restricted to the 12,041 training
species, 629,742 images) and scores them with lepinet's rescoring function: mini_metrics' current
macro convention at threshold 0, with a calibrated threshold over `--seeds` calibration splits.
Never filter the test fold by class counts.

## Trap crops

```bash
python dev/lepinet/flemming.py index --parquet <lepinet>/data/flemming/flemming_names_labels.parquet --images <flemming>/images --out trap.json
mt_hpredict --weights <run>/weights/last.pt --input <flemming>/images --data_index trap.json --output <predictions> --name trap
PYTHONPATH=<lepinet>/data/tools/mini_metrics-61db373 <lepinet>/.venv/bin/python dev/lepinet/flemming.py score \
    --lepinet <lepinet> --predictions <predictions>/trap/mini_metric.csv
```

One prediction run covers the Flemming trap set (47,905 crops) and its probe and probe-HO splits.
`score` writes each set in lepinet's `predictions.csv` layout and scores it with lepinet's
`dev/093_rescore_mini_metrics.py`, including calibration on whole (trap, night) groups. On lepinet's
V0BU predictions it reproduces lepinet's scores (species 0.3304, 0.3508, 0.2878).

## Self-training data

`pseudo.py` builds the shift runs' training data from a teacher's trap predictions with lepinet's
rules: `select` is `dev/065_pseudolabel.py`'s strict gate (adapt crops at or above the 70th
percentile of species confidence, ties kept), `merge` is `dev/066_combine_pseudo.py`, and `--fold`
applies `dev/089_trap_holdout.py`'s folds.

```bash
python dev/lepinet/pseudo.py select --predictions <predictions>/trap/mini_metric.csv \
    --adapt <lepinet>/data/flemming/splits/flemming_adapt.parquet --out pseudo.parquet [--true-labels]
python dev/lepinet/pseudo.py merge --global-parquet <gbifxdl.parquet> --labels pseudo.parquet --target-frac 0.02 [--fold LV1] \
    --out <data>/train.parquet
while IFS=$'\t' read -r src dst; do mkdir -p "<data>/images/${dst%/*}" && cp "<flemming>/images/$src" "<data>/images/$dst"
done < <data>/train.parquet.staging.tsv
```

`<data>/images` also needs the global images. `--target-frac` counts all global rows, as dev/066 does;
`train_share` in `<out>.summary.json` is the share of training rows.

| Run | `merge` | lepinet (12,230 pseudo-labels) |
| --- | --- | --- |
| Self-training doses (lepinet: 0.24 / 2.6 / 12.3 % of training rows) | `--target-frac 0` / `0.02` / `0.1` | 1x / 11x / 58x |
| T1rep, true labels at 2.6 % | labels from `select --true-labels`, `--target-frac 0.02` | other crops: dev/067's `labels_12230.parquet`, spread over (trap, night) groups; 11x |
| X1 / X2 | `--fold LV1` … `SS4` / `--fold site-LV`, `site-OH`, `site-SS` | 11-13x / 13-23x |

For X1 and X2, `pool --pattern '<predictions>/holdout-{fold}/trap/mini_metric.csv' --folds trap|site`
keeps each crop's prediction from the fold that never saw its trap or site; score its output with
`flemming.py score`. On lepinet's `pseudo_strict.parquet` and `labels_12230.parquet`, `merge` reproduces
lepinet's summaries for all of these runs, and B3rep5x and B3rep26x row for row.

## Open-set scores

`python dev/lepinet/openset.py <mt_hpredict arguments>` writes `summary.parquet`: per image and rank, the
top two raw outputs, their logsumexp, the softmax entropy and the argmax, enough for the max-logit, energy,
MSP, entropy and margin rules without the full log-probabilities.

## What the recipe sets

| Component | Setting | Implementation |
| --- | --- | --- |
| Classes | Species with ≥ 50 images outside set 0; rows of other species dropped | `min_count`, `drop_unknown_classes` |
| Reading | Largest square, random offset in training, centred otherwise; PIL bilinear to 460 | `resize_method: crop`, `read_size` |
| Augmentation | Per image: dihedral (p 0.5), rotation ±15° and zoom 1–1.1 (each p 0.75), one bilinear resample to 256 with reflection padding | `RandomAffineResize` |
| Evaluation input | Centre square at 460, bilinear resample to 256 | `ResampleNormalize` from checkpoint metadata |
| Sampling | Row weight `count(species) ** -0.5`, with replacement, one draw per training row per epoch | `class_sampling_power`, `WeightedEpochSampler` |
| Head | Cosine head with trainable per-class scale and bias, in float32 without TF32; lepinet's repulsion init | `trainable_scale`, `trainable_bias`, `float32_head`, `legacy_spherical_init` |
| Optimiser | AdamW (0.9, 0.999), one learning rate 1e-3 and weight decay 0.01 on every trainable tensor | `optimizer_cls`, `backbone_lr`, `decay_all` |
| Schedule | Per step: linear from lr/100 over 0.5 epoch, then cosine to 0, same curve for all groups | `lr_schedule_builder_kwargs` |
| Loss | Per-level cross entropy, smoothing 1/C at species (shrunk per level), no prototype regulariser | defaults, `strength: 0` |
| Precision | float16 autocast, GradScaler 2^16 with growth every 2,000 steps; clipping at 5 | `dtype`, `scaler_builder_kwargs` |
| Validation | Per-level macro-F1 over classes present in the truth, each epoch; no diagnostic figures | `macro_f1`, `diagnostics: false` |

Image paths are decoded with PIL, which matches fastai's decoding and resizing byte for byte.
The crop reader normalises with the backbone transform's `mean` and `std` when it exposes them, as
torchvision's transforms do. mini_trainer's timm wrapper exposes neither, so timm backbones are read
with ImageNet statistics. That is right for `convnext_large.dinov3_lvd1689m` and
`tf_efficientnetv2_b0.in1k`, but wrong for a timm checkpoint trained with other statistics (check
its `pretrained_cfg`).
Random crops cannot be cached. The other knobs follow the comparison rules in
[the training feature protocol](../../docs/training-feature-validation.md): one factor per run,
paired seeds, and the test split scored once with a fixed scorer.
