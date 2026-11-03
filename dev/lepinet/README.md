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
| Validation | Per-level macro-F1 over classes present in the truth, each epoch | `macro_f1` |

Image paths are decoded with PIL, which matches fastai's decoding and resizing byte for byte.
Random crops cannot be cached. The other knobs follow the comparison rules in
[the training feature protocol](../../docs/training-feature-validation.md): one factor per run,
paired seeds, and the test split scored once with a fixed scorer.
