# Experiments

## Configuration

To configure and orchestrate an experiment with SLURM use the following recipe:

1) Create a configuration YAML file (**`<CONFIG_FILE>`**) following the template [config.template.yaml](./config.template.yaml).
   - `name`: Set the experiment name (**`<NAME>`**), also used for the SLURM job name.
   - `slurm`: Configure SBATCH arguments.
   - `experiment`: Configure experiment matrix parameters for `mini_trainer`.
   - `eval`: Configure evaluation datasets for specific experiment parameters.
   - `args`: Configure fixed (shared) `mini_trainer` parameters used for all experiment matrix parameter combinations.
2) Create the SLURM array script for running the experiment matrix: `uv run orchestrate.py <CONFIG_FILE>`.
   *(Validate the correct construction of the experiment matrix in the file `slurm_jobs/<NAME>/tasks.txt`).*
3) Run the SLURM array script `sbatch slurm_jobs/<NAME>/array.sh`.

## Notes

Loose notes for the experiment configuration and matrix.

## Template train command

```sh
mt_htrain -i <INPUT> \
    -o <OUTPUT> \
    --model <MODEL> \
    --head <HEAD> \
    --dtype float16 \
    --batch_size 256 \
    --epochs <EPOCHS> \
    --warmup_epochs 0.1 \
    --class_weighted \
    --loss_weights <W0> <W1> [...] \
    --wandb
```

## Experiment matrix

- Datasets (2)
  - global_lepi
  - plantnet300k

- Models (6)
  - efficientnet_v2_[s/m/l]
  - ViT_L_16
  - ViT_H_14
  - BioClip2 (finetune only and/or zero-shot)

- Heads (6)
  - Flat
  - Bottom-up
  - Top-down
  - Independent
  - Autoregressive (independent)
  - Autoregressive (geometrically nested)

### Table templates

#### Global Lepidoptera

| Model | Head |
|-------|------|
| ...   | ...  |

#### Flemming eval

OOD test using model trained on Global Lepidoptera

| Model | Head |
|-------|------|
| ...   | ...  |

#### Pl@ntNet300K

| Model | Head |
|-------|------|
| ...   | ...  |
