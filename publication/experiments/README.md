# Research experiments

## Generate a SLURM matrix

From the repository root, use the [installed environment](../../README.md#local-installation)
and copy [config.template.yaml](config.template.yaml) to a campaign configuration.

| Configuration | Role |
| --- | --- |
| `name`, `output_dir` | Campaign name and shared output base; defaults to `slurm_jobs/<name>`. |
| `stubs`, `slurm` | Installed training/prediction/metric commands and SBATCH settings. |
| `datasets`, `eval` | Dataset paths/indexes and training-to-evaluation dataset mappings. |
| `experiment` | Cartesian product of model, head, dataset and other axes. |
| `args` | `shared`, `train`, `eval` and `metrics` options; dictionaries select values by matrix axis. |

```sh
.venv/bin/python -m publication.experiments.orchestrate campaign.yaml
```

Inspect `train_tasks.txt`, `eval_tasks.txt`, `metric_tasks.txt` and `array.sh` in
`<output_dir>/<name>/` before submitting `sbatch <output_dir>/<name>/array.sh`.
Each array task runs training, prediction from `weights/last.pt`, then metrics;
a failed command stops that task. Results go under the campaign's `results/`.
Generation can resolve taxonomy while constructing evaluation combinations.

The generated script assumes commands and dataset/output paths are available on
the compute node; it does not install or activate an environment. Use explicit
indexes for external evaluation datasets. Without one, evaluation only supports
the training dataset and reuses its generated `data_index.json`.

## Research scope

Proposed matrix: Global Lepidoptera and Pl@ntNet300K; EfficientNetV2 S/M/L,
ViT-L/16, ViT-H/14 and BioCLIP2 (fine-tuned or zero-shot); flat, bottom-up,
top-down, independent and autoregressive heads (independent or geometrically
nested). Flemming supplies out-of-domain evaluation for Global Lepidoptera.
These are planned comparisons, not recorded results.

The separate [prototype-coordinate study](prototype_linearization/README.md)
contains its own reproduction workflow and evidence.
