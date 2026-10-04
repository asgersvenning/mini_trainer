# Research experiments

The [flat training ablation study](training_ablations/README.md) provides a bounded
UCloud campaign for normalized heads, prototype regularization, EMLA, projection
and optimizer comparisons. It uses manual allocation with an optional API pilot;
the SLURM generator below is a separate workflow.

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
The larger matrix remains proposed. The Gefion archive contains the completed
subset reviewed below; do not treat unexecuted cells as results.

The separate [prototype-coordinate study](prototype_linearization/README.md)
contains its own reproduction workflow and evidence.

[boot_metrics.py](statistics/boot_metrics.py) repeats seeded `mini_metrics`
threshold calibration/evaluation and writes metrics by seed and rank. Its legacy
name does not imply bootstrap resampling: sampling is delegated to the installed
`mini_metrics`. Record that dependency revision when retaining results.

## Consolidating main experiments and ablations

The October 2026 audit of the Gefion `experiments/` archive found 24 runs with
`last.pt` and ten logged epochs: EfficientNet-B0, EfficientNetV2-S and ViT-L/16,
each with flat, hierarchical, conditional and independent heads on corrected
PlantNet and Global Lepidoptera. There are 36 primary prediction/evaluation sets,
including Flemming transfer evaluation; four additional ViT PlantNet exports have
`_1` suffixes and must not be counted as independent replicates. Autoregressive
runs lack final checkpoints and are excluded. File presence and epoch counts are
completion evidence, not proof of matching training provenance.

For PlantNet, all twelve primary prediction CSVs were checked against every image
and species label in the corrected index's test partition: 31,088 images and 987
species. Recomputed **unthresholded species macro recall (%)** is:

| Backbone | Flat | Hierarchical | Conditional | Independent |
| --- | ---: | ---: | ---: | ---: |
| EfficientNet-B0 | 46.45 | 44.98 | 35.97 | 46.06 |
| EfficientNetV2-S | 53.70 | 53.06 | 42.61 | 53.68 |
| ViT-L/16 | 42.41 | 42.83 | 32.71 | 43.35 |

These descriptive results do not establish a universal hierarchy advantage.
Hierarchical minus flat micro recall is also negative for all three backbones
(-2.93, -2.38 and -2.29 percentage points, respectively). Saved threshold-optimized
`mini_metric_metrics.csv` values answer a different question and can change the
apparent ordering. Recompute a common metric before comparing architectures.

Reproduce this table and the per-class/confusion inputs with:

```sh
python -m publication.experiments.statistics.saved_predictions MAIN_ARCHIVE MAIN_REPORT \
  --evaluation plantnet --path-marker images_gbif/ \
  --data-index MAIN_ARCHIVE/plantnet/data_index.json --split test
```

The command requires the original campaign-relative layout, rejects duplicate
samples, thresholded inputs and split/label mismatches, and records prediction
and index hashes. It needs only the Python standard library. Preserve the original
CSV files, including their original path strings; normalization is an explicit
analysis option. The shared sample-label identity hash is
`8140b44607a4dce355c94a35c7ccedb712f7289c08f02645fcf5b58aaa309af9`.

### Which evidence can be combined?

| Evidence | Role | Boundary |
| --- | --- | --- |
| Historical head/backbone comparisons | Architecture and backbone dependence; Global Lepidoptera to Flemming transfer | Within-backbone/dataset comparisons; audit split and seed provenance before inference about uncertainty |
| Lepidoptera factorial and duration studies | N × R × adjustment mechanisms and schedule dependence | Preserve each cohort, seed and schedule; do not pool image-level records from different cohorts |
| Complete-family hierarchy study | Hierarchy × regularization under preserved branch imbalance | Paired controls within this cohort; reduced-vocabulary screening is a separate study |
| Corrected PlantNet targeted replication | Cross-dataset replication of selected interactions and EMLA versus fixed dynamics | New validation results are not directly comparable to historical test results |

The historical PlantNet heads supervise **five ranks** (987/325/111/46/5 classes),
whereas the new hierarchy ablation uses species/genus/family. Saved V2-S settings
also differ: batch 64 versus 512, LR .001 versus .003, weight decay .01 versus .001,
and warmup .25 versus one epoch. Retain these recipe differences in the evidence
catalog; matching species vocabulary does not make the campaigns interchangeable.
A current Gefion checkout revision cannot retrospectively identify training code;
recover historical versions from run metadata/W&B where possible and mark unknowns.
Do not treat backbones as replicated random seeds.

Next analyses should use existing artifacts first: align sample IDs, recompute
unthresholded class-frequency/confusion effects and hierarchical errors, then
report paired contrasts within each experiment and consistency across experiments.
Use observation-group resampling where observations contain multiple images;
image-level uncertainty cannot replace training-seed uncertainty. Keep threshold
calibration disjoint from final test evaluation and record its seed, objective and
`mini_metrics` revision. No new training is justified merely to harmonize reports.

Historical prediction CSVs contain top predictions/confidence at each rank, not
full probability vectors. They suffice for recall, confusion and top-confidence
calibration; they cannot reconstruct NLL, Brier scores, soft prediction mass or
parent probabilities aggregated from leaf probabilities. If those comparisons
remain necessary, run inference from retained checkpoints on frozen splits and
archive raw scores. Likewise, old epoch summaries cannot reconstruct unlogged
sub-epoch gates. Neither gap requires retraining merely to restate an endpoint.

## Reproducing analyses from ERDA artifacts

Follow the useful separation in
[flat-bug's analysis downloads](https://github.com/darsa-group/flat-bug/blob/main/scripts/manuscript/statistics/helpers/flatbug_download_results.R):
readers download evaluation inputs and precomputed statistics; training is a
separate, optional workflow. Use immutable versioned ERDA folders with a public
read-only download link. Keep upload access separate from reader access. ERDA
publication is pending configuration of the destination and access; no public
mini_trainer archive link has been established yet.

Retain three explicit artifact sets, with a manifest per set:

| Set | Contents | What readers can reproduce |
| --- | --- | --- |
| Curves and tables | Configs, class maps/counts, source/split hashes, plan and completion manifests, epoch confusion matrices, scalar/gate logs, analysis tables and figure inputs | Learning dynamics and reported aggregate/interaction figures on CPU |
| Prediction evidence | Above plus sample/split/taxonomy metadata and per-image targets/scores; historical `mini_metric.csv` and combinations | Recompute per-class errors and metrics without images or training; probability metrics only where full scores exist |
| Optional geometry | Checkpoints, fixed embedding selection and extracted embedding summaries/arrays, preprocessing and extraction provenance | Prototype geometry on CPU; new image embeddings still require images and inference |

Include the original corrected taxonomy mapping/converter and dependency lock,
training revision when known, analysis revision, seeds and exact reproduction
commands. Preserve failed/excluded runs in the catalog without counting them as
completed results. Freeze only completed artifacts; do not archive a live file
while training writes it. Raw images and authentication material are not part of
these analysis bundles. Full model checkpoints are optional for prediction-only
analysis, and pretrained downloads are unnecessary.

[artifacts.py](artifacts.py) creates and verifies SHA-256/size manifests and fetches
only missing or corrupt files from an HTTPS archive. An explicit file list avoids
accidentally publishing unrelated files. Publish the reviewed manifest with the
analysis release so readers have a trusted reference independent of the download:

```sh
python -m publication.experiments.artifacts create SNAPSHOT manifest.json \
  --files artifact-paths.txt --revision ANALYSIS_COMMIT
python -m publication.experiments.artifacts verify manifest.json SNAPSHOT
# Upload listed files under one immutable ERDA folder, preserving relative paths.
python -m publication.experiments.artifacts fetch manifest.json CACHE \
  --base-url READ_ONLY_ERDA_URL
python -m publication.experiments.artifacts verify manifest.json CACHE
```

Use `training_ablations.dynamics` for the curves bundle and
`training_ablations.analysis` without `--geometry` for prediction evidence; their
commands are documented in the [ablation workflow](training_ablations/README.md).
The latter tools use the repository's pinned CPU environment. Geometry additionally
needs retained checkpoints; do not require readers to download them for ordinary
metric reproduction. Reproduce figures from regenerated tables and compare numeric
tables with recorded tolerances, rather than demanding byte-identical image files.
