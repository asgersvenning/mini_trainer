# Production evaluation with mini_metrics

Run expert evaluation as soon as expert inference has completed:

```bash
cd /work/mini_trainer
git pull --ff-only
bash dev/ucloud/evaluate-results.sh expert /work/evaluation-1
```

After staged test inference completes, add its evaluation to the same root:

```bash
bash dev/ucloud/evaluate-results.sh test /work/evaluation-1
```

Alternatively use `all` once both predictions exist. Each dataset directory must
be fresh; use a new root for reruns. Defaults are:

- Expert: `/work/expert-full-2/predictions/mini_metric.csv`.
- Test: `/work/test-full-1/predictions/mini_metric.csv`.

Override these with `MT_EXPERT_CSV` and `MT_TEST_CSV` if inference used another
output directory. Only run against completed inference outputs; file existence
alone is not a completeness guarantee. Confirm the inference exit status and
staging manifest count (expert: 58,640; test: the saved test split count).

The script uses Python 3.13 through `uvx` and pins mini_metrics to revision
`70cc69adc05362863439277048e06386c1f885e1`. It leaves the training environment
unchanged and uses no GPU. Dependency versions are resolved by uv, not locked
by this helper. Internet access is required for initial tool installation.

Each dataset gets:

- `all_labels.csv` and `.json`: aggregate metrics including unseen labels.
- `known_labels.csv` and `.json`: the package's known-label-only metrics.
- `per_class.csv`: per-class metrics, including unseen labels.
- Logs for each invocation, input SHA-256, pinned revision, exact commands and
  a `COMPLETED` marker after all three invocations succeed.

The package supplies vocabulary coverage and micro/macro statistics. Preserve
both coverage and known-only results when assessing the external dataset; do
not describe known-only accuracy as accuracy on the entire benchmark. Compare
species, genus and family separately. The current package disables its separate
hierarchy-wide metrics; per-level metrics are still produced normally.

No optimal-threshold fitting, subsampling or label filtering is enabled. The
thresholds recorded in prediction files are retained. These collector files
contain the selected prediction per level, so they cannot establish top-5
accuracy. The expert dataset evaluates external performance; it should not be
used for selecting this model's thresholds or checkpoints.

Aggregate metric tables and progress are printed live in the terminal and retained
in the corresponding logs. `pipefail` preserves failures through `tee`.

For compact inspection after completion:

```bash
cat /work/evaluation-1/expert/all_labels.csv
cat /work/evaluation-1/expert/known_labels.csv
cat /work/evaluation-1/test/all_labels.csv
```

Keep per-class output as a file rather than pasting it into the terminal.
All outputs remain under `/work`; computation can also be rerun on a CPU job
from the saved prediction CSVs without retaining the source images or GPU node.

## Regional candidate vocabulary

Both `mt_predict` and `mt_hpredict` accept `--class-list FILE` (YAML key
`class_list`). Supply one exact model class label per UTF-8 line; for GBIF
hierarchical models these are species IDs. Blank lines and duplicates are
ignored. The list restricts the model's current candidate vocabulary and cannot
add species absent from the checkpoint. Missing requested labels are reported;
an empty overlap fails before loading images.

Every input image and ground-truth label remains in evaluation, including labels
excluded by the list. Species outputs and parent mappings are filtered together.
`class_filter.json` in the prediction output records retained, excluded and
missing labels, plus the list's SHA-256. With a pre-masked checkpoint, filtering
further restricts its active vocabulary rather than restoring excluded classes.

For example, in an environment containing this CLI feature:

```bash
mt_hpredict --config /work/expert-full-2/inference.yaml \
  --class-list /work/regional-evaluation/mambo-v2-reduced.txt \
  --output /work/regional-evaluation --name predictions
```

Use the all-label report for global versus regional comparisons on the same
images. `known_label` now describes the active filtered vocabulary. Confidence
scores are computed over the restricted candidates, so thresholds calibrated
for the global model should not be assumed equivalent.
