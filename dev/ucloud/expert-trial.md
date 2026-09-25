# Bounded expert inference staging trial

Historical helper for the September training campaign's expert images and weights.
For current MAMBO release comparisons use [the release runbook](../releases/mambo_v3/ucloud-release.md).
The [training post-mortem](../../docs/training-workflow-postmortem.md) records the
completed staging/inference work and lessons.

Run from the reviewed checkout on the allocated node, without competing cold-read
jobs. This older collector retains predictions in memory until completion, so
interruption loses unfinished prediction output.

```bash
cd /work/mini_trainer
bash dev/ucloud/expert-trial.sh
```

No manual YAML editing or dataset index construction is required. The launcher
extracts the focused discovery fix at `0c572ca` into a fresh `/work` source overlay
and uses the existing `/work/venvs/mt-quant` interpreter and dependencies. It does
not install packages or update the environment of another process. The underlying
Python helper also accepts explicit source, weights and output paths (`--help`).

Defaults:

- GPU 1; 1,024 JPEG/PNG candidates selected round-robin across class folders.
- At most 2 GiB of encoded source images, copied with four readers to
  `/dev/shm/mt-expert-expert-staging-trial-1`.
- Five-minute staging limit and ten-minute inference limit. Timeout/interrupt
  stops the phase process group, including inference loader workers.
- Normal prediction defaults: model/preprocessing metadata from the weights,
  no separately supplied class mapping, resize recipe or fabricated labels.
- Persistent output: `/work/expert-staging-trial-1`. The minimal `inference.yaml`
  contains only input and weights; output/name are explicit CLI arguments.

`/dev/shm` is RAM-backed. The byte cap limits copied images, not total process or
loader memory. This assumes the original large-memory allocation. `/tmp` overlay
storage has not been established as node-local and is not used for this trial.

Watch either phase in another terminal:

```bash
tail -f /work/expert-staging-trial-1/stage.log
# Once inference starts:
tail -f /work/expert-staging-trial-1/inference.log
```

The launcher prints phase durations. On success, predictions are in
`/work/expert-staging-trial-1/predictions/mini_metric.csv`; `staging.json` records
original and staged paths, bytes and staging duration. Originals are unchanged.
Newly encountered hierarchical labels may still need GBIF/cache resolution; the
existing policy fails explicitly if it cannot resolve ancestors. Removing index
creation does not remove that lookup.

If either phase times out, inspect its log before retrying. Partial staging is
retained without a completed manifest and is never treated as ready input. A retry
uses a fresh output/staging name, for example:

```bash
bash dev/ucloud/expert-trial.sh /work/expert-staging-trial-2 --images 256
```

Proceed to larger staging only if this trial completes and staged inference is
fast. Measure the full expert dataset's encoded size and available job memory
before raising the cap. This round-robin subset is a storage/functionality trial,
not the expert benchmark; do not report its accuracy as a full-dataset result.
Retain the full benchmark's unknown species when subsequently running mini_metrics.
Full test staging uses [its separate helper](test-inference.md).

RAM staging disappears with the job. Predictions, logs, configuration and manifest
are retained under `/work`. Staged files can be deleted once their corresponding
inference has exited; the helper deliberately does not delete existing directories.

To retry inference after a code fix, reuse completed staging without another read
of the source images:

```bash
bash dev/ucloud/expert-trial.sh /work/expert-staging-trial-4 \
  --reuse-stage /work/expert-staging-trial-3
```

This checks the previous completed manifest and staged file sizes. The source
paths remain recorded, and the new output receives its own configuration and logs.
Taxonomy still uses the node's GBIF response cache; cache/API failures are
separate from filesystem staging failures.
