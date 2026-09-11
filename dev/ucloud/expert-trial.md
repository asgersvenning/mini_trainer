# Bounded expert inference staging trial

Stop the stalled full-test prediction and obsolete expert index creation with
Ctrl+C in their terminals first. Do not run this alongside additional cold-read
jobs. Saved training weights are unaffected; interrupted prediction currently
loses in-memory predictions because its collector writes at completion.

This helper targets the mounted expert folder and trained weights used in the
current qualification. Run from the node after pulling the committed helpers:

```bash
cd /work/mini_trainer
git pull --ff-only
bash dev/ucloud/expert-trial.sh
```

No manual YAML editing or dataset index construction is required. The launcher
extracts the focused discovery fix at `cba4ecd` into a fresh `/work` source overlay
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
loader memory. This assumes the current large-memory allocation. `/tmp` overlay
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
The full test split remains deferred until throughput is adequate.

RAM staging disappears with the job. Predictions, logs, configuration and manifest
are retained under `/work`. Staged files can be deleted once their corresponding
inference has exited; the helper deliberately does not delete existing directories.
