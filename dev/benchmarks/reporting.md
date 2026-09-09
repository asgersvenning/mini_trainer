# Continuous deployment reporting

### Opt-in target GPU workflow

`.github/workflows/tensorrt-deployment.yml` runs the same target command as a
local or manually allocated Linux GPU session:

```bash
export BENCHMARK_PYTHON=/absolute/path/to/prepared-tensorrt-env/bin/python
export BENCHMARK_METRICS_PYTHON=/absolute/path/to/metrics-env/bin/python
export TRT_BASELINE_MODEL=/absolute/path/to/float/model.onnx
export TRT_CANDIDATE_MODEL=/absolute/path/to/calibrated/model.onnx
export TRT_INFERENCE_MANIFEST=/absolute/path/to/heldout/manifest.json
export TRT_INPUTS=/absolute/path/to/heldout/batch-00000.npz
export TRT_PROFILES=/absolute/path/to/profiles.json
CUDA_VISIBLE_DEVICES=0 bash dev/check-tensorrt-deployment.sh tensorrt-results
```

Keep ONNX external-weight files beside each model. Both models must implement
the manifest's class order and score semantics. Supply a calibrated candidate;
this command does not select calibration data or fit a quantizer. It checks
runtime availability, rebuilds both engines on the actual runner, then invokes
the composed evaluator. Both builds allow FP16, disable TF32, share the profile
JSON, and default to optimization level 1 and 1 GiB workspace. Override those last
two settings with `TRT_OPTIMIZATION` and `TRT_WORKSPACE_MIB`. Resource threads
default to one (`BENCHMARK_THREADS`) and visible device index to zero
(`BENCHMARK_DEVICE`). No environment installation, synchronization or deletion
occurs. Relative paths resolve from the repository root.

The GitHub workflow requires a configured self-hosted Linux GPU runner and these
repository variables:

| Repository variable | Meaning |
| --- | --- |
| `TRT_BENCHMARK_PYTHON` | Prepared TensorRT/CUDA Python; maps to `BENCHMARK_PYTHON` |
| `BENCHMARK_METRICS_PYTHON` | Prepared `mini_metrics` Python |
| `TRT_BASELINE_MODEL`, `TRT_CANDIDATE_MODEL` | Absolute source ONNX paths |
| `TRT_INFERENCE_MANIFEST`, `TRT_INPUTS`, `TRT_PROFILES` | Absolute held-out contract, resource inputs and build profile paths |
| `TRT_RUNNER_LABEL` | Additional self-hosted runner label; defaults to `gpu` |
| `TRT_CUDA_VISIBLE_DEVICES` | Explicit visible GPU selection; defaults to `0` |
| `TRT_THREADS`, `TRT_OPTIMIZATION`, `TRT_WORKSPACE_MIB` | Optional matching execution/build settings |
| `ENABLE_TENSORRT_BENCHMARKS` | Set exactly `true` to enable weekly scheduled execution |

Prepared environments and source artifacts should live outside the checkout,
which the checkout action can clean between runs. Use quiescent, allocated target
hardware; the workflow's concurrency group prevents overlapping executions of
this workflow with the same runner label, but does not coordinate unrelated jobs.
Manual dispatch is available independently of the schedule opt-in. GitHub requires
the workflow on the default branch for manual/scheduled activation; scheduled
start times can be delayed. See [GitHub's event documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

Every run retains the repository revision, harness hash, preflight/build/evaluation
logs, failing phase/exit code, rebuilt engines and inspections, predictions,
metrics, paired timings and memory reports. A completed comparison appears in
the job summary. Failures instead show their status and point to retained logs.
Artifacts are named by run ID and attempt and retained for 90 days. This supplies
visible per-run evidence, **not durable cross-run history**; permanent publication
and target acceptance still require additional work. Source models, datasets and
prepared environments are not uploaded by the workflow.

### Compact report history and local dashboard

`dev.benchmarks.report_history` archives version-one composed TensorRT and ONNX CPU reports as
immutable, compact JSON records and renders a standalone HTML history. Use the
repository's supported Python (3.12 or newer); the command itself only needs the
standard library and does not load PyTorch, TensorRT or datasets.

```bash
.venv/bin/python -m dev.benchmarks.report_history archive \
  --report tensorrt-results/evaluation/report.json --history tmp-report-history \
  --run-id run-123-attempt-1 --revision FULL_SOURCE_COMMIT_HASH \
  --profile 'Blair flat / target GPU / batch 8' \
  --run-url https://github.com/OWNER/REPO/actions/runs/123 \
  --note 'Describe workload and measurement conditions'
.venv/bin/python -m dev.benchmarks.report_history render --history tmp-report-history
```

Open `tmp-report-history/index.html`. Every record retains the source report
hash, source revision, input/manifest hashes, matching build settings, aggregate
quality values/deltas, engine identities and resource snapshots. Runtime identity
comes from a retained latency report whose hash must match the evaluation report.
Keep that child report beside the source evaluation report when archiving.
Prediction rows, class/sample labels, local artifact paths and exception text are
not copied into the compact record. Profile/note/run URL are explicitly supplied
publication metadata; review them before hosting.

Run IDs accept a restricted filename-safe alphabet. Repeating the identical run
is idempotent; changing its evidence or metadata under the same ID fails without
overwriting the record. New records are published with a no-overwrite filesystem
operation. The HTML page is derived and can be regenerated from `records/`.
Retain that directory in persistent storage; creating these files alone is not a
backup or permanent hosting service. Original reproduction artifacts remain
necessary; compact records do not replace engine files, inputs or raw reports.

Resource comparisons are **excluded by default**. Supply `--performance-valid`
only when the run's conditions justify their use, and describe those conditions
in `--note`. This is a publisher declaration, not an automatic certification.
Failed runs never display eligible performance comparisons. For a preflight/build
failure before an evaluation report exists, pass the shared command's nonzero
`status.json` instead. A successful top-level status alone is insufficient:
archive its detailed evaluation report. Missing quality and undefined metrics
remain visible, and completed evaluation is not displayed as production acceptance.

The renderer supports TensorRT and ONNX CPU deployment records. Training
history adapters remain unfinished. No remote uploads occur from either command. Local HTML structure
and escaping have tests; browser rendering still needs visual qualification.

The target workflow now also calls `bash dev/check-report-history.sh RESULTS HISTORY`
and uploads a separate `tensorrt-history-RUN-ATTEMPT` artifact containing only the
compact record and standalone page. This runs after failed evaluations too; an
invalid evaluation report fails archival instead of being replaced with a less
detailed status. A missing report uses the target command's failure status.
The reporting command uses the prepared `BENCHMARK_PYTHON` and needs no GPU imports.
If that executable is missing, raw failure artifacts remain the diagnostic source.

Set repository variables `TRT_REPORT_PROFILE` and `TRT_REPORT_NOTE` to describe
the workload and measurement conditions for readers. Set `TRT_PERFORMANCE_VALID`
to exactly `true` only for a runner whose conditions justify performance comparisons;
the default is `false`. Failed evaluations remain excluded regardless of this setting.
Run IDs include the workflow attempt, and source revision/run links come from Actions.
Locally, supply `BENCHMARK_RUN_ID`, `BENCHMARK_REVISION`, `BENCHMARK_PROFILE`, and
optionally `BENCHMARK_RUN_URL`, `BENCHMARK_NOTE`, `BENCHMARK_PERFORMANCE_VALID`.
These artifacts still expire after 90 days. The optional publisher below stores
compact records separately and deploys the historical page.

### CPU deployment history

CPU reports use the same `report_history archive` command, with `--report` pointing
to the composed `cpu_deployment` report. Retain its `quality/report.json`,
`placement-ROLE/report.json`, and `trial-N-ROLE/report.json` children: the adapter
checks their recorded hashes, quality deltas, operator-placement summaries,
model identities, timing inputs, runtime/settings and resource ratios. The compact
record contains actual ONNX Runtime provider/operation counts, model-file hashes
and sizes, architecture, affinity, thread settings, and the five metrics.

CPU rows explicitly compare **separate-process latency medians** and show process
RSS and approximate peak RSS. They do not acquire GPU memory fields or adjacent
GPU timing semantics. The resource declaration remains opt-in. New CPU reports
retain requested trials, threads, warmups and repeats even after a failed stage.
The archive verifies requested settings against the measured child reports and
requires the requested number of trials for a completed evaluation. Failed runs
can retain fewer completed trials. Older CPU reports without requested settings
remain readable, and the page explicitly says their requested count was not
retained; the archive does not infer it from completed pairs. A completed report
is still not acceptance.
Failed CPU reports remain visible and never display eligible resource comparisons.

The adapter was exercised on retained flat/hierarchical Blair x86 reports with
three trials each. Synthetic `aarch64` metadata in tests checks display handling,
not ARM execution. CPU records can share local history and draft storage with
TensorRT records; the current automated producing workflow still runs TensorRT
only. An ARM runner and its benchmark-to-publisher handoff remain to be qualified.

### Draft release storage client

`dev.benchmarks.release_history` can restore compact history from GitHub release
assets and optionally append incoming records. It requires Python 3.12+ and
GitHub CLI `gh`, authenticated for the explicit repository. For automation, set
`GH_TOKEN` in the environment with repository contents write permission; never
include the token in command arguments or reports. Only github.com is supported.

```bash
# Restore existing remote history and preview the incoming records locally.
.venv/bin/python -m dev.benchmarks.release_history \
  --repository OWNER/REPO --records tensorrt-results/history/records \
  --output tmp-history-preview

# Explicitly store the incoming records, then verify each upload by readback.
# Use a fresh output directory for every invocation, including retries.
.venv/bin/python -m dev.benchmarks.release_history \
  --repository OWNER/REPO --records tensorrt-results/history/records \
  --output tmp-history-stored --upload
```

Omit `--records` for restore only. Each record belongs to its UTC creation month
under `benchmark-history-YYYY-MM`. New storage releases are drafts/prereleases
and are never marked latest. This namespace does not match this repository's
PyPI workflow trigger, `v*`. Keep an active month's release in draft: the client
can restore published archives, but refuses to append to them. Draft assets are
maintainer storage, not a public dashboard; publish the restored HTML separately.

The client paginates both releases and assets, validates and renders all merged
records before remote writes, rejects conflicting bytes, and never replaces or
deletes assets. An interrupted upload may leave partial remote progress; retry
with the same input records and a fresh output directory. Authentication errors
fail instead of being interpreted as an empty archive. Monthly release limits
and API failures remain visible errors, with existing records retained.

This avoids Actions artifact expiration when explicitly used, but maintainers can
still alter/delete draft assets. Keep an independent backup for stronger retention.
Tests simulate the GitHub CLI boundary, pagination, conflicts, interrupted uploads,
readback corruption and authentication failures. No live release upload has yet
been verified; the optional workflow integration below still needs remote qualification.
See the [GitHub CLI API options](https://cli.github.com/manual/gh_api) and
[release API](https://docs.github.com/en/rest/releases/releases) for the underlying
pagination and draft-release contract.

### Opt-in public history publisher

The TensorRT workflow includes a separate GitHub-hosted `publish-history` job.
It downloads the GPU job's compact artifact by its output ID, restores all monthly
records, appends new records with readback verification, and deploys the regenerated
HTML and compact JSON to GitHub Pages. It does not download engines or predictions.
The producing artifact ID survives a publisher-only retry; the new Pages artifact
name includes the current attempt to avoid colliding with a previous upload.

To activate after the workflow is merged to the default branch:

1. Configure the target GPU job using the environment/data instructions above.
2. Enable GitHub Pages with **GitHub Actions** as its publishing source and configure
   the `github-pages` environment to permit the default branch. This workflow owns
   the repository's Pages site, so incorporate any existing site before enabling it.
3. Review `TRT_REPORT_PROFILE`, `TRT_REPORT_NOTE` and `TRT_PERFORMANCE_VALID` as public
   metadata; set repository variable `ENABLE_BENCHMARK_HISTORY=true`.
4. Dispatch TensorRT deployment on the default branch. Inspect the stored draft
   assets, compare their bytes to the compact artifact, and open the deployment URL
   shown on the `github-pages` environment. Confirm failed records and excluded
   resource comparisons remain visible before relying on continuous publication.

Feature-branch runs never publish. The publisher uses its job-scoped GitHub token
with contents write, Pages write and OIDC permissions; the self-hosted GPU job
retains read-only repository permissions. Publication is serialized and is eligible
even when evaluation fails, provided a compact artifact was successfully uploaded.
If archival or rendering fails, deployment does not proceed. If only Pages fails,
stored records remain available and the publisher job can be rerun. Recover an
expired or otherwise unavailable producing artifact from retained local records
using the storage client; a new benchmark is not required to restore old evidence.

This is implemented workflow wiring, not a verified live service. The feature
branch has not enabled repository settings or published a site. Local validation
covers YAML/Bash syntax and storage/report behavior; the first remote run must
qualify GitHub authentication, release assets, Pages deployment and browser display.
The setup follows [GitHub's custom Pages workflow requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
