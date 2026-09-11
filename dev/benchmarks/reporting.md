# Continuous deployment reporting

The maintained path is target execution → compact history → optional release
storage and Pages. Local adapters are tested; target runners, authenticated storage
and public rendering still need live qualification.

## Opt-in target GPU workflow

```bash
export BENCHMARK_PYTHON=/path/to/prepared-tensorrt-env/bin/python
export BENCHMARK_METRICS_PYTHON=/path/to/metrics-env/bin/python
export TRT_BASELINE_MODEL=/path/to/float/model.onnx
export TRT_CANDIDATE_MODEL=/path/to/calibrated/model.onnx
export TRT_INFERENCE_MANIFEST=/path/to/heldout/manifest.json
export TRT_INPUTS=/path/to/heldout/batch-00000.npz
export TRT_PROFILES=/path/to/profiles.json
CUDA_VISIBLE_DEVICES=0 bash dev/check-tensorrt-deployment.sh fresh-results
```

The harness checks prepared runtimes, rebuilds both engines and composes
[inference evaluation](inference.md). It installs nothing. Keep external weights
beside ONNX models; source models must match the manifest's class/score contract.
Both builds allow FP16 and disable TF32, with optimization level 1 and 1 GiB
workspace by default. Override with `TRT_OPTIMIZATION` and `TRT_WORKSPACE_MIB`.

`.github/workflows/tensorrt-deployment.yml` exposes the same harness on a self-hosted
Linux GPU runner. Keep prepared environments/artifacts outside the checkout.

| Repository variable | Purpose |
| --- | --- |
| `TRT_BENCHMARK_PYTHON` | Maps to the harness's `BENCHMARK_PYTHON` |
| `BENCHMARK_METRICS_PYTHON` | Prepared metrics interpreter |
| `TRT_BASELINE_MODEL`, `TRT_CANDIDATE_MODEL` | Absolute source ONNX paths |
| `TRT_INFERENCE_MANIFEST`, `TRT_INPUTS`, `TRT_PROFILES` | Absolute input/profile paths |
| `TRT_RUNNER_LABEL`, `TRT_CUDA_VISIBLE_DEVICES` | Defaults `gpu`, `0` |
| `TRT_THREADS`, `TRT_OPTIMIZATION`, `TRT_WORKSPACE_MIB` | Optional resource/build settings |
| `ENABLE_TENSORRT_BENCHMARKS` | Exactly `true` enables scheduled execution |
| `TRT_REPORT_PROFILE`, `TRT_REPORT_NOTE` | Public workload/measurement description |
| `TRT_PERFORMANCE_VALID` | Exactly `true` opts into resource comparisons; default false |

Raw and compact artifacts include run ID/attempt and expire after 90 days.
Failures retain phase, exit code and available logs. Workflow concurrency does
not coordinate unrelated jobs: use an idle allocation for performance evidence.

## Compact report history and local dashboard

```bash
.venv/bin/python -m dev.benchmarks.reporting.report_history archive \
  --report fresh-results/evaluation/report.json --history local-history \
  --run-id run-123-attempt-1 --revision FULL_SOURCE_COMMIT_HASH \
  --profile 'Blair flat / target GPU / batch 8' --note 'Measurement conditions'
.venv/bin/python -m dev.benchmarks.reporting.report_history render --history local-history
```

Open `local-history/index.html`. CPU composed reports use the same command with
`--report cpu-comparison/report.json`. Keep all referenced child reports in place:
archival verifies hashes, metric deltas, identities and measurement settings.
New CPU reports enforce requested trial counts; older reports visibly retain an
unknown requested count. CPU medians/RSS and GPU paired timing/snapshots remain
distinct. Missing/undefined values and failed runs remain visible.

Run IDs are immutable: identical retries succeed, conflicting content fails.
`records/` is the durable input; HTML is regenerated. Records omit predictions,
local artifact paths and exception text but include explicitly supplied public
metadata. Review that metadata before hosting. `--performance-valid` is a publisher
declaration about measurement conditions, not automatic qualification; failed runs
never receive eligible resource comparisons.

For preflight/build failure, archive the nonzero root `status.json`. A successful
root status cannot replace detailed evaluation. The shared wrapper
`bash dev/check-report-history.sh RESULTS HISTORY` selects the proper source and
requires `BENCHMARK_RUN_ID`, `BENCHMARK_REVISION`, `BENCHMARK_PROFILE`; optional
variables are `BENCHMARK_RUN_URL`, `BENCHMARK_NOTE`, `BENCHMARK_PERFORMANCE_VALID`.
It uses `BENCHMARK_PYTHON` and needs only Python's standard library.

## Draft release storage client

```bash
.venv/bin/python -m dev.benchmarks.reporting.release_history \
  --repository OWNER/REPO --records local-history/records --output fresh-preview
```

Add `--upload` to append records with readback verification; omit `--records` to
restore only. Requires authenticated `gh` for github.com, with contents write
permission for uploads. Supply tokens through `GH_TOKEN`, never command arguments.
Use a fresh output directory on every attempt.

Records are stored in `benchmark-history-YYYY-MM` draft/prereleases by UTC creation
month. Published archives are readable but not appendable. The client validates
merged history before writing and never overwrites/deletes assets. Partial uploads
can be retried with identical inputs. Draft assets remain mutable by maintainers;
keep an independent backup. These tags do not match the PyPI workflow's `v*` trigger.

## Opt-in public history publisher

After review and merge to the default branch:

1. Configure the target runner and variables above.
2. Set GitHub Pages publishing to GitHub Actions and enable `ENABLE_BENCHMARK_HISTORY=true`.
3. Run a real success, failure and publisher-only retry. Verify stored asset readback,
   visible metrics/resource scopes, failure visibility and the public page.

The separate GitHub-hosted publisher downloads only the compact artifact by ID,
restores/appends monthly history and deploys HTML/JSON. It requires contents/pages
write and id-token permissions, and runs only on the default branch. Publishing
is serialized; retry artifacts include the attempt number. No live activation has
been performed. CPU-producing workflows and training-history adapters remain open;
see [the roadmap](../../docs/quantization-roadmap.md).
