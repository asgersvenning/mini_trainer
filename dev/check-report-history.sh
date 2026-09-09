#!/usr/bin/env bash
# Prepare a compact artifact from a completed or failed target deployment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
results="${1:?Supply the target results directory}"
history="${2:?Supply the compact history directory}"
: "${BENCHMARK_PYTHON:?Set BENCHMARK_PYTHON to a prepared Python 3.12+ executable}"
: "${BENCHMARK_RUN_ID:?Set a unique run and attempt identifier}"
: "${BENCHMARK_REVISION:?Set the full source commit hash}"
: "${BENCHMARK_PROFILE:?Describe the dataset, model and target configuration}"
report="$results/evaluation/report.json"
if [[ ! -f "$report" ]]; then report="$results/status.json"; fi
args=(--report "$report" --history "$history" --run-id "$BENCHMARK_RUN_ID"
      --revision "$BENCHMARK_REVISION" --profile "$BENCHMARK_PROFILE"
      --note "${BENCHMARK_NOTE:-}")
if [[ -n "${BENCHMARK_RUN_URL:-}" ]]; then args+=(--run-url "$BENCHMARK_RUN_URL"); fi
case "${BENCHMARK_PERFORMANCE_VALID:-false}" in
    true) args+=(--performance-valid) ;;
    false) ;;
    *) echo 'BENCHMARK_PERFORMANCE_VALID must be true or false.' >&2; exit 2 ;;
esac
"$BENCHMARK_PYTHON" -m dev.benchmarks.report_history archive "${args[@]}"
"$BENCHMARK_PYTHON" -m dev.benchmarks.report_history render --history "$history"
