#!/usr/bin/env bash
# Shared local and CI checks; deliberately never installs or synchronizes packages.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

usage() {
    echo 'Usage: bash dev/check.sh [static|test|all] [pytest arguments...]'
}

mode="${1:-static}"
if (( $# > 0 )); then
    shift
fi
case "$mode" in
    -h|--help) usage; exit 0 ;;
    static)
        if (( $# > 0 )); then
            usage >&2
            exit 2
        fi
        ;;
    test|all) ;;
    *) usage >&2; exit 2 ;;
esac

if [[ ! -x .venv/bin/python ]]; then
    echo 'Missing .venv/bin/python. Follow the README local installation instructions first.' >&2
    exit 1
fi

if [[ "$mode" == static || "$mode" == all ]]; then
    .venv/bin/python -m ruff check mini_trainer tests dev
    .venv/bin/python -m ruff format --check --diff mini_trainer tests dev
    .venv/bin/lint-imports
fi

if [[ "$mode" == test || "$mode" == all ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    .venv/bin/python -m pytest "$@"
fi
