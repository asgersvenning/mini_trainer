#!/usr/bin/env bash
# Claude Code PreToolUse hook: stop implicit changes to the shared uv environment.
# Prefix a command with MINI_TRAINER_ALLOW_ENV_CHANGE=1 after explicitly choosing
# the PyTorch backend (README "Installation") and the target environment.
set -uo pipefail

if ! command -v jq >/dev/null; then
    echo 'block-env-sync: jq not found; environment guard skipped' >&2
    exit 0
fi
command="$(jq -r '.tool_input.command // empty')"
[[ "$command" == *MINI_TRAINER_ALLOW_ENV_CHANGE=1* ]] && exit 0

# Split chained commands so each segment is checked separately.
while IFS= read -r segment; do
    if [[ "$segment" =~ (^|[[:space:]/])uv[[:space:]]+(sync|add|remove)([[:space:]]|$) ]] ||
        [[ "$segment" =~ (^|[[:space:]/])uv[[:space:]]+pip[[:space:]]+(install|uninstall|sync)([[:space:]]|$) ]] ||
        [[ "$segment" =~ (^|[[:space:]/])pip3?[[:space:]]+(install|uninstall)([[:space:]]|$) ]] ||
        [[ "$segment" =~ -m[[:space:]]+pip[[:space:]]+(install|uninstall)([[:space:]]|$) ]]; then
        echo 'Blocked: this would change a Python environment. Never implicitly synchronize .venv;' \
            'see AGENTS.md "Environment and validation". If installation is intended, select the' \
            'PyTorch backend explicitly and prefix the command with MINI_TRAINER_ALLOW_ENV_CHANGE=1.' >&2
        exit 2
    fi
    if [[ "$segment" =~ (^|[[:space:]/])uv[[:space:]]+run([[:space:]]|$) && "$segment" != *--no-sync* ]]; then
        echo 'Blocked: plain `uv run` synchronizes .venv. Use `uv run --no-sync` or .venv/bin executables.' >&2
        exit 2
    fi
done < <(printf '%s\n' "$command" | sed -E 's/(&&|\|\||;|\|)/\n/g')
exit 0
