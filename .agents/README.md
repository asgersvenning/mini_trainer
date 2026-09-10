# Agent workspace

This directory contains material primarily for coding agents. Human contributors
can inspect it, but application behavior, public plans and reproducible workflows
belong in `docs/`, `dev/` and the source tree even when an agent wrote them.
The root [AGENTS.md](../AGENTS.md) is the instruction entry point and takes precedence.

## Locations and loading

| Location | Purpose | Read when |
| --- | --- | --- |
| [rules/](rules/) | Durable repository constraints | The task needs the rule |
| [skills/mini-trainer-maintenance/](skills/mini-trainer-maintenance/) | Repeatable maintenance procedure | Editing or validating the package |
| [notes/](notes/README.md) | Selected handoffs, decisions and research | A note matches the task |
| `local/` (ignored) | Scratch plans, logs, session state and temporary experiments | Only in the current local workflow |

Start with `AGENTS.md`, then follow relevant links. Do not load every note or skill
into every session. Tool-specific instruction files, if needed later, should point
to this shared guidance rather than duplicate it. Do not assume a tool discovers
arbitrary `.agents/rules/` files automatically.

Keep temporary work local. Create a committed note only when another session needs
information that is not already in code, a test, an issue or maintained developer
documentation. Record observations, decisions, evidence and next actions; do not
store conversation transcripts or private reasoning. Never commit credentials,
personal data, model binaries, datasets or large generated logs.

## Commit boundary

- Agent-only instructions and notes use **`agent: <specific summary>`** commits.
  The prefix describes the files' purpose, not whether an AI authored them.
- Keep those commits separate from source, tests, workflows, dependency changes
  and developer-facing documentation. Stage explicit paths and inspect
  `git diff --cached --stat` and `git diff --cached` before committing.
- Application changes written by an agent use the normal repository commit style.
  Do not relabel code as `agent:` or add `[skip ci]` to suppress validation.
- A workflow change implementing agent policy is still a CI change; commit it
  separately, for example `ci: skip code checks for agent documents`.
- Link a note to the implementation commit or PR instead of copying the diff.
  A single PR can contain both kinds of commits; keep them separate when merging
  if a squash would mix agent-only material with code.

These are contributor/agent rules, not a globally installed Git hook. CI decides
from file paths, independently of the message prefix. See
[CI scope](../dev/README.md#agent-only-changes-and-ci).

## Maintenance

Use the [note format](notes/README.md). Maintain one note per coherent topic, with
an explicit status and last verification date. When work finishes, close or
supersede the note and promote developer-relevant conclusions to their canonical
docs. Remove obsolete duplication; Git preserves history. No session-by-session
journal, parallel roadmap or automatic archive tree is required.

Existing developer roadmaps, benchmark evidence and execution plans remain where
they are. Being produced by an agent does not make them agent-only documentation.
