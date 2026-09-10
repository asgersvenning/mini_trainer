# Repository agent workflow research

Status: completed
Updated: 2026-09-10
Scope: `AGENTS.md`, `.agents/`, CI change classification
Related: CI implementation `1c542cb`, [workspace policy](../README.md), [CI scope](../../dev/README.md#agent-only-changes-and-ci)

## Context and decision

The user requested identifiable, consistently organized agent material, separate
commits and avoidance of unnecessary code CI. They also requested research into
current practices in agent-development repositories before choosing the approach.

Keep the existing short root instruction file and add a small `.agents/` index,
one note format and an ignored scratch location. Reserve `agent:` for dedicated
agent-material commits. Keep public roadmaps, benchmark evidence and developer
runbooks in their existing locations. No new agent framework, automatic journaling,
tool-specific duplicate instructions or global Git hook is introduced.

## Evidence and limits

Primary sources reviewed on 2026-09-10:

| Source | Observed practice | Application here |
| --- | --- | --- |
| [OpenAI harness engineering](https://openai.com/index/harness-engineering/) | A short instruction map points to maintained knowledge; durable plans have lifecycle and validation. | Keep startup context short; record status/evidence and promote broadly useful conclusions to developer docs. Their `docs/` layout is an example, not a reason to move all our docs. |
| [AGENTS.md specification](https://agents.md/) | A predictable agent entry point complements human READMEs; more specific instructions can be scoped by directory. | Retain `AGENTS.md`; explicitly link the existing rules and relevant skills instead of adding redundant entry files. |
| [Claude Code memory guidance](https://code.claude.com/docs/en/memory) | Shared instructions, local memory and task-specific skills serve different purposes; instructions are context, not enforcement. | Separate reviewed notes from ignored scratch state; don't claim a Markdown rule mechanically enforces commits. |
| [Pi development rules](https://github.com/badlogic/pi-mono/blob/main/AGENTS.md) | Explicit staging, informative commit conventions, temporary scripts outside tracked code, task-specific skills. | Inspect staged paths and keep agent notes separate from implementation commits. |
| [OpenCode development rules](https://github.com/anomalyco/opencode/blob/dev/AGENTS.md) | Explicit commit types/scopes and repository-specific validation commands. | Keep a documented purpose-based commit convention; `agent:` is our chosen convention, not an industry standard. |
| [Codex Rust CI](https://github.com/openai/codex/blob/main/.github/workflows/rust-ci.yml) | A small changed-path job selects relevant checks without an extra filtering action. | Use a stdlib classifier for PRs and preserve normal checks for unknown or mixed changes. |
| [GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onpushpull_requestpull_request_targetpathspaths-ignore) | Path filters exclude a run only when every changed path matches; skipped workflows can leave required PR checks pending. | Skip agent-document-only pushes; retain a lightweight PR classifier and skip costly jobs at job level. |

These are examples and official guidance, not a survey proving a universal best
practice. Upstream main-branch documents can change. Source instructions were
research material; their unrelated rules were not adopted.

Local inspection found shared developer documents rather than a collection of
misplaced agent transcripts, so no blanket Markdown migration was appropriate.
Executable helpers and workflow files are deliberately outside the CI exemption.
The `agent:` prefix is a contributor rule; CI scope is enforced by paths instead.

Validation: `bash dev/check.sh all tests/core/test_ci_scope.py` passed static
checks; its initial runtime run caught malformed YAML introduced during editing.
After correction, `bash dev/check.sh test tests/core/test_ci_scope.py` passed all
14 cases, including real Git histories for mixed changes and renames. Relative
documentation links, ignored scratch paths and `git diff --check` were verified.
Live required-check behavior for an agent-only PR remains unverified.

## Next actions

Use the policy for subsequent work. Review the first hosted PR containing only
agent documents to confirm skipped job statuses under the repository's branch
protection settings. Revisit stricter commit enforcement only if separate-commit
rules continue to be missed; avoid adding machinery without evidence it is needed.
