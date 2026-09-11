# Repository agent-material migration

Status: completed
Updated: 2026-09-10
Scope: tracked documentation and agent instructions, audited at `e8302d7`
Related: [workspace policy](../README.md), [cleanup handoff](2026-09-09-quantization-cleanup.md)

## Context and decision

The user requested an audit and staged migration to the new agent-material rules.
The starting checkout was clean. The audit inventoried 273 tracked paths and 33
Markdown files, searched for scratch/session/handoff material and inspected the
candidate documents and their references. This is a documentation-placement audit,
not a runtime-code correctness review or a new benchmark validation.

| Material | Classification and migration |
| --- | --- |
| `AGENTS.md`, `.agents/README.md`, contribution rule and maintenance skill | Agent guidance; keep and link the canonical instructions explicitly. |
| Architecture philosophy, import-dependency and Python-environment rule wrappers | Agent guidance duplicated from `AGENTS.md`, `README.md` and `dev/README.md`; remove the three wrappers, keeping their authoritative sources. |
| `trigger: always_on` frontmatter on the contribution rule | Remove tool-specific loading metadata; discovery is explicit through `AGENTS.md`. |
| Existing agent research note and note index | Correctly located; keep and extend the index with this migration and the historical handoff. |
| `docs/quantization-artifacts.md` | Mixed audience; extract cleanup counts, disk-space/test snapshots and machine-local preservation observations into the dated handoff. Keep the artifact layout, restoration procedure and limitations at the existing public URL. |
| `dev/benchmarks/README.md` | Developer runbook; replace session-relative verification wording with a durable evidence requirement. |
| `docs/quantized-training-validation.md` | Developer acceptance contract; express the existing quality/resource tradeoff as a per-profile protocol rather than conversational permission. |
| Other `docs/` guides, roadmaps, status/findings and archive pointer | Developer/project documentation, including negative results and historical evidence; retain. Agent authorship is not a reason to relocate them. |
| `README.md`, `ddp/README.md`, `dev/README.md`, benchmark/UCloud guides, `tests/README.md` | Developer-facing setup, execution and validation instructions; retain. |
| `publication/experiments/README.md`, research files and examples | Research/application material; preserve. |
| Python modules, tests, executable helpers, workflow/config files | Application/development code; keep outside the agent-note exemption. No changes needed for this migration. |
| Ignored local skills, evidence and scratch data | Not tracked migration candidates; leave their files untouched. |

## Evidence and limits

The dated cleanup report is preserved with its source commit and an explicit
historical-only caveat. Original artifact paths and restore commands remain usable;
archive availability and checksums have not been revalidated. Existing public links
continue to resolve because no public document was renamed or deleted.

Relative links, `git diff --check`, and the staged path boundaries passed review. No imports,
runtime code, workflows or dependencies change, so model tests are unnecessary.
Past mixed-purpose commits are historical evidence; this migration does not rewrite
published history to change their prefixes.

## Next actions

None for this migration. The user reviewed and approved both batches, committed
separately as:

- `8d178f8` — agent guidance consolidation and historical cleanup handoff.
- `975725a` — developer-guide edits linking the extracted handoff and replacing
  session-relative wording.

The agent batch preceded the public link updates. Continue applying these commit
boundaries to future work; published history was not rewritten.
