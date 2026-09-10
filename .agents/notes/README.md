# Agent notes

Use `YYYY-MM-DD-short-topic.md` (creation date) for a durable handoff or focused
research note. Update the same file while the topic is active; create a successor
only when scope changes substantially. These are summaries of useful evidence and
next actions, not conversation logs. Do not create notes for routine edits.

Use this structure, omitting empty sections:

```markdown
# Topic

Status: active | completed | superseded
Updated: YYYY-MM-DD
Scope: relevant paths or subsystem
Related: implementation commit, issue, or canonical document

## Context and decision
What was requested, what was chosen, and why.

## Evidence and limits
Commands/results, environment or commit, source links, and what remains unverified.
Distinguish observed behavior from a hypothesis or recommendation.

## Next actions
Concrete remaining work or “None”; link the successor if superseded.
```

Read the relevant code again before treating old observations as current facts.
Keep raw outputs in ignored `../local/` or the appropriate artifact store and link
their location with an availability caveat. Commit notes separately using `agent:`.

Current durable references:

- [Repository agent workflow research](2026-09-10-agent-workflow.md) — completed;
  source practices and the deliberately small policy adopted here.
