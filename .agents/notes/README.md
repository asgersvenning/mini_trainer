# Agent handoffs

No active tracked handoffs. Completed migration, policy-research and quantization
cleanup notes were retired after their useful content was consolidated into
[workspace policy](../README.md), [CI guidance](../../dev/README.md#agent-only-changes-and-ci)
and [artifact retention](../../docs/quantization-artifacts.md). Git retains history.

Create `YYYY-MM-DD-topic.md` only for a cross-session handoff that cannot live in an
existing maintained guide. Update the same topic; do not create per-session notes.
Keep scratch, raw logs and measurements in ignored `../local/` or the evidence store.

Use a short title, status/date, scope and related commit/document, followed by:

- **Decision and reason:** only context needed to continue the work.
- **Evidence and limits:** source identities, useful observations and what is unverified.
- **Remaining work:** concrete next actions and where their outcome will be maintained.

Recheck old observations before treating them as current. When work closes, promote
useful conclusions to the canonical guide and remove the redundant handoff. Preserve
unique evidence with provenance when it still informs a decision. Commit agent-only
material separately with `agent:`; never store credentials or private reasoning.
