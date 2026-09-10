---
trigger: always_on
---

# Contributions

Follow [AGENTS.md](../../AGENTS.md) and the review checklist in [dev/README.md](../../dev/README.md).
Keep changes concise and focused. Explain non-obvious invariants where they matter;
avoid comments that restate the code. Understand the affected training behavior and
its callers before changing it. Keep behavior fixes separate from mechanical cleanup.

Follow [the agent workspace policy](../README.md) for note locations and separate
`agent:` commits. Keep application changes and developer-facing docs out of those
commits; the prefix records purpose, not authorship.
