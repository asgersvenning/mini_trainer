# Contributions

Follow [AGENTS.md](../../AGENTS.md) and the review checklist in [dev/README.md](../../dev/README.md).
Keep changes concise and focused. Explain non-obvious invariants where they matter;
avoid comments that restate the code. Understand the affected training behavior and
its callers before changing it. Keep behavior fixes separate from mechanical cleanup.

Follow [the agent workspace policy](../README.md) for note locations and separate
`agent:` commits. Keep application changes and developer-facing docs out of those
commits; the prefix records purpose, not authorship.

## Choosing fixes and experiments

- Start from the user's intended outcome and constraints. When corrected about
  priorities or scope, revise the proposed work instead of repeatedly defending
  the previous approach. For deployment, count recurring developer effort across
  environments as part of the cost, alongside implementation complexity.
- Separate the observed failure, suspected mechanism, and the layer we control.
  Prefer the smallest durable change in that layer, using existing package-manager
  and framework capabilities. A kernel workaround, diagnostic probe, or clearer
  error may help investigation; identify whether it prevents the failure or only
  detects/contains it. Do not present a guardrail as a compatibility fix.
- Test cheap, reversible hypotheses before adding custom infrastructure or
  repeatedly debating their limitations. State what the experiment changes, what
  result would support it, and what remains uncertain. Inspect environment details
  when they distinguish hypotheses or guide action, rather than as an end in itself.
- For dependency hypotheses, test the actual resolved and installed environment
  in isolation. Changing version ranges while retaining the same locked packages
  does not test fresh resolution. Compare installed versions and exercise real
  inference; successful dependency resolution alone is insufficient. Follow the
  root README's environment conventions and preserve working environments.
- Distinguish reproducibility within an evaluation campaign from adaptability of
  deployment installation. Preserve resolved versions as evidence without assuming
  one evaluation lock must govern all future installations. Neither removing locks
  nor pinning versions is automatically a compatibility solution.
- Use available representative environments for bounded qualification. Agreement
  across them supports a practical improvement without proving universal support
  or identifying which changed dependency caused it. Expand the matrix or isolate
  individual changes when that would change the decision; do not require exhaustive
  architecture access before trying a useful fix.
- Review the complete operator workflow before handing it over. Minimize new flags,
  configuration copies, path edits, and environment switches; count these as real
  complexity even when the code is short. If a setup change alters paths or commands,
  update downstream steps consistently rather than asking users to translate them.
  State exactly what must be rerun and what existing assets/results can be reused.
