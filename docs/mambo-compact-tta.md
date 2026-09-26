# Preliminary TTA composition study

This exploratory study shortlisted the recipes evaluated in the
[full comparison and selection record](mambo-composed-tta.md). Its useful finding
was to combine padding with rotations inside existing views, rather than add
separate inference passes. Three-view ±30° rotations with stronger padding
deserved full evaluation; the later comparison selected 25% padding.

The sample comprised 38 flagged errors, 128 controls and 1,024 randomly selected
Flemming reporting images. The random sample was disjoint from the flagged
cases, but not independent of the previously studied reporting population.
Fixed thresholds came from ordinary V3, not candidate-specific calibration.
These results cannot replace the full-data comparison.

The [recorded evidence](assets/mambo-compact-tta.json) retains all candidates.
Replay uses [compact_tta.py](../dev/releases/mambo_v3/compact_tta.py) and
[compact_tta_metrics.py](../dev/releases/mambo_v3/compact_tta_metrics.py);
[historical commands and tables](https://github.com/asgersvenning/mini_trainer/blob/852bf712e85b8d1a6b9c9c6d31b3b5d807904303/docs/mambo-compact-tta.md)
are available in Git. For integration decisions, use the
[deployment README](../deployment/README.md).
