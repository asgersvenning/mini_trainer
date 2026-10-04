# Presenting the methodological evidence

Present conditional contributions and their interactions, rather than a monotonic
ladder in which every addition must increase accuracy. Keep datasets, cohorts,
schedules and seeds identifiable in every panel. All current ablation choices and
comparisons use validation data; historical test evaluations provide separate
external context, not extra ablation replicates.

| Question | Primary evidence | Necessary companion evidence |
| --- | --- | --- |
| Normalized head package | Paired N-on minus N-off macro/rare recall at each R setting | Empirical and equal-class NLL, Brier and reliability; N×R finite differences |
| Prototype regularization | R-on minus R-off rare-to-rare error probability at each N/H setting | Rare recall and rare-to-common errors, prototype effective rank; spreading alone does not establish benefit |
| Frequency adjustment | EMLA minus CE frequency–soft-prediction-mass association under equal true-class weighting | Frequency–recall slope and correlation, empirical accuracy and common/rare recall; correlation need not universally move toward zero |
| Adaptive rather than fixed adjustment | Common/rare learning trajectories during and after warmup | Training gates and available batch accuracy/loss histories; endpoint effects separately; adjusted losses are not directly comparable |
| Hierarchy | Genus/family macro recall from identical aggregation of leaf probabilities | Species/rare recall, taxonomic error destinations and H×R; retain natural branching imbalance |
| Replication and duration | Individual paired seed effects; corrected PlantNet comparison | Twenty-epoch schedule separately; do not describe it as continuation or pool its endpoint with ten epochs |

The current evidence is mixed: normalization improves the first PlantNet seed;
regularization spreads prototypes but does not consistently improve rare recall;
EMLA preserves early common-class learning relative to fixed adjustment in
Lepidoptera; hierarchy repeatedly improves coarse-rank recall while its species
benefit and interaction with regularization remain uncertain. Tomorrow's second
PlantNet seed and adjustment/hierarchy treatments test whether these patterns
replicate. Avoid selecting a favorable endpoint or seed after seeing those results.

## Reproducible presentation commands

Use a frozen study mirror with `prepared.json` and `plan.json`. Generate analysis
from completed hashed artifacts using the existing analyzer, then assemble tables
and figures. Each output path must be new:

```bash
uv run --no-sync python -m publication.experiments.training_ablations.analysis \
  STUDY NEW_REPORT
uv run --no-sync python -m publication.experiments.training_ablations.presentation \
  STUDY NEW_PRESENTATION NEW_REPORT --require-complete
```

The final flag refuses an incomplete frozen plan. Omit it for an explicitly
preliminary snapshot. To reuse completed analysis, supply multiple **disjoint**
report directories from that same prepared study and analysis implementation:

```bash
uv run --no-sync python -m publication.experiments.training_ablations.presentation \
  STUDY NEW_PRESENTATION EARLIER_REPORT NEW_RUNS_REPORT --require-complete
```

Use the analyzer's `--variants` selector to restrict a new report to newly completed
treatments; if additional seeds of an already analyzed treatment have completed,
regenerate that treatment's report and omit its superseded report. Duplicate runs
are rejected. Do not manually merge different prepared studies or schedules.
Run each of the two original seed campaigns separately, then juxtapose their
paired effects; their frozen study identities differ.

Outputs include:

- `coverage.json`: planned/analyzed counts and missing run/contrast identities.
- `endpoints.csv`: empirical/equal-class performance and calibration, frequency
  relationships, rare-class error destinations, parent recall, available prototype rank.
- `paired-contrasts.csv`: treatment minus comparator within each seed;
  `replicates.csv`: count/mean/min/max, **not confidence intervals** from two seeds.
- `interactions.json`: existing factorial estimator, retaining conditions and seed;
  incomplete factorial blocks are omitted, never filled using another seed.
- Paired-effect PNG/PDF and `diagnostics/` with reliability, frequency bias,
  confusion flows, available epoch dynamics and training gate trajectories.
- Small source tables in `analysis/`, frozen plan and input hashes in provenance.
  `analysis_plots` can rerender these diagnostics without models or logits.

A presentation directory is also a self-contained input for rebuilding all paired
results and figures, with no original study mount:

```bash
uv run --no-sync python -m publication.experiments.training_ablations.presentation \
  SAVED_PRESENTATION REBUILT_PRESENTATION SAVED_PRESENTATION/analysis
```

CSV rates use fractions; rate differences multiply by 100 to obtain percentage
points. Figure contrasts explicitly label their units. Equal-class ECE uses
pooled confidence bins after weighting each true class equally; it is not the
arithmetic mean of separately binned per-class ECEs. Lower ECE need not imply better
accuracy or lower NLL. Preserve reliability diagrams alongside the scalar ECE.

## Completion review tomorrow

1. Verify all planned completion manifests and prediction hashes; analyze only
   completed runs. A controller reporting error alone is not evidence of failed
   training. Require 16/16 PlantNet and 8/8 hierarchy before calling those studies complete.
2. Render the four scientific comparisons above and inspect individual seeds,
   conditional effects and error destinations before revising conclusions.
3. Retain the existing Lepidoptera `dynamics` outputs for both ten-epoch seeds and
   the six twenty-epoch runs. Existing archived batch-history review supplies
   sub-epoch context; epoch confusion/gate aggregates must not be labelled batch-level
   class diagnostics. Final FP32 reload scores and AMP training-time curves are
   different evaluation paths and should remain labelled.
4. Keep the historical Gefion backbone/head results in a separate table. Their
   recipes, ranks and taxonomy differ. Their saved top-1 CSVs support accuracy and
   confusion analysis, not reconstruction of NLL, Brier or ECE. Likewise, do not
   invent duration-study calibration where final probability analysis is unavailable.
5. Archive a versioned presentation bundle plus manifest on ERDA under
   `/publications/hierarchical_classification`. Preserve the earlier snapshots.
   Tables suffice to rerender; raw probabilities are additionally required to
   recompute scores. Public download access remains separate from the private
   SFTP archive. See the artifact-manifest commands in the parent study README.

No new training is required for these presentation steps. New embedding extraction,
spherical means and isotropy analyses remain a subsequent task; existing prototype
geometry must not be described as measured image-embedding geometry.
