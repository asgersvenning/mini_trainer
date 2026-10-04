# Publication metric protocol and legacy-analysis audit

Audit date: 4 October 2026. This records the proposed publication contract and
confirmed implementation gaps; it does not retroactively redefine archived results.
The R scripts are useful exploratory work, not yet a reproducible final-table pipeline.
Current unthresholded ablation reports remain valid under their recorded definitions.
The selective main-table extension is **not yet qualified**.

## Two complementary evaluations

Keep the existing **unthresholded mechanism analysis**, including frequency effects,
calibration and confusion flows. Add **calibration-selected selective performance**
for deployment relevance. Threshold selection can compensate for differences in the
head or loss; reporting only selected operating points would obscure their mechanisms.
Change the draft's blanket “all reported metrics” statement accordingly.

For each dataset/cohort and model, record ordinary macro recall/F1, selective macro
accuracy/precision, pooled selective accuracy, coverage, threshold and class-support
counts at each taxonomic rank. Report empirical and equal-class coverage, including
rare-class coverage: equal overall coverage can conceal rejection of rare species.
Keep NLL/Brier/ECE on the unthresholded predictions as the primary calibration analysis.
Use risk/coverage curves as a secondary operating-point comparison where useful.

Use the same leaf-probability summation to construct genus/family predictions for
flat and hierarchical models in the ablations. A native parent head and the ancestor
of the winning species are different prediction rules; identify them separately.
`mini_metrics.MetricDF.add_combinations` maps the winning leaf and copies its confidence;
it cannot recover parent argmax/confidence from summed leaf probabilities.
Historical top-1 CSVs support the ancestor-mapped rule, not probability aggregation.

## Mathematical clarifications for the draft

Let `n` be all evaluated instances, `a` accepted predictions and `t` accepted correct
predictions in exhaustive, single-label classification with a common vocabulary.
The proposed metric-specific micro weights give exactly:

- selective micro accuracy = micro precision = `t/a`;
- micro recall = `t/n` = coverage × selective accuracy;
- micro F1 = `2t/(n+a)` = `2 × coverage × accuracy / (1 + coverage)`;
- coverage = `a/n`.

Thus micro accuracy/precision are redundant here, while recall and F1 penalize
abstention. They are **not all metrics computed solely on retained instances**.
Suggested wording: “after applying the abstention rule, retaining rejected instances
in the true-class supports used by recall and F1.” The half-factor in the draft's
F1 micro weights cancels correctly. Weights may be zero, so specify nonnegative
weights and the case where their total is zero. Zero-threshold exhaustive outputs
must have equal micro accuracy, precision, recall and F1.

The draft also needs a metric-specific supported class set, or a different explicit
zero-denominator policy. At inspected `mini_metrics` revision
`70cc69adc05362863439277048e06386c1f885e1`, the current definitions are:

| Macro metric | Classes receiving nonzero averaging weight |
| --- | --- |
| Selective accuracy | True classes with at least one accepted instance |
| Selective precision | Classes with at least one accepted prediction |
| Recall | Classes with true support, including fully rejected classes |
| F1 | Union of classes with true support or accepted predicted support |

Recommended initial contract: document these support-conditioned macro definitions,
retain supported-class counts, and make macro recall/F1 the main balanced performance
measures. Do not silently replace undefined conditional accuracy with zero or claim
that its changing denominator is a fixed-class macro average. At complete rejection,
selective accuracy/precision are undefined, coverage/recall/F1 are zero when true
support is nonempty. A fixed-vocabulary alternative requires an explicitly named
policy and matching optimizer; it must not enter through undocumented defaults.

## Theil's U

Retain `U(Y|X) = I(Y;X)/H(Y)` as an explicitly **unthresholded association diagnostic**.
This matches the current implementation. It measures information, not correctness:
a deterministic swap of all class identities gives U=1 even with accuracy=0.
It uses the empirical class distribution; it is not immune to imbalance, does not
replace macro metrics, and raw plug-in mutual information is not chance-corrected.
See the primary [mutual-information documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html)
and [discussion of chance adjustment](https://scikit-learn.org/stable/modules/clustering.html#mutual-information-based-scores).

The draft should replace the claim of automatically avoiding imbalance distortion
with the narrower information-reduction interpretation. Constant true labels have
`H(Y)=0`, so U is undefined. Accepted-only U and U with a single abstention symbol
are distinct possible measures; neither is the current library result, and neither
should silently replace archived U. No additional U variant is needed for the first
publication-table pass.

## Freeze threshold calibration outside evaluation

Recommended sequence:

1. Freeze common calibration and evaluation membership across treatments using
   canonical sample IDs, with observation-level grouping when multiple images share
   an observation. Record dataset, taxonomy, known/unknown policy and hashes.
2. Calibrate ordinary `MacroF1` on calibration predictions only. Explicitly record
   tolerance, search coordinates and tie/plateau selection, not just “optimal”.
   The inspected library defaults to an exact F1-state curve followed by selection
   from a near-optimal connected region (`eps=.01`, rejection-rate coordinates,
   no bootstrap). This is not necessarily the exact argmax.
3. For the first standardized evaluation, explicitly pin that documented library
   contract rather than tuning its tolerance against evaluation results. The legacy
   `boot_metrics.py` explicitly uses `.05`, a different tolerance. Retain those
   outputs as legacy sensitivity results until regenerated.
4. Freeze one threshold per model/rank and pass it to `evaluate_file` with
   `optimal=False` on the common evaluation set. Independent rank thresholds evaluate
   rank-specific operating points; a coherent “deepest accepted rank” policy is a
   separate deployment rule and needs its own definition.
5. Store calibration IDs, thresholds, objective, achieved calibration score, class
   supports, library commit, exact arguments and raw predictions. Filter explicitly
   before calibration when intending known-only calibration; the current
   `known_only` option filters reporting but not the optimizer input.

`evaluate_file(optimal=True)` partitions the supplied file into 90% reporting and
10% calibration. Those subsets are disjoint, so this is not itself within-split
leakage, but it does not evaluate the complete original test set or persist its IDs.
`boot_metrics.py` repeats this partitioning; its seeds are not independent trained
models and its spread is not automatically a bootstrap confidence interval.

Existing ablation validation outputs support exploratory comparisons. For final
selective test tables, use separate calibration predictions and untouched test
predictions. Audit existing files before requesting inference; if predictions are
missing, checkpoint inference is sufficient. No retraining is required. Keep
open-set Flemming results separate from known-only species classification.

## Confirmed implementation issues and qualification boundary

Source references below are relative to the inspected `mini_metrics` checkout.
These are source-verified and reproduced with tiny fixtures; no broad test suite
or full-data optimization sweep was needed.

| Finding | Evidence | Required action |
| --- | --- | --- |
| Rejected predictions receive rank-recall credit | `hierarchical.py:84–90,125–126`; two correct leaf predictions at confidence .1, threshold .5 yield coverage0, ordinary macro recall/F1=0, but macro rank recall/F1=1 | Zero rejected contributions while preserving true support; test mixed/all rejection and partial hierarchical credit |
| RankError crashes on complete rejection | `hierarchical.py:56–57`; empty accepted frame reaches `.max()` | Return an explicit undefined empty result and zero support |
| U ignores threshold | `metrics.py:253–271`; `prediction_made` unused | Name/document raw U; only change implementation for a separately specified new estimand |
| Macro supports differ | `helpers.py:338`, `metrics.py:61,113,193` | Document supported sets and empty cases; change code only if adopting a different explicit policy |
| Known-only calibration is not enforced | `metrics.py:728–735,762` | Prefilter calibration or fix API ordering with regression coverage |

Do not put thresholded rank-distance recall/F1 or rank-error in confirmatory tables
until the first two bugs are fixed in a dedicated `mini_metrics` change. This does
not block ordinary species/genus/family metrics under consistent prediction rules.
Keep metric computation in `mini_metrics`; do not duplicate it in R or create a
parallel implementation inside `mini_trainer`. Pin the validated revision after
fixes; the research runner must not rely on the sibling checkout being installed.

Qualification should test count identities, confidence ties, mixed/all abstention,
missing/predicted-only classes, unknown-label policy, calibration/test disjointness,
and parent-score semantics. Use a small real-file replay after those contracts pass.
Only run a full CLI benchmark if a fix has a credible performance consequence.

## What to reuse and repair in the R workflow

Audited source directory:
`C:/Users/asger/OneDrive - Aarhus universitet/Documents/PhD/Projekter/Structured Classification/Testing/metrics_first_experiment`.
`tables.R` reads the separate Downloads directory
`gefion_experiment_metrics_25082026`; its data is not self-contained in the R project.

- Reuse the dataset/backbone/rank table layout (`tables.R:94–157`), per-rank metric
  profiles (`demo.qmd:285–348`) and class-recall ECDFs (`672–774`). Keep unweighted
  class and image-weighted ECDFs distinct, and associate recall with measured
  training frequency rather than inferring frequency from recall itself.
- Replace broad CSV discovery (`tables.R:23–35`) with an explicit manifest. It also
  admits taxonomy CSVs from `results/combinations`. Retain thresholds and vocabulary
  metadata currently removed at line46. Label U and coverage as standalone, not micro.
- Rank full-precision values before display rounding (`tables.R:109–115`); boldface
  must not imply statistical significance. Prefer displaying paired effects over
  a winner count across correlated metrics.
- `demo.qmd` expects `_opt`, `_zero`, `_thr` inputs but the current directory only
  contains excluded `_old` directories for those summaries. Restore a manifest of
  actual inputs rather than allowing an empty analysis to proceed.
- Historical zero-threshold autoregressive Global Lepidoptera summaries have
  coverage1 but micro accuracy .928861, precision .935724, F1 .930504. These fail
  the proposed ordinary pooled identities. Recompute from raw rows under pinned
  semantics; the old summaries alone do not identify the generating-version cause.
- The mean/SE over metric rankings (`demo.qmd:379–415`) is variation across chosen,
  correlated metrics, not model uncertainty. Training seeds, observations and
  calibration partitions are different replication units and must remain separate.
- Paired correctness (`507–539`) needs unique canonical IDs plus label/filename
  checks. Do not allow many-to-many joins or silently compare only the intersection
  of accepted predictions. Keep abstention as an explicit outcome.
- ECDF prose (`777–785`) cannot infer global stochastic dominance or significance
  from a local difference. The consensus plot includes the focal model (`467–469`)
  despite its description; the deepest-correct-rank attempt filters to level0
  (`795–808`) before measuring depth. These require correction before reuse.
- `metric_var.R` explores useful optimizer sensitivity, but generating settings and
  uncertainty units are not recoverable from its filenames alone. Preserve as EDA,
  not a rationale for more threshold sweeps now.

The practical next increment is therefore a pinned calibration/evaluation runner
and manifest-driven table input, qualified against the above contracts. Do that
before regenerating main tables or extending embedding-space analysis. Preserve
legacy scripts and numbers as historical evidence rather than rewriting them in place.

### Minimal replay of the confirmed rejection defects

Run with the inspected `mini_metrics` environment, without installing or updating
packages. This is a diagnostic, not a proposed replacement implementation:

```python
from mini_metrics.data import MetricDF
from mini_metrics.metrics import MacroRecall, MacroF1
from mini_metrics.hierarchical import MacroRankRecall, MacroRankF1, RankError

frame = MetricDF(
    instance_id=[0, 1], filename=["", ""], level=[0, 0],
    label=["a", "b"], prediction=["a", "b"],
    confidence=[0.1, 0.1], threshold=[0.5, 0.5],
)
taxonomy = {"a": ("a", "g", "f"), "b": ("b", "g", "f")}
for metric in (MacroRecall, MacroF1, MacroRankRecall, MacroRankF1):
    options = {"combinations": taxonomy} if "Rank" in metric.__name__ else {}
    print(metric.__name__, metric()(frame, verbose=0, **options))
try:
    print(RankError()(frame, combinations=taxonomy))
except ValueError as error:
    print(type(error).__name__, str(error))
```

Observed per-level results: ordinary macro recall/F1 `{0: 0.0}`, rank recall/F1
`{0: 1.0}`; RankError raises `ValueError: zero-size array to reduction operation
maximum which has no identity`. These rank metrics are distinct from ordinary
per-rank genus/family recall used in the current ablation reports.
