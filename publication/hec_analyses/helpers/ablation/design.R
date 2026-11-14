# Ablation studies: run metadata. Depends on the ERDA accessors (data.R).

# Run metadata with derived method flags (adjusted/gated loss, hierarchical head, number of
# methods, run_type = run_id without the seed, a label of the methods used). Keeps
# `hierarchical` and every column that varies within the study.
ablation_runs <- function(study) {
  runs(study) |>
    collect() |>
    mutate(
      adjusted = loss %in% c("fixed", "emla"),
      gated = loss == "emla",
      hierarchical = map_lgl(rank_weights, \(w) {
        !is.na(w) && any(fromJSON(w)[-1] > 0)
      }),
      n_methods = normalized + regularization + adjusted + gated + hierarchical + hidden,
      run_type = str_remove(run_id, "_seed\\d+$"),
      method_str = apply(
        across(c(hidden, normalized, regularization, adjusted, gated, hierarchical)),
        1,
        function(x) {
          str_c(c("hidden", "normalized", "regularized", "LA", "EMLA", "hierarchical")[x > 0], collapse = " × ")
        }
      )
    ) |>
    select(c(hierarchical, where(~ n_distinct(.) > 1)))
}
