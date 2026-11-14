# Ablation replicates: the bootstrap replicate tables of the ablation studies, pooled over studies,
# in the form used by the win/loss and path analyses. Depends on data.R, labels.R and design.R.

# Components of a configuration (columns of the dot matrix, most significant first)
component_levels <- c(
  "hidden",
  "normalized",
  "regularized",
  "LA",
  "EMLA",
  "hierarchical"
)

# The components of a configuration label ("hidden × LA"; "standard" has none)
config_components <- function(model) {
  lapply(model, \(m) {
    if (m == "standard") character() else strsplit(m, " × ", fixed = TRUE)[[1]]
  })
}

# Binary code of a configuration: each component is a bit, on = 1, the first component is the
# most significant bit ("standard" is 0)
config_code <- function(model, components = component_levels) {
  bits <- 2^(seq_along(components) - 1)
  vapply(
    config_components(model),
    \(k) sum(bits * (components %in% k)),
    numeric(1)
  )
}

# Replicates of one ablation study: `settings`, `metrics`, the first `n_rep` bootstrap draws per
# seed and the ranks `ranks` (labels of relabel(.., "level")). One row per run, draw, metric and
# rank, with the run metadata (`method_str`, `n_methods`, `regularization`, `seed`) and `draw`, a
# unique id of the seed x draw combination.
load_ablation <- function(study, settings, metrics, ranks, n_rep) {
  metadata <- ablation_runs(study) |>
    mutate(method_str = if_else(method_str == "", "standard", method_str)) |>
    select(run_id, seed, method_str, n_methods, regularization)

  replicates(study) |>
    filter(
      setting %in% !!settings,
      metric %in% !!metrics,
      replicate >= 1,
      replicate <= !!n_rep
    ) |>
    collect() |>
    inner_join(metadata, by = "run_id") |>
    mutate(
      level = relabel(rank_factor(level, .env$study), "level"),
      study = .env$study,
      draw = seed * 10000 + replicate
    ) |>
    filter(level %in% ranks)
}

# Metrics that carry no information for the comparison within a setting, found from the data:
# constant across the configurations in every resample (e.g. coverage without abstention), or
# identical (within `tol`) to a metric earlier in `metrics` in every run and resample (e.g.
# recall = accuracy without abstention). The first of a set of identical metrics is kept.
redundant_metrics <- function(df, metrics, tol = 1e-6) {
  find <- function(d) {
    wide <- d |>
      select(study, level, draw, run_id, metric, value) |>
      pivot_wider(names_from = metric, values_from = value)
    metrics <- intersect(metrics, names(wide))

    is_constant <- \(m) {
      ranges <- summarize(
        wide,
        range = max(.data[[m]]) - min(.data[[m]]),
        .by = c(study, level, draw)
      )
      all(ranges$range < tol, na.rm = TRUE)
    }
    constant <- Filter(is_constant, metrics)

    kept <- character()
    duplicates <- tibble(metric = character(), duplicate_of = character())
    for (m in setdiff(metrics, constant)) {
      same <- Filter(
        \(k) all(abs(wide[[m]] - wide[[k]]) < tol, na.rm = TRUE),
        kept
      )
      if (length(same)) {
        duplicates <- bind_rows(
          duplicates,
          tibble(metric = m, duplicate_of = same[1])
        )
      } else {
        kept <- c(kept, m)
      }
    }
    bind_rows(
      tibble(metric = constant, reason = "constant"),
      mutate(duplicates, reason = "duplicate")
    )
  }

  df |>
    nest(.by = setting) |>
    mutate(drop = map(data, find)) |>
    select(setting, drop) |>
    unnest(drop)
}

# The ablation replicates of `studies` pooled, without the uninformative metrics: the constant
# ones (found in the data, and printed with the duplicates for checking) and, without abstention
# (`zero`), recall, which equals accuracy there. The studies must have the same configurations.
# Returns a list: `results` and `config_levels`, the configurations in ascending binary code, so
# that the dot matrix reads as a counter from top to bottom.
ablation_results <- function(studies, ranks, settings, metrics, n_rep) {
  results <- map(
    studies,
    \(s) load_ablation(s, settings, metrics, ranks, n_rep)
  ) |>
    bind_rows()

  redundant <- redundant_metrics(results, metrics)
  print(redundant, n = Inf)
  dropped <- bind_rows(
    filter(redundant, reason == "constant"),
    tibble(setting = "zero", metric = "recall", reason = "manual")
  ) |>
    distinct(setting, metric)
  results <- anti_join(results, dropped, by = c("setting", "metric"))

  config_sets <- results |>
    distinct(study, method_str) |>
    summarize(configs = list(sort(method_str)), .by = study)
  stopifnot(length(unique(config_sets$configs)) == 1)

  list(
    results = results,
    config_levels = results |>
      distinct(method_str) |>
      arrange(config_code(method_str)) |>
      pull(method_str)
  )
}
