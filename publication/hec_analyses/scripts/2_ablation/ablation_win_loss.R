source(here::here("shared.R"))

# Which ablation configurations win and lose, per rank: the same winner/loser rule as the main
# comparison (compare_models(), helpers/compare_models.R: values rounded to 3 decimals; winner =
# contender, plausibly at or above the chance share 1/K for K competing configurations; clear
# winner = every rival credibly below chance), pooled over studies and metrics (one figure per setting).
# Configurations are compared within the same seed and bootstrap draw; every seed x draw is one
# resample, so seed variation enters the win shares. Three versions: all configurations, without
# regularization (incl. `standard`), and the regularized configurations only.
# Only the species, genus and family rank, which every study has.

# Study options:
# studies() |>
#   filter(status == "current") |>
#   filter(!str_detect(study, "gefion")) |>
#   filter(role == "replicates") |>
#   collect() |>
#   pull(study)
ablation_studies <- c("plantnet", "lepi-1513")
ranks_used <- c("Species", "Genus", "Family")
# `calibrated` is the main figure and `zero` a separate one (suffix `_zero`), since the settings
# answer different questions (plain predictive quality vs. usefulness of the confidence scores)
settings <- c("calibrated", "zero")
# Micro precision, recall and F1 are left out: micro precision equals micro accuracy, and micro
# recall and F1 are functions of micro accuracy and coverage (recall = accuracy x coverage).
selected_metrics <- c(
  "accuracy",
  "recall",
  "precision",
  "f1",
  "micro_accuracy",
  "theilU",
  "coverage"
)
n_rep <- 1000 # bootstrap draws per seed

ablation <- ablation_results(
  ablation_studies,
  ranks_used,
  settings,
  selected_metrics,
  n_rep
)
results <- ablation$results
config_levels <- ablation$config_levels

# Winners and losers per study, setting, metric and rank among the configurations in `df`
compare_configs <- function(df) {
  df |>
    group_by(study, setting, metric, level) |>
    group_modify(\(g, key) {
      compare_models(transmute(
        g,
        model = method_str,
        replicate = draw,
        value = value
      ))
    }) |>
    ungroup()
}

# ---- Figure: dot matrix of the components next to the win/loss bars --------------------------------

# plot_model_matrix() and scale_y_config(): helpers/ablation/figures.R

# Win/loss bars per configuration (rows, in `configs` order) and `strata` (columns)
plot_win_loss <- function(comparison, configs, strata = "level") {
  comparison |>
    outcome_shares(c("model", strata)) |>
    win_loss_long() |>
    ggplot(aes(share, model, fill = outcome)) +
    geom_col(width = 0.8) +
    geom_vline(xintercept = 0) +
    scale_fill_manual(values = win_colours, breaks = rev(win_legend_breaks)) +
    scale_x_continuous(
      breaks = seq(-1, 1, by = 0.5),
      labels = \(x) label_percent()(abs(x)),
      limits = ~ c(-1, 1) * max(abs(.))
    ) +
    scale_y_config(configs) +
    facet_grid(stats::reformulate(strata)) +
    labs(x = NULL, y = NULL, fill = NULL) + # legend and axis title: win_loss_key()
    theme(
      axis.text = element_text(size = rel(0.8)),
      axis.text.y = element_blank(), # the row identity is in the dot matrix
      axis.ticks.y = element_blank(),
      axis.title.x = element_text(size = rel(0.9)),
      axis.ticks.y.left = element_line(),
      legend.position = "none",
      plot.margin = margin()
    )
}

versions <- list(
  all = results,
  no_regularization = filter(results, !regularization),
  regularization = filter(results, regularization)
)

for (version in names(versions)) {
  cat(str_glue("Computing winner-loser comparison '{version}'\n"))
  comparison <- compare_configs(versions[[version]])
  configs <- config_levels[config_levels %in% comparison$model]
  K <- length(configs)
  height <- 0.22 * (K + 4) + 2.6

  # One figure per setting. Row 1: dot matrix and bars; row 2: the legend and axis title, centred
  # on the zero line of the middle panel (panel heights in units of one configuration row; the
  # key gets four)
  for (s in settings) {
    figure <- wrap_plots(
      A = plot_model_matrix(configs),
      B = plot_win_loss(filter(comparison, setting == s), configs),
      C = plot_spacer(),
      D = win_loss_key(),
      design = "AB\nCD",
      widths = c(1, 6),
      heights = c(K, 4)
    )
    save_figure(
      figure,
      str_c(
        "ablation_win_loss_",
        version,
        if (s != "calibrated") str_c("_", s)
      ),
      width = 12,
      height = height
    )
  }
}
