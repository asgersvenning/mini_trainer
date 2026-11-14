source(here::here("shared.R"))

# Every configuration against the `standard` baseline (no hidden layer), per metric and rank: the
# quantitative counterpart of ablation_win_loss.R, with the same rows, dot matrix and rank panels,
# and a heatmap (one column per metric, macro metrics first) instead of the win/loss bars.
# A cell is the median paired difference to `standard` (same study, seed and bootstrap draw),
# pooled over seeds and draws and then averaged over the studies; the cell text shows it in
# percentage points. The colour is the same difference divided by its standard error from the
# paired bootstrap (the SD over bootstrap draws of that difference, within a seed, root mean square
# over seeds; combined over the studies as the SE of their mean), a z-score on the method used in
# the other analyses, so cells are comparable across metrics, ranks and settings. Seed noise is
# left out of the colour: `boot_share` below shows that the bootstrap SD covers most of the SD of
# a 3-seed mean difference. The colour scale is that of the win/loss figures, symmetric around
# zero. One figure per setting (`zero` with the suffix `_zero`).
ablation_studies <- c("plantnet", "lepi-1513")
ranks_used <- c("Species", "Genus", "Family")
settings <- c("calibrated", "zero")
metric_order <- c(
  # macro, then micro
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
  metric_order,
  n_rep
)
results <- ablation$results
configs <- ablation$config_levels
by_cell <- c("study", "setting", "level", "metric")

paired <- results |>
  mutate(
    diff = value - value[method_str == "standard"],
    .by = c(all_of(by_cell), draw)
  )

# Per study: the median difference and the bootstrap SD of the difference (within a seed, root
# mean square over seeds)
bootstrap_sd <- paired |>
  summarise(v = var(diff), .by = c(all_of(by_cell), method_str, seed)) |>
  summarise(boot_sd = sqrt(mean(v)), .by = c(all_of(by_cell), method_str))

gain <- paired |>
  summarise(diff = median(diff), .by = c(all_of(by_cell), method_str)) |>
  left_join(bootstrap_sd, by = c(by_cell, "method_str")) |>
  summarise(
    se = sqrt(sum(boot_sd^2)) / n(), # SE of the mean over the studies
    diff = mean(diff),
    .by = c(setting, level, metric, method_str)
  ) |>
  mutate(z = if_else(method_str == "standard", 0, diff / se))
print(
  gain |>
    select(setting, level, metric, method_str, z) |>
    pivot_wider(names_from = c(level, metric), values_from = z) |>
    arrange(setting, match(method_str, configs)),
  n = Inf
)

# How much of the uncertainty of a 3-seed mean difference does the bootstrap cover? `test`: the
# bootstrap SD (shared by all seeds, so it does not average down); `seed`: the SD over seeds of the
# draw-averaged difference, which does (divided by the square root of the number of seeds). Both
# as root mean squares over configurations and studies. `boot_share` is the bootstrap SD as a
# share of the combined SD.
noise <- paired |>
  filter(method_str != "standard") |>
  summarise(
    m = mean(diff),
    v = var(diff),
    .by = c(all_of(by_cell), method_str, seed)
  ) |>
  summarise(
    test = sqrt(mean(v)),
    seed = sqrt(var(m)),
    .by = c(all_of(by_cell), method_str)
  ) |>
  summarise(
    across(c(test, seed), \(x) sqrt(mean(x^2))),
    .by = c(setting, level, metric)
  ) |>
  mutate(boot_share = test / sqrt(test^2 + seed^2 / n_distinct(results$seed)))
print(arrange(noise, setting, level, match(metric, metric_order)), n = Inf)

# Colour limit: one for all cells, so that colours compare across metrics, ranks and settings
limit <- 5 # beyond |z| = 5 a difference is clearly not noise; the size is in the cell text

plot_gain <- function(gain, configs) {
  present <- metric_order[metric_order %in% gain$metric]
  macro <- present[present %in% c("accuracy", "precision", "recall", "f1")]
  micro <- setdiff(present, macro)
  gain |>
    mutate(
      metric = factor(
        metric,
        c(macro, micro),
        labels = unname(label_sets$metric[c(macro, micro)])
      )
    ) |>
    ggplot(aes(metric, method_str, fill = z)) +
    geom_tile(width = 0.95, height = 0.95) +
    geom_text(
      aes(
        label = if_else(
          method_str == "standard",
          "",
          if_else(
            round(diff * 100, 1) == 0,
            "0.0",
            sprintf("%+.1f", diff * 100)
          )
        ),
        colour = abs(z) > 0.6 * limit
      ),
      size = 2.2,
      family = plot_font
    ) +
    scale_colour_manual(
      values = c(`TRUE` = "white", `FALSE` = "grey20"),
      guide = "none"
    ) +
    scale_fill_gradient2(
      low = win_colours[["Clear loser"]],
      mid = "white",
      high = win_colours[["Clear winner"]],
      limits = c(-1, 1) * limit,
      oob = oob_squish,
      labels = \(x) {
        str_c(c("≤", rep("", length(x) - 2), "≥"), label_number()(x))
      }
    ) +
    scale_y_config(configs) +
    facet_grid(cols = vars(level)) +
    geom_text(
      data = \(d) {
        d |> distinct(level) |> arrange(level) |> slice_tail(n = 1)
      },
      aes(x = Inf, y = Inf),
      label = "Cell text: median difference to standard in percentage points",
      inherit.aes = FALSE,
      hjust = 1,
      vjust = -4.5,
      size = 3.1,
      family = plot_font
    ) +
    coord_cartesian(clip = "off") +
    labs(
      x = NULL,
      y = NULL,
      fill = "Difference to standard\n(paired-bootstrap SEs)"
    ) +
    theme(
      legend.position = "bottom",
      legend.key.width = unit(5, "lines"),
      legend.key.height = unit(2.25, "lines"),
      legend.title = element_text(size = rel(1), face = "plain", vjust = 1),
      legend.text = element_text(family = "Latin Modern Math", size = rel(0.9)),
      axis.text = element_text(size = rel(0.8)),
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      axis.text.y = element_blank(),
      axis.title.x = element_text(size = rel(0.8), face = "plain"),
      axis.line = element_blank(),
      plot.margin = margin()
    )
}

for (s in settings) {
  cat(str_glue("Computing baseline comparison '{s}'\n"))
  figure <- wrap_plots(
    A = plot_model_matrix(configs),
    B = plot_gain(filter(gain, setting == s), configs),
    design = "AB",
    widths = c(1, 5)
  )
  save_figure(
    figure,
    str_c("ablation_vs_standard", if (s != "calibrated") str_c("_", s)),
    width = 10,
    height = 0.22 * length(configs) + 3.2
  )
}
