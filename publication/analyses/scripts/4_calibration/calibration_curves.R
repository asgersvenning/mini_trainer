source(here::here("shared.R"))

# Confidence calibration of the architecture-comparison models: for every model (backbone x head) and
# every rank it predicts, the confidence of the top prediction against how often that prediction is right,
# in equal-width confidence bins (the reliability curve, drawn as accuracy minus confidence: above 0 is
# underconfident, with Wilson intervals of the accuracy of each bin) and as the expected calibration error,
#   ECE = sum over bins of (images in the bin / all images) * |accuracy - mean confidence|.
# The confidence is the one stored in the predictions of the replicates folder. The reliability bins per
# study are cached in data/derived/calibration/ (per number of bins; delete to recompute). How much the
# heads differ in calibration error, with intervals, is in calibration_vs_flat.R.
studies_used <- c("gefion-global-lepi", "gefion-plantnet")
n_bins <- 30
min_bin_images <- 5 # sparser bins are left out of the figure (not of the ECE)

# Wilson score interval (95%) of the accuracy of a bin with `n` images, as columns `lower` and `upper`
# (the bin's mean confidence is taken as fixed)
wilson_interval <- function(accuracy, n, z = stats::qnorm(0.975)) {
  centre <- (accuracy + z^2 / (2 * n)) / (1 + z^2 / n)
  half <- z *
    sqrt(accuracy * (1 - accuracy) / n + z^2 / (4 * n^2)) /
    (1 + z^2 / n)
  tibble(lower = centre - half, upper = centre + half)
}

# Reliability bins of one study: one row per run, rank and confidence bin
reliability_bins <- function(study) {
  truth <- calibration_truth(study)
  map(pull(runs(study), run_id), \(run_id) {
    run_calibration(study, run_id, truth) |>
      mutate(bin = confidence_bin(confidence, n_bins)) |>
      summarise(
        n = n(),
        accuracy = mean(correct),
        confidence = mean(confidence),
        .by = c(rank, bin)
      ) |>
      mutate(
        run_id = run_id,
        head = run_head(run_id),
        backbone = run_backbone(run_id),
        .before = 1
      )
  }) |>
    list_rbind()
}

bins <- map(set_names(studies_used), \(study) {
  cached(
    derived_path("calibration", str_c(study, "_", n_bins, ".rds")),
    reliability_bins(study)
  )
})

for (study in studies_used) {
  b <- bins[[study]] |>
    filter(!rank %in% cfg$calibration_skip_ranks[[study]]) |>
    mutate(
      head = relabel(head, "head"),
      backbone = relabel(backbone, "backbone"),
      rank = relabel(rank, "level")
    )
  ece <- b |>
    summarise(
      ECE = sum(n * abs(accuracy - confidence)) / sum(n),
      .by = c(backbone, head, rank)
    )
  cat(
    "\n",
    study,
    ": expected calibration error (",
    n_bins,
    " bins)\n",
    sep = ""
  )
  print(
    pivot_wider(
      arrange(ece, backbone, head, rank),
      names_from = rank,
      values_from = ECE
    ),
    n = Inf
  )

  plot <- b |>
    filter(n >= min_bin_images) |>
    mutate(wilson_interval(accuracy, n)) |>
    ggplot(aes(confidence, accuracy - confidence, colour = head)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
    geom_ribbon(
      aes(
        ymin = lower - confidence,
        ymax = upper - confidence,
        fill = after_scale(color)
      ),
      alpha = 0.4,
      linewidth = .3
    ) +
    geom_line() +
    geom_point(size = 0.8) +
    facet_grid(rank ~ backbone, scales = "free_y") +
    scale_x_continuous(
      limits = 0:1,
      expand = expansion(),
      breaks = c(0, 0.25, 0.5, 0.75, 1),
      labels = c("0%", "25%", "50%", "75%", "") # no "100%": it would collide with the next panel's "0%"
    ) +
    scale_y_continuous(labels = label_percent()) +
    scale_color_manual(values = head_colours) +
    labs(x = "Confidence", y = "Accuracy \u2212 confidence", colour = "Head") +
    theme(panel.spacing.x = unit(2, "lines"))
  save_figure(
    plot,
    str_c("calibration_", cfg$study_dataset[[study]]),
    width = 1.4 * cfg$fig_width,
    height = 1 + 1.6 * n_distinct(b$rank),
    scale = 1.5
  )
}
