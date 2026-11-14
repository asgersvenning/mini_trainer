source(here::here("shared.R"))

# Do post-hoc clusters of a head's species prototypes align with the taxonomy (genus to class)?
# AMI of clusters vs taxa per head, rank and clustering method, with a paired subsample
# bootstrap over species. Helpers: helpers/embeddings/ (geometry.R, heads.R, alignment.R).
studies <- c(
  "gefion-plantnet" = label_sets$dataset[["plantnet"]],
  "gefion-global-lepi" = label_sets$dataset[["global_lepi"]]
)
methods <- names(cluster_methods()) # all (labels: label_sets$cluster_method)
n_boot <- 1000
size <- 512 # species per subsample (whole genera, drawn until they hold this many), the same for both studies

# Bootstrap paths in the figure (one per head, method and draw, across ranks): "density" (the paths
# counted on a grid, colours mixed and opacity by count, spread horizontally by a Gaussian of
# `smooth_sd` in Delta AMI truncated at 3 SD; cells fainter than `min_alpha` are dropped), "lines"
# (one faint line per path) or "none". The grid steps are powers of two so that geom_raster gets an
# exactly regular grid.
path_layer <- "none"
smooth_sd <- 0.01
min_alpha <- 0.01
x_bin <- 2^-10
y_bin <- 2^-7

head_names <- c(
  "flat",
  "hierarchical",
  "independent",
  "conditional",
  "conditional (path sum)"
)

# AMI of the post-hoc clusters of the head prototypes (species in vocabulary order, unit length) with
# the taxa (genus to class), per head, rank and method, on `n_boot` random subsamples of `size` species
# (the same for every head, re-clustering each time). Cached (delete data/derived/alignment/ to
# recompute).
alignment_of <- \(study) {
  vocab <- vocabulary(study)
  prototypes <- cached(
    derived_path("alignment", str_c(study, "_prototypes.rds")),
    {
      message("Fetching the prototypes of ", study, " (slow the first time)")
      run <- \(head) str_c("efficientnet_v2_s_", head)
      prototypes_of <- \(head, path = FALSE) {
        head_prototypes(study, run(head), vocab, path)
      }
      conditional <- head_rows(study, run("conditional"), vocab)
      list(
        "flat" = prototypes_of("flat"),
        "hierarchical" = prototypes_of("hierarchical"),
        "independent" = prototypes_of("independent"),
        "conditional" = prototypes_from(conditional, vocab),
        "conditional (path sum)" = prototypes_from(
          conditional,
          vocab,
          path = TRUE
        )
      )
    }
  )
  cached(
    derived_path(
      "alignment",
      str_c(
        study,
        "_",
        n_boot,
        "_",
        size,
        "_",
        substr(digest(methods), 1, 8),
        ".rds"
      )
    ),
    {
      message(
        "Bootstrapping ",
        study,
        ": ",
        n_boot,
        " draws x ",
        length(prototypes),
        " heads"
      )
      set.seed(1)
      taxa <- vocab |> select(any_of(rank_keys[-1])) # genus to class (lepi: genus and family only)
      boot <- alignment_bootstrap(
        prototypes,
        taxa,
        n_boot = n_boot,
        size = size,
        methods = methods
      )
      list(
        observed = summarize(
          boot,
          ami = mean(ami),
          .by = c(head, method, rank)
        ),
        boot = boot
      )
    }
  )
}
results <- map(names(studies), alignment_of) |>
  set_names(names(studies))

# ---- Plots -------------------------------------------------------------------

plt_data <- results |>
  list_flatten(name_spec = "{outer}___{inner}") |>
  list_rbind(names_to = "id") |>
  separate_wider_delim(id, delim = "___", names = c("study", "type")) |>
  filter(type == "boot") |>
  mutate(
    diff = ami - ami[head == "flat"],
    .by = c(study, type, method, rank, b)
  ) |>
  filter(head != "flat") |>
  mutate(
    study = factor(studies[study], studies),
    rank = str_remove(rank, "Key") |>
      factor(c("genus", "family", "order", "class")) |>
      droplevels() |>
      fct_relabel(str_to_sentence),
    head = head |>
      factor(head_names) |>
      droplevels() |>
      fct_relabel(str_to_sentence),
    offset = (((as.integer(head) - 0.5) / length(levels(head))) - 0.5) * 2 / 3,
  )

x_limits <- quantile(plt_data$diff, c(0.025 / (4 * 6), 1 - 0.025 / (4 * 6)))

path_layers <- switch(
  path_layer,
  none = list(),
  lines = list(geom_path(
    aes(group = interaction(head, method, b)),
    linewidth = 0.05,
    alpha = 0.05
  )),
  density = {
    x_breaks <- x_bin *
      seq(floor(x_limits[[1]] / x_bin), ceiling(x_limits[[2]] / x_bin))
    n_x <- length(x_breaks) - 1
    sd_bins <- smooth_sd / (x_breaks[[2]] - x_breaks[[1]])
    spread <- tibble(o = -ceiling(3 * sd_bins):ceiling(3 * sd_bins)) |>
      mutate(
        weight = stats::dnorm(o, sd = sd_bins) /
          sum(stats::dnorm(o, sd = sd_bins))
      )
    head_rgb <- (grDevices::col2rgb(head_colours[levels(plt_data$head)]) /
      255)^2.2

    path_cells <- \(d) {
      y_grid <- y_bin * seq(ceiling(min(d$y) / y_bin), floor(max(d$y) / y_bin))
      split(d, list(d$head, d$method, d$b), drop = TRUE) |>
        map(
          \(p) {
            tibble(
              head = p$head[[1]],
              y = y_grid,
              x = stats::approx(p$y, p$diff, y_grid)$y
            )
          },
          .progress = T
        ) |>
        list_rbind() |>
        filter(!is.na(x), x >= x_limits[[1]], x <= x_limits[[2]]) |>
        mutate(col = findInterval(x, x_breaks, rightmost.closed = TRUE)) |>
        dplyr::count(head, y, col, name = "k") |>
        dplyr::cross_join(spread) |>
        mutate(col = col + o, k = k * weight) |>
        filter(col >= 1, col <= n_x) |>
        summarise(k = sum(k), .by = c(head, y, col))
    }

    density <- plt_data |>
      mutate(y = as.integer(rank) + offset) |>
      arrange(study, head, method, b, rank) |>
      (\(d) split(d, d$study))() |>
      map(path_cells) |>
      list_rbind(names_to = "study") |>
      mutate(study = factor(study, levels(plt_data$study))) |>
      summarise(
        r = sum(k * head_rgb[1, as.integer(head)]) / sum(k),
        g = sum(k * head_rgb[2, as.integer(head)]) / sum(k),
        b = sum(k * head_rgb[3, as.integer(head)]) / sum(k),
        k = sum(k),
        .by = c(study, y, col)
      ) |>
      mutate(
        fill = grDevices::rgb(r^(1 / 2.2), g^(1 / 2.2), b^(1 / 2.2)),
        alpha = 0.9 * pmin(1, k / quantile(k, 0.9)),
        x = (x_breaks[col] + x_breaks[col + 1]) / 2
      ) |>
      filter(alpha >= min_alpha)

    list(
      geom_raster(
        aes(x, y, fill = fill, alpha = alpha),
        data = density,
        inherit.aes = FALSE
      ),
      scale_fill_identity(),
      scale_alpha_identity()
    )
  }
)

# Median and the central 95% interval of the bootstrap draws (ggplot2's median_hilow needs Hmisc)
median_interval <- \(x) {
  tibble(
    y = stats::median(x),
    ymin = stats::quantile(x, 0.025, names = FALSE),
    ymax = stats::quantile(x, 0.975, names = FALSE)
  )
}

figure <- plt_data |>
  arrange(study, b, head, method, rank) |>
  ggplot(aes(diff, as.integer(rank) + offset, colour = head)) +
  path_layers +
  geom_vline(xintercept = 0, linetype = "dashed") +
  stat_summary(
    aes(group = after_stat(color)),
    geom = "line",
    fun = median,
    linetype = "dashed",
    orientation = "y",
  ) +
  stat_summary(
    aes(group = interaction(rank, head)),
    fun.data = median_interval,
    position = "identity",
    orientation = "y",
    fatten = 6,
    linewidth = 1.25
  ) +
  scale_x_continuous(
    limits = x_limits,
    oob = oob_keep
  ) +
  scale_y_continuous(
    transform = "reverse", # lowest rank (the first level) at the top, as in the other figures
    expand = expansion(0, 0.25),
    breaks = seq(length(levels(plt_data$rank))),
    labels = levels(plt_data$rank),
    minor_breaks = breaks_width(1, 0.5)
  ) +
  scale_color_manual(values = head_colours) +
  facet_wrap(vars(study), ncol = 1, scales = "free_y", space = "free_y") +
  labs(
    x = TeX("$\\Delta_{flat} \\, AMI$"),
    y = NULL,
    color = "Head"
  ) +
  guides(
    color = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  theme(
    legend.position = "bottom",
    axis.title.x = element_text(family = "Latin Modern Math"),
    axis.text.y = element_text(face = "bold", size = 16),
    panel.grid.minor.y = element_line(color = "gray75", linewidth = 0.33)
  )

save_figure(
  figure,
  "composite_taxonomic_alignment",
  width = 6,
  height = 6,
  scale = 1.5
)

# Values for the caption: \result[taxonomic_alignment, methods] and \result[taxonomic_alignment, smoothing_sd]
# Delta AMI per dataset, rank and head as in the figure (median and central 95% interval over the
# methods and subsamples), e.g. \result[taxonomic-alignment, plantnet, genus, hierarchical, median]
alignment_summary <- plt_data |>
  summarise(
    median = stats::median(diff),
    lower = stats::quantile(diff, 0.025, names = FALSE),
    upper = stats::quantile(diff, 0.975, names = FALSE),
    .by = c(study, rank, head)
  ) |>
  mutate(
    dataset = unname(cfg$study_dataset[names(studies)[match(study, studies)]]),
    rank = str_to_lower(rank),
    head = str_remove(str_replace_all(str_to_lower(head), "[^a-z]+", "_"), "_$") # conditional_path_sum
  )
print(alignment_summary, n = Inf)
alignment_values <- list()
for (i in seq_len(nrow(alignment_summary))) {
  r <- alignment_summary[i, ]
  alignment_values[[r$dataset]][[r$rank]][[r$head]] <- list(
    median = r$median,
    lower = r$lower,
    upper = r$upper
  )
}

do.call(save_results, c(
  list(
    "taxonomic_alignment",
    methods = join_and(unname(label_sets$cluster_method[methods])),
    subsamples = as.integer(n_boot),
    species_per_subsample = as.integer(size)
  ),
  alignment_values
))
