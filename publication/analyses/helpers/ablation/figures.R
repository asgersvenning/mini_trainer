# Building blocks of the ablation figures that put one row per configuration next to a dot matrix of
# its components: the shared row axis and the dot matrix. Depends on replicates.R (component_levels).

# Row axis shared by the two panels: the configurations in `configs` order (first at the top).
# Using the same scale object for both guarantees that their rows line up.
scale_y_config <- function(configs) {
  scale_y_discrete(limits = rev(configs), expand = expansion(add = 0.6))
}

# Dot matrix: one row per configuration, one column (facet) per component, named by its rotated
# strip right above the dots; dark dot = on, light dot = off ("standard" has everything off).
plot_model_matrix <- function(configs, components = component_levels) {
  expand_grid(model = configs, component = components) |>
    mutate(
      on = mapply(
        \(m, k) k %in% strsplit(m, " × ", fixed = TRUE)[[1]],
        model,
        component,
        USE.NAMES = FALSE
      ),
      component = factor(component, components)
    ) |>
    ggplot(aes(x = "", y = model, colour = on)) +
    geom_point(size = 2.5) +
    scale_colour_manual(
      values = c(`TRUE` = "grey20", `FALSE` = "grey85"),
      guide = "none"
    ) +
    scale_y_config(configs) +
    facet_grid(cols = vars(component)) +
    labs(x = NULL, y = NULL) +
    theme(
      strip.text.x = element_text(
        face = "plain",
        angle = 90,
        hjust = 0,
        vjust = 0.5,
        size = rel(0.8)
      ),
      strip.background = element_blank(),
      panel.spacing.x = unit(0, "pt"),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.line = element_blank()
    )
}
