# Use CMU Serif if it is available to extrafont; otherwise fall back to "serif".
plot_font <- if ("CMU Serif" %in% fonts()) {
  "CMU Serif"
} else {
  warning(
    "Font 'CMU Serif' not found (see extrafont::font_import()); using 'serif'.",
    call. = FALSE
  )
  "serif"
}

theme_set(
  theme_pubr(legend = "right", base_family = plot_font) +
    theme(
      title = element_text(face = "bold", size = 16),
      strip.text = element_text(face = "bold", size = 16),
      plot.title = element_text(face = "plain", size = 20, hjust = 0.5),
      legend.title = element_text(hjust = 0.5),
      legend.text = element_text(size = 14)
    )
)

# Colours of the models (keyed by the display labels of `label_sets$head`, plus the conditional
# head's path sum), the same in every figure whichever heads it shows. Colourblind-safe (Okabe-Ito
# shades), without red and green (the win/loss colours) or the genus palette's hues. Edit here only.
head_colours <- c(
  "Flat" = "#7F7F7F",
  "Hierarchical" = "#E69F00",
  "Independent" = "#CC79A7",
  "Conditional" = "#0072B2",
  "Conditional (path sum)" = "#56B4E9",
  "Autoregressive" = "#000000"
)

# Font for the arrow glyphs of the win/loss axis titles (CMU Serif has none): Latin Modern Math,
# which has the arrows of the LaTeX math font. Installed on the system, or registered here from
# `cfg$math_font_file` (data/fonts/latinmodern-math.ttf).
# NULL if unavailable: the titles then use drawn arrows.
arrow_font <- local({
  found <- \() {
    "Latin Modern Math" %in% fonts() ||
      identical(font_info("Latin Modern Math")$family, "Latin Modern Math")
  }
  file <- cfg[["math_font_file"]]
  # registered only if not installed (or registered by an earlier script in the session)
  if (!found() && !is.null(file) && file.exists(file)) {
    register_font("Latin Modern Math", plain = file)
  }
  if (found()) "Latin Modern Math" else NULL
})
