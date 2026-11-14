# Helpers for comparison tables built from the bootstrap replicate tables.

# Sizes of the runs of equal consecutive values of `x`, named by value: the `index` of
# pack_rows(), e.g. pack_rows(index = col_to_index(df$backbone)).
col_to_index <- function(x) {
  index <- diff(c(0, which(x != lead(x)), length(x)))
  names(index) <- x[cumsum(index)]
  index
}

# ---- Winner / loser summaries ---------------------------------------------------------------------

# Outcomes of compare_models() (helpers/compare_models.R) for the figures: wins up, losses down, the clear outcomes at the
# outer ends (stacking order of ggplot2: the first level is outermost, the last one next to zero)
win_levels <- c("Clear winner", "Winner", "Clear loser", "Loser")
win_colours <- c(
  "Clear winner" = "#1E9E4A",
  "Winner" = "#5DDE7F",
  "Loser" = "#F2777A",
  "Clear loser" = "#C4161C"
)
win_legend_breaks <- c("Clear winner", "Winner", "Loser", "Clear loser")

# Share of the rows of `df` (the output of compare_models(), stacked over comparisons) that are
# a clear winner, a plain winner, a clear loser and a plain loser, per group `by`; `n` rows.
outcome_shares <- function(df, by) {
  summarize(
    df,
    n = n(),
    "Clear winner" = mean(dominant),
    "Winner" = mean(tied & !dominant),
    "Clear loser" = mean(dominant_worst),
    "Loser" = mean(worst & !dominant_worst),
    .by = all_of(by)
  )
}

# outcome_shares() in long form for plotting: `outcome` (factor in stacking order) and `share`,
# negative for losses
win_loss_long <- function(shares) {
  shares |>
    pivot_longer(
      all_of(win_levels),
      names_to = "outcome",
      values_to = "share"
    ) |>
    mutate(
      outcome = factor(outcome, win_levels),
      share = if_else(outcome %in% c("Clear loser", "Loser"), -share, share)
    )
}

# Arrows of the win/loss axis title, pointing away from the zero line (towards losses on the
# negative side, towards wins on the positive side). With `arrow_font` (a font with arrow glyphs,
# see arrow_font in ggplot.R) they are glyphs centred in their slots; without it they are thin
# drawn arrows. `cross` is the position on the unsigned axis; `horizontal` says which axis is signed.
win_loss_arrows <- function(
  cross,
  horizontal,
  spacing,
  length,
  size,
  line_width,
  arrow_font = get("arrow_font", envir = globalenv())
) {
  side <- c(-1, 1)
  if (is.null(arrow_font)) {
    from <- side * spacing
    to <- side * (spacing + length)
    data <- if (horizontal) {
      tibble(x = from, xend = to, y = cross, yend = cross)
    } else {
      tibble(x = cross, xend = cross, y = from, yend = to)
    }
    geom_segment(
      data = data,
      aes(x = x, xend = xend, y = y, yend = yend),
      arrow = arrow(angle = 18, length = unit(0.07, "in"), type = "open"),
      linewidth = line_width, # thin, about the stem width of the font
      lineend = "butt"
    )
  } else {
    centre <- side * (spacing + length / 2)
    data <- if (horizontal) {
      tibble(x = centre, y = cross)
    } else {
      tibble(x = cross, y = centre)
    }
    data$label <- c("\u27F5", "\u27F6") # long left and right arrows
    geom_text(
      data = data,
      aes(x, y, label = label),
      family = arrow_font,
      size = size / ggplot2::.pt,
      angle = if (horizontal) 0 else 90 # rotated, they point down (losses) and up (wins)
    )
  }
}

# Legend and axis title of horizontal win/loss bars as one small plot, mirrored about x = 0 by
# construction: every item has a fixed slot at +-x, so nothing depends on text widths. Combined
# with the bars by patchwork (one panel in the column of the bars' panels), x = 0 lines up with the
# centre of the bars' panels, which is the zero line of the middle panel for three symmetric
# panels. Items run outwards from the zero line: the keys of the inner outcomes next to it, with
# their labels beyond; the arrows start at the zero line.
win_loss_key <- function(font = plot_font, size = 12, line_width = 0.3) {
  y_legend <- 0.5
  legend_key_height <- 0.7
  legend_key_width <- 0.05

  y_title <- 2.7
  title_vert_height <- 0.6
  lineheight <- 0.6
  arrow_length <- 0.1
  spacing <- 0.03

  slots <- tibble(
    outcome = c("Loser", "Winner", "Clear loser", "Clear winner"),
    side = c(-1, 1, -1, 1),
    distance = c(0.04, 0.04, 0.22, 0.25) # key centre, from the zero line
  ) |>
    mutate(
      outcome = factor(outcome, win_levels),
      x = side * distance,
      label_x = x + side * 0.045, # label beyond its key, away from the zero line
      hjust = (1 - side) / 2 # 1 for the left side, 0 for the right
    )
  text_size <- size / ggplot2::.pt

  ggplot(slots) +
    # Legend
    geom_tile(
      aes(x, y = y_legend, fill = outcome),
      width = legend_key_width,
      height = legend_key_height
    ) +
    geom_text(
      aes(label_x, y = y_legend, label = outcome, hjust = hjust),
      family = font,
      size = text_size
    ) +
    # Title
    win_loss_arrows(y_title, TRUE, spacing, arrow_length, size, line_width) +
    annotate(
      "segment",
      x = 0,
      xend = 0,
      y = y_title - title_vert_height / 2,
      yend = y_title + title_vert_height / 2,
      linewidth = line_width
    ) +
    geom_text(
      data = tibble(
        x = c(-(arrow_length + 2 * spacing), arrow_length + 2 * spacing),
        label = c("Losses", "Wins"),
        hjust = c(1, 0)
      ),
      aes(x, y = y_title, label = label, hjust = hjust),
      family = font,
      size = text_size
    ) +
    annotate(
      "text",
      x = 0,
      y = y_title - lineheight,
      label = "(share of comparisons)",
      family = font,
      size = text_size
    ) +
    # Layout and color
    scale_fill_manual(values = win_colours, guide = "none") +
    coord_cartesian(xlim = c(-1, 1), ylim = c(0, 3), expand = FALSE) +
    theme_void()
}

# Axis title of vertical win/loss bars ("Losses <- | -> Wins" above "(share of comparisons)",
# read bottom to top) as a small plot to the left of the bars, mirrored about y = 0 by construction
# like win_loss_key(): the arrows are segments that start at the zero line, and the words are
# anchored beyond them, so nothing depends on font glyphs or text widths. Combine with the bars by
# patchwork, e.g. wrap_plots(win_loss_title(), bars, widths = unit(c(0.7, 1), c("in", "null"))):
# its single panel has the height of the bars' panels, whose axis limits must be symmetric.
win_loss_title <- function(font = plot_font, size = 12, line_width = 0.3) {
  x_title <- 0.6 # first line (the outer one) ; the second line follows to its right
  lineheight <- 0.8
  tick_width <- 0.3
  arrow_length <- 0.25
  spacing <- 0.04
  text_size <- size / ggplot2::.pt

  ggplot() +
    win_loss_arrows(x_title, FALSE, spacing, arrow_length, size, line_width) +
    annotate(
      "segment",
      x = x_title - tick_width / 2,
      xend = x_title + tick_width / 2,
      y = 0,
      yend = 0,
      linewidth = line_width
    ) +
    geom_text(
      data = tibble(
        y = c(-(arrow_length + 2 * spacing), arrow_length + 2 * spacing),
        label = c("Losses", "Wins"),
        hjust = c(1, 0)
      ),
      aes(x = x_title, y = y, label = label, hjust = hjust),
      angle = 90,
      family = font,
      size = text_size
    ) +
    annotate(
      "text",
      x = x_title + lineheight,
      y = 0,
      label = "(share of comparisons)",
      angle = 90,
      family = font,
      size = text_size
    ) +
    coord_cartesian(xlim = c(0, 2), ylim = c(-1, 1), expand = FALSE) +
    theme_void()
}
