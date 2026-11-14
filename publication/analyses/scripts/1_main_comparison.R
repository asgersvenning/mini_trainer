source(here::here("shared.R"))

# Main comparison: one table per dataset and threshold setting with the heads of every backbone
# at every rank. Cells are estimate +- bootstrap SD (percent); bold marks every head that is a
# plausible winner ("contender": its win share, the share of resamples in which it is best,
# plausibly reaches the chance level 1/K for K heads) and underline the head that is the clear
# winner (all other heads are credibly below chance); see helpers/compare_models.R.
# Estimates and resamples come from the replicate tables on ERDA (replicate 0 = reporting
# partition, replicates >= 1 = bootstrap resamples of the test set).
# Writes results/tables/ (\resulttable{...}): main_comparison_<setting>.tex (main_backbone only,
# all datasets in one table, datasets as row groups) and appendix_comparison_<dataset>_<setting>.tex
# (the other backbones, backbones as row groups); and from the same win shares the win/loss summary:
# figures/main_comparison_wins_losses, values/main_comparison_win_loss and
# tables/appendix_win_loss.tex.

digits_for_win <- 3
# Winner: P(share >= 1/K) > alpha; clear winner: P(every rival below 1/K) > 1 - alpha (compare_models())
alpha <- 0.05
studies_used <- c("gefion-plantnet", "gefion-global-lepi", "gefion-flemming")
settings <- c("calibrated", "zero")
main_backbone <- "EfficientNetV2-Small" # tables in the main text; other backbones go to the appendix

# Metric columns (id = header) and their spanners above the header
table_metrics <- c(
  accuracy = "Accuracy",
  recall = "Recall",
  precision = "Precision",
  f1 = "F1",
  micro_accuracy = "$\\mu$Accuracy",
  theilU = "Theil's U",
  coverage = "Coverage"
)
metric_spanners <- c("Macro" = 4, "Micro" = 3) # after the Rank and Head columns

# ---- Data ---------------------------------------------------------------------------------

load_study <- function(study) {
  metric_ids <- names(table_metrics)
  models <- runs(study) |>
    collect() |>
    distinct(run_id, backbone, head)

  replicates(study) |>
    filter(setting %in% !!settings, metric %in% !!metric_ids) |>
    collect() |>
    inner_join(models, by = "run_id") |>
    mutate(level = rank_factor(level, .env$study), study = .env$study)
}

results <- map(studies_used, load_study) |>
  bind_rows() |>
  mutate(
    dataset_id = unname(cfg$study_dataset[study]),
    dataset = relabel(dataset_id, "dataset"),
    backbone = relabel(backbone, "backbone"),
    head = relabel(head, "head"),
    level = relabel(level, "level")
  )

# ---- Estimates, bootstrap SD and probability of being the best head -----------------------------

# Competing models are the heads of one backbone at one rank, per dataset, setting and metric
comparison <- results |>
  group_by(setting, dataset_id, dataset, backbone, level, metric) |>
  group_modify(\(g, key) {
    compare_models(
      rename(g, model = head),
      digits = digits_for_win,
      alpha = alpha
    )
  }) |>
  ungroup() |>
  mutate(head = factor(model, levels(results$head)))

# Check: how far replicate 0 (the estimate) is from the centre of the bootstrap resamples, in
# bootstrap SDs. Large values mean the partition estimate is not typical of the resamples.
comparison |>
  summarize(
    median_abs_z = median(abs(boot_mean - est) / sd),
    .by = c(setting, metric)
  ) |>
  print(n = Inf)

# ---- Tables -------------------------------------------------------------------------------------

# One-decimal number, \textbf{} where `bold` and additionally \underline{} where `underline`
fmt_bold <- function(x, bold, underline) {
  s <- sprintf("%.1f", x)
  s <- if_else(bold, str_c("\\textbf{", s, "}"), s)
  if_else(underline, str_c("\\underline{", s, "}"), s)
}

# LaTeX tabular (booktabs) of the comparison and its caption, as list(table, caption); the manuscript
# wraps the table in a float. Written directly rather than with
# kableExtra so that the side labels (\multirow), the rules between blocks and the cell layout
# are fully controlled.
# `outer` is the column (dataset or backbone) of the rotated side labels; ranks are nested within
# it. `scope` describes the table's subject in the caption.
make_table <- function(df, setting, outer, scope) {
  metric_ids <- names(table_metrics)
  n_col <- 3 + length(metric_ids) # side label, rank, head, metrics
  rule <- str_c("\\cmidrule(lr){2-", n_col, "}")

  wide <- df |>
    arrange(.data[[outer]], level, head) |>
    mutate(
      # Fixed-width boxes (right-aligned estimate, left-aligned SD) so that the digits and the
      # \pm line up whether or not the cell is bold (bold digits are wider)
      cell = str_c(
        "\\makebox[2.2em][r]{",
        fmt_bold(100 * est, tied, dominant),
        "}$\\,\\pm\\,$",
        "\\makebox[1.6em][l]{",
        fmt_bold(100 * sd, tied, dominant),
        "}"
      )
    ) |>
    select(all_of(outer), level, head, metric, cell) |>
    pivot_wider(names_from = metric, values_from = cell)
  wide$cells <- do.call(
    str_c,
    c(unname(as.list(wide[metric_ids])), sep = " & ")
  )

  # The outer label is a rotated \multirow in the first row of each outer block and the rank a
  # \multirow in the first row of each rank block; a rule follows every rank block
  rows <- wide |>
    mutate(
      rank_first = row_number() == 1,
      rank_last = row_number() == n(),
      rank_size = n(),
      .by = c(all_of(outer), level)
    ) |>
    mutate(
      outer_first = row_number() == 1,
      outer_size = n(),
      outer_last = row_number() == n(),
      .by = all_of(outer)
    ) |>
    mutate(
      row = str_c(
        if_else(
          outer_first,
          str_c(
            "\\multirow{",
            outer_size,
            "}{*}{\\rotatebox[origin=c]{90}{\\textbf{",
            .data[[outer]],
            "}}}"
          ),
          ""
        ),
        " & ",
        if_else(
          rank_first,
          str_c(
            "\\multirow{",
            rank_size,
            "}{*}{\\textit{",
            as.character(level),
            "}}"
          ),
          ""
        ),
        " & ",
        as.character(head),
        " & ",
        cells,
        "\\\\",
        # rank rule; full-width rule after an outer block; nothing after the last row
        case_when(
          row_number() == n() ~ "",
          outer_last ~ "\n\\midrule",
          rank_last ~ str_c("\n", rule),
          .default = ""
        )
      )
    )

  body <- paste(rows$row, collapse = "\n")

  spans <- metric_spanners
  ends <- 3 + cumsum(spans)
  header <- c(
    str_c(
      "\\multicolumn{3}{c}{ } & ",
      paste0(
        "\\multicolumn{",
        spans,
        "}{c}{",
        names(spans),
        "}",
        collapse = " & "
      ),
      " \\\\"
    ),
    paste0(
      "\\cmidrule(l{3pt}r{3pt}){",
      ends - spans + 1,
      "-",
      ends,
      "}",
      collapse = " "
    ),
    str_c(" & Rank & Head & ", paste(table_metrics, collapse = " & "), "\\\\")
  )

  # Resamples per dataset, e.g. "1000 (Pl@ntNet300K) or 200 (Global Lepidoptera)" when they differ
  # (the appendix tables are one dataset each and have no `dataset` column)
  n_boot_text <- if (n_distinct(df$n_boot) == 1) {
    df$n_boot[[1]]
  } else {
    n_boot <- df |>
      distinct(dataset, n_boot) |>
      summarise(datasets = join_and(as.character(sort(unique(dataset)))), .by = n_boot) |>
      arrange(desc(n_boot))
    str_c(n_boot$n_boot, " (", n_boot$datasets, ")", collapse = " or ")
  }

  caption <- str_c(
    "Test metrics for {scope} ({setting} thresholds), ",
    "in percent: estimate on the reporting partition (90\\% of the test set) $\\pm$ ",
    "standard deviation over {n_boot_text} bootstrap resamples. ",
    "Bold: the head is a plausible winner, i.e. its share of resamples in which it is best ",
    "plausibly reaches chance ($1/K$ for $K$ heads; posterior probability above {100 * alpha}\\%). ",
    "Underlined: clear winner, i.e. every other head is below chance with posterior probability ",
    "above {100 * (1 - alpha)}\\%. The probabilities concern the resampling shares given the fixed test data and ",
    "predictions. ",
    "Wins are decided on metrics rounded to {100 / 10^digits_for_win} percentage points. ",
    "Compared within each {if (outer == 'dataset') 'dataset' else 'backbone'} and rank."
  ) |>
    str_glue()

  table <- str_c(
    c(
      "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\\renewcommand{\\arraystretch}{0.9}",
      str_c("\\begin{tabular}{cll", strrep("c", length(metric_ids)), "}"),
      "\\toprule",
      header,
      "\\midrule",
      body,
      "\\bottomrule",
      "\\end{tabular}",
      "\\endgroup"
    ),
    collapse = "\n"
  )
  list(table = table, caption = caption)
}

# Main text: one table per setting, main backbone only, datasets as row groups
main_tables <- comparison |>
  filter(backbone == main_backbone) |>
  nest(.by = setting) |>
  mutate(
    table = map2(
      data,
      setting,
      make_table,
      outer = "dataset",
      scope = main_backbone
    ),
    name = str_c("main_comparison_", setting)
  )
walk2(main_tables$table, main_tables$name, \(x, name) {
  save_table(x$table, name, x$caption)
})

# Appendix: one table per dataset and setting, the other backbones as row groups
appendix_tables <- comparison |>
  filter(backbone != main_backbone) |>
  nest(.by = c(dataset_id, dataset, setting)) |>
  mutate(
    table = pmap(
      list(data, setting, dataset),
      \(data, setting, dataset) {
        make_table(data, setting, "backbone", as.character(dataset))
      }
    ),
    name = str_c("appendix_comparison_", dataset_id, "_", setting)
  )
walk2(appendix_tables$table, appendix_tables$name, \(x, name) {
  save_table(x$table, name, x$caption)
})

# ---- Figure: how often each head wins and loses at the species, genus and family rank -------------

# Share of comparisons (dataset x backbone x setting x metric, per rank) in which the head is a
# winner / loser (contender) and a clear winner / loser (the only head not credibly below chance); wins up, losses down, the clear outcomes at the
# outer ends. Coverage is excluded.
rank_data <- comparison |>
  filter(metric != "coverage", level %in% c("Species", "Genus", "Family")) |>
  mutate(level = as.character(level))

# Per head and rank, plus the mean over the ranks ("Mean": the three ranks have the same number
# of comparisons, so this is the pooled share)
shares <- bind_rows(
  outcome_shares(rank_data, c("head", "level")),
  outcome_shares(rank_data, "head") |> mutate(level = "Mean", n = n / 3)
) |>
  mutate(level = factor(level, c("Species", "Genus", "Family", "Mean"))) |>
  arrange(head, level)

# The figure shows the three ranks only; the means are in the appendix table and the macros
win_loss <- shares |>
  filter(level != "Mean") |>
  mutate(level = droplevels(level)) |>
  win_loss_long()

win_loss_plot <- win_loss |>
  ggplot(aes(level, share, fill = outcome)) +
  geom_col(width = 0.8) +
  geom_hline(yintercept = 0, linetype = "solid") +
  scale_fill_manual(values = win_colours, breaks = win_legend_breaks) +
  scale_y_continuous(
    breaks = seq(-1, 1, 0.1),
    labels = \(x) label_percent()(abs(x)),
    limits = ~ c(-1, 1) * max(abs(.))
  ) +
  facet_grid(~head, switch = "x") +
  labs(x = NULL, y = NULL, fill = NULL) + # axis title: win_loss_title()
  theme(
    axis.text = element_text(size = rel(0.7)),
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    strip.placement = "outside",
    strip.text = element_text(size = rel(0.75))
  )

# The axis title is a separate plot to the left of the bars, centred on their zero line
save_figure(
  wrap_plots(
    win_loss_title(),
    win_loss_plot,
    widths = unit(c(0.7, 1), c("in", "null"))
  ),
  "main_comparison_wins_losses",
  width = 8.5,
  height = 3.5
)

# ---- Numbers for the manuscript: macros and an appendix table -------------------------------------

# \result[main_comparison_win_loss, <head>, <rank>, <wins|clear_winner|...>], in percent
win_loss_values <- shares |>
  mutate(
    wins = `Clear winner` + Winner,
    losses = `Clear loser` + Loser,
    clear_winner = `Clear winner`,
    clear_loser = `Clear loser`,
    across(c(wins, losses, clear_winner, clear_loser, Winner, Loser), \(x) {
      100 * x
    }),
    winner = Winner,
    loser = Loser
  )
win_loss_list <- win_loss_values |>
  split(str_to_lower(win_loss_values$head)) |>
  map(\(h) {
    split(h, str_to_lower(h$level)) |>
      map(\(r) {
        as.list(r[c(
          "wins",
          "clear_winner",
          "winner",
          "losses",
          "clear_loser",
          "loser"
        )])
      })
  })
do.call(save_results, c(list("main_comparison_win_loss"), win_loss_list))

win_loss_table <- win_loss_values |>
  transmute(
    head,
    Rank = as.character(level),
    Wins = wins,
    `of which clear` = clear_winner,
    Losses = losses,
    `of which clear ` = clear_loser # distinct from the first, as column names must differ
  )
win_loss_caption <- str_c(
  "Share (\\%) of comparisons in which each head is a winner or a loser, and a clear ",
  "winner or loser (see Methods), at the species, genus and family rank. ",
  "A comparison is one dataset, backbone, setting and metric (n = ",
  max(shares$n),
  " per rank, excluding coverage). Mean: average over the three ranks."
)
win_loss_table |>
  select(!head) |>
  kable(
    "latex",
    escape = FALSE,
    booktabs = TRUE,
    digits = 1,
    linesep = "",
    col.names = c("Rank", "Wins", "of which clear", "Losses", "of which clear")
  ) |>
  pack_rows(index = col_to_index(win_loss_table$head)) |>
  add_header_above(c(" " = 1, "Winner" = 2, "Loser" = 2)) |>
  save_table("appendix_win_loss", caption = win_loss_caption)
