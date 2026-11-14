# Export results for the LaTeX manuscript. Depends on `cfg`, `here`, `result_path` (io.R).
#
# The whole `results/` folder is the Overleaf payload (upload it as-is):
#   results/setup.tex                   copied from latex/setup.tex on every export
#   results/figures/<name>.pdf          save_figure()    \resultgraphic{name}
#   figure_previews/<name>.png          save_figure()    quick viewing only; untracked, not part of results/
#   results/tables/<name>.tex           save_table()     \resulttable{name} (a tabular; caption: \resultcaption{name})
#   results/values/<name>.tex           save_results()   \result[name, key, ...]
#   results/values/index.tex            inputs every values file (kept up to date)
#   results/sections/<name>.tex         save_section()  \input{results/sections/name} (a whole section)
# Manuscript preamble: \input{results/setup}. Every file is one result, so any
# script can regenerate its outputs independently.

# Directory for one kind of export; keeps the LaTeX support file in place.
export_dir <- function(kind) {
  file.copy(
    here("latex", "setup.tex"),
    ensure_dir(result_path()),
    overwrite = TRUE
  )
  ensure_dir(result_path(kind))
}

# Save a ggplot as vector PDF (results/figures/, for the manuscript) and PNG (figure_previews/, for quick
# viewing; not tracked).
save_figure <- function(
  plot,
  name,
  width = cfg$fig_width,
  height = cfg$fig_height,
  ...
) {
  dir <- export_dir("figures")
  ggsave(
    file.path(dir, paste0(name, ".pdf")),
    plot,
    device = grDevices::cairo_pdf,
    width = width,
    height = height,
    ...
  )
  ggsave(
    file.path(ensure_dir(here("figure_previews")), paste0(name, ".png")),
    plot,
    dpi = cfg$fig_dpi,
    width = width,
    height = height,
    ...
  )
  invisible(plot)
}

# Save a LaTeX table (a `tabular`, without table environment or caption, e.g. kable(format = "latex")
# without a caption) as a .tex fragment. The manuscript wraps it, with its own placement, caption and
# label:
#   \begin{table}[htb] \centering \resulttable{name} \caption{...} \label{tab:name} \end{table}
# `caption`: a generated caption (with the numbers of the analysis) to use or adapt there, saved as a
# value: \resultcaption{name}.
save_table <- function(table, name, caption = NULL) {
  writeLines(
    as.character(table),
    file.path(export_dir("tables"), paste0(name, ".tex"))
  )
  if (!is.null(caption)) {
    save_results(str_c("caption_", name), text = caption)
  }
  invisible(table)
}

# Save (nested) values for retrieval in LaTeX through a single accessor, e.g.
#   save_results("experiment_1", accuracy = list(flat = 46.45, independent = 46.06), n = 987L)
# writes, to results/values/experiment_1.tex,
#   \setresult{experiment_1/accuracy/flat}{46.45}  ...
# retrieved with \result[experiment_1, accuracy, flat]. `name` is the first key, so
# scripts cannot collide. Doubles use `cfg$digits` decimals; strings are written
# verbatim, so they can contain LaTeX ("46.5\\%").
save_results <- function(name, ...) {
  values <- flatten_results(list(...), name)
  lines <- sprintf(
    "\\setresult{%s}{%s}",
    names(values),
    vapply(values, tex_value, character(1))
  )
  dir <- export_dir("values")
  writeLines(
    c("% Generated; do not edit.", lines),
    file.path(dir, paste0(name, ".tex"))
  )
  # Index of all values files, input by results/setup.tex.
  files <- sort(setdiff(
    tools::file_path_sans_ext(list.files(dir, pattern = "\\.tex$")),
    "index"
  ))
  writeLines(
    sprintf("\\input{results/values/%s}", files),
    file.path(dir, "index.tex")
  )
  invisible(values)
}

# Save lines of LaTeX as a section file to include as it is (results/sections/<name>.tex),
# e.g. an appendix section with its \\section and \\label.
save_section <- function(lines, name) {
  writeLines(
    c("% Generated; do not edit.", lines),
    file.path(export_dir("sections"), paste0(name, ".tex"))
  )
  invisible(lines)
}

# Nested list -> flat list named by "key1/key2/...".
flatten_results <- function(x, prefix) {
  if (!is.list(x)) {
    return(setNames(list(x), paste(prefix, collapse = "/")))
  }
  keys <- names(x)
  if (is.null(keys) || any(!grepl("^[A-Za-z0-9_.-]+$", keys))) {
    stop(
      "Result keys must be named, using only letters, digits, '_', '.' and '-'.",
      call. = FALSE
    )
  }
  unlist(
    unname(Map(
      function(value, key) flatten_results(value, c(prefix, key)),
      x,
      keys
    )),
    recursive = FALSE
  )
}

tex_value <- function(x) {
  if (length(x) != 1) {
    stop("Result values must be length 1.", call. = FALSE)
  }
  if (is.double(x)) {
    formatC(x, format = "f", digits = cfg$digits)
  } else {
    as.character(x)
  }
}

# Text for LaTeX captions: "a, b and c", and \textit{x}
join_and <- \(x) {
  if (length(x) < 2) {
    x
  } else {
    str_c(str_c(x[-length(x)], collapse = ", "), " and ", x[length(x)])
  }
}
italic <- \(x) str_c("\\textit{", x, "}")

# Text for LaTeX: the special characters of `x` escaped (backslashes are not handled)
tex_escape <- \(x) {
  str_replace_all(x, c(
    "&" = "\\\\&", "%" = "\\\\%", "\\$" = "\\\\\\$", "#" = "\\\\#", "_" = "\\\\_",
    "\\{" = "\\\\{", "\\}" = "\\\\}", "~" = "\\\\textasciitilde{}",
    "\\^" = "\\\\textasciicircum{}"
  ))
}
