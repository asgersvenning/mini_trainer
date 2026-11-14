# Required prerequisite for every script and runnable document in the project.
# Usage, from any script at any depth:
#   source(here::here("shared.R"))   # `here::` is the only unavoidable bootstrap

if (!requireNamespace("box", quietly = TRUE)) {
  stop(
    "Package 'box' is required but not installed. ",
    "Install it with install.packages(\"box\") and re-run.",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  box::use(
    here[here]
  )
})
source(here("helpers", "main.R"))

set.seed(cfg$seed)
