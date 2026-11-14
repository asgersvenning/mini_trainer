# Design & Idea: Asger Svenning
# 09-11-2026
# Written by Claude Sonnet 5.5 based on a similar R script used for the publication
# "A General Method for Detection and Segmentation of Terrestrial Arthropods",
# Svenning et al. 2026, Methods in Ecology and Evolution
#

# Runs every analysis script in reading order and writes everything to results/. From the project
# root: `Rscript run_all.R`, or `source("run_all.R")` in R. Set the options below first.

#################### User Control ####################

# Delete the caches of the (very expensive) computations in `recompute_caches` before running
do_recompute <- FALSE

# Do not run any scripts (the steps and the caches that would be deleted are only listed)
dry_run <- FALSE

# Run the species-photo candidates and screening (needs a GEMINI_API_KEY, and you pick the photos
# by hand; see README). The thumbnails and photo credits are always made from the current picks
do_photos <- FALSE

# Scripts to leave out: file names without extension, or groups (main_comparison, ablation,
# embedding, photos, calibration)
skip <- character()

# Stop at the first script that fails; otherwise go on and list the failures at the end
stop_on_error <- TRUE

# Show what the scripts print (tables of numbers, diagnostics)
verbose <- TRUE

# Folders of data/derived/ cleared by `do_recompute` (the downloaded data, the vocabulary and
# taxon names, the photo pipeline and the manual picks are never cleared)
recompute_caches <- c(
  "alignment",
  "calibration",
  "families",
  "species_points"
)

#######################################################

library(progressr)

handlers(global = TRUE)
handlers(handler_pbcol(
  adjust = 1.0,
  complete = function(s) cli::bg_blue(cli::col_black(s)),
  incomplete = function(s) cli::bg_none(cli::col_white(s)),
))

steps <- data.frame(
  group = c(
    "main_comparison",
    rep("ablation", 3),
    "embedding",
    rep("photos", 2),
    "embedding", # species_thumbnails: only fetches what is missing, and writes the photo credits
    "embedding",
    "embedding",
    rep("calibration", 2)
  ),
  script = c(
    "1_main_comparison",
    "2_ablation/hierarchy_contrast",
    "2_ablation/ablation_win_loss",
    "2_ablation/ablation_vs_standard",
    "3_embedding/taxonomic_alignment",
    "3_embedding/prep/species_candidates",
    "3_embedding/prep/species_screening",
    "3_embedding/prep/species_thumbnails",
    "3_embedding/outlyingness",
    "3_embedding/embedding_structure",
    "4_calibration/calibration_curves",
    "4_calibration/calibration_vs_flat"
  )
)
steps$name <- basename(steps$script)
steps <- steps[
  !(steps$group == "photos" & !do_photos) &
    !(steps$name %in% skip) &
    !(steps$group %in% skip),
]

reset_dev <- function() {
  while (
    tryCatch(
      {
        dev.off()
        TRUE
      },
      error = function(...) FALSE
    )
  ) {}
}

# Each script runs in its own environment (its variables do not leak into the next one); the
# helpers are sourced by the script itself (shared.R)
run_script <- function(script) {
  file <- here::here("scripts", paste0(script, ".R"))
  run <- \() source(file, local = new.env(parent = globalenv()))
  if (verbose) {
    run()
  } else {
    suppressMessages(utils::capture.output(run()))
  }
  invisible()
}

main <- function() {
  pb <- progressr::progressor(steps = nrow(steps) + 1, auto_finish = FALSE)

  pb("Preparing...", amount = 0)
  caches <- file.path(here::here("data", "derived"), recompute_caches)
  if (do_recompute) {
    for (cache in caches[dir.exists(caches)]) {
      message("Clearing ", cache)
      if (!dry_run) unlink(cache, recursive = TRUE)
    }
  }
  pb()

  status <- rep("ok", nrow(steps))
  seconds <- numeric(nrow(steps))
  for (i in seq_len(nrow(steps))) {
    pb(sprintf("%s (%d/%d)...", steps$name[[i]], i, nrow(steps)), amount = 0)
    if (!dry_run) {
      start <- proc.time()[["elapsed"]]
      result <- tryCatch(
        {
          run_script(steps$script[[i]])
          NULL
        },
        error = function(e) conditionMessage(e)
      )
      seconds[[i]] <- proc.time()[["elapsed"]] - start
      reset_dev()
      if (!is.null(result)) {
        status[[i]] <- paste("FAILED:", result)
        if (stop_on_error) {
          stop(steps$name[[i]], " failed: ", result, call. = FALSE)
        }
      }
    } else {
      status[[i]] <- "dry run"
    }
    pb()
  }

  summary <- data.frame(
    script = steps$script,
    seconds = round(seconds),
    status = status
  )
  print(summary, row.names = FALSE)
  if (any(startsWith(status, "FAILED"))) {
    warning(
      sum(startsWith(status, "FAILED")),
      " script(s) failed",
      call. = FALSE
    )
  }
  invisible(summary)
}

main()
