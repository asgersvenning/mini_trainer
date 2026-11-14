source(here::here("shared.R"))

# Does the hierarchical head have the same effect in the architecture comparison and in the ablations?
# Per test image the contrast is the species-level error of the hierarchical model minus that of the flat
# model (EfficientNetV2-S, full recipe, Pl@ntNet300K; the test set is the same in both studies, see
# docs/studies.md): 1 only the hierarchical model errs, -1 only the flat one, 0 both or neither. Its mean
# is the effect of the hierarchy (in accuracy, negative: the hierarchical model is worse). The result is
# the effect in each run (the architecture comparison and each ablation seed), and the difference
# between the architecture comparison and the mean over the ablation seeds, with a 95% interval over the
# images.
architecture <- list(
  study = "gefion-plantnet",
  flat = "efficientnet_v2_s_flat",
  hierarchical = "efficientnet_v2_s_hierarchical"
)
ablation <- list(
  study = "plantnet",
  flat = "full",
  hierarchical = "hierarchy_regularized" # variants; run ids are <variant>_seed<seed>
)

# Species-level error (0/1) per image of one run. `prediction` is the predicted taxon key, compared with
# the species key of the image's class.
species_errors <- function(study, run_id) {
  level <- which(str_remove(resampling(study)$ranks, "Key") == "species") - 1
  truth <- images(study) |>
    select(image_id, class_id) |>
    collect() |>
    left_join(select(vocabulary(study), class_id, speciesKey), by = "class_id")
  predictions(study, run_id) |>
    filter(level == !!level) |>
    select(image_id, prediction) |>
    collect() |>
    inner_join(truth, by = "image_id") |>
    transmute(image_id, error = as.numeric(as.character(prediction) != as.character(speciesKey)))
}

contrast <- function(study, flat_id, hierarchical_id) {
  inner_join(
    species_errors(study, flat_id),
    species_errors(study, hierarchical_id),
    by = "image_id",
    suffix = c("_flat", "_hierarchical")
  ) |>
    transmute(image_id, contrast = error_hierarchical - error_flat)
}

seeds <- runs(ablation$study) |>
  pull(run_id) |>
  str_extract("seed\\d+") |>
  unique() |>
  stats::na.omit()
effects <- c(
  list(architecture = contrast(architecture$study, architecture$flat, architecture$hierarchical)),
  map(set_names(seeds), \(seed) {
    contrast(ablation$study, str_c(ablation$flat, "_", seed), str_c(ablation$hierarchical, "_", seed))
  })
) |>
  imap(\(x, name) set_names(x, c("image_id", name))) |>
  reduce(inner_join, by = "image_id")

difference <- effects$architecture - rowMeans(effects[seeds])
test <- stats::t.test(difference)
effect_points <- -100 * colMeans(select(effects, -image_id))
cat("\nEffect of the hierarchy on accuracy (percentage points):\n")
print(round(effect_points, 2))
cat(
  "Architecture comparison minus mean of the ablation seeds:", round(-100 * test$estimate, 2),
  "(95% interval", round(-100 * test$conf.int[[2]], 2), "to", round(-100 * test$conf.int[[1]], 2), ")\n"
)

save_results(
  "hierarchy_contrast",
  images = nrow(effects),
  effect = as.list(effect_points),
  difference = list(
    estimate = -100 * unname(test$estimate),
    lower = -100 * test$conf.int[[2]],
    upper = -100 * test$conf.int[[1]]
  )
)
