# Per-image correctness and confidence of the saved predictions (replicates folder), for the
# calibration scripts. Depends on the ERDA accessors (data.R) and `vocabulary` (embeddings/heads.R).

# The key of every test image's true taxon at each rank of a study: one row per image and level
# (level k is rank ranks[k + 1]); `ranks` are the rank names, species upwards
calibration_truth <- function(study) {
  ranks <- str_remove(resampling(study)$ranks, "Key")
  vocab <- vocabulary(study)
  image_class <- images(study) |>
    select(image_id, class_id) |>
    collect()
  map(seq_along(ranks), \(k) {
    tibble(
      image_id = image_class$image_id,
      level = k - 1,
      truth = as.character(vocab[[str_c(ranks[k], "Key")]][match(image_class$class_id, vocab$class_id)])
    )
  }) |>
    list_rbind() |>
    mutate(rank = ranks[level + 1])
}

# One row per image and rank of one run: whether the predicted taxon is the true one, and the
# confidence of the prediction. `truth`: calibration_truth() of the study.
run_calibration <- function(study, run_id, truth) {
  predicted <- collect(predictions(study, run_id))
  stopifnot(c("image_id", "level", "prediction", "confidence") %in% names(predicted))
  predicted |>
    inner_join(truth, by = c("image_id", "level")) |>
    transmute(
      image_id,
      rank,
      correct = as.character(prediction) == truth,
      confidence
    )
}

# Equal-width confidence bin (1 to n_bins) of confidences in [0, 1]
confidence_bin <- function(confidence, n_bins) {
  pmin(floor(confidence * n_bins), n_bins - 1) + 1
}

# The head and the backbone of a run id (<backbone>_<head>)
run_head <- \(run_id) str_extract(run_id, "[a-z]+$")
run_backbone <- \(run_id) str_remove(run_id, "_[a-z]+$")
