# The manual picks of the species photos and the photos they name. Depends on inaturalist.R.

# The picks of a study: one row per species with its chosen photo, from
# data/species_picks/<study>.csv (written by the export of the review sheet). Columns: speciesKey,
# name, photo_id (an iNaturalist photo: a candidate, or any other photo of the species' taxon),
# rotate (degrees clockwise), omit (TRUE: no photo of this species) and comment.
picks_path <- \(study) here("data", "species_picks", str_c(study, ".csv"))
read_picks <- \(study) {
  if (!file.exists(picks_path(study))) {
    return(NULL)
  }
  read_csv(
    picks_path(study),
    col_types = cols(
      speciesKey = col_double(),
      photo_id = col_double(),
      rotate = col_double(),
      omit = col_logical(),
      .default = col_character()
    )
  )
}

# The chosen photo of every species that has a pick, with its licence, attribution and source link:
# from the candidates, or looked up on iNaturalist if the photo is not one. Picks that fail (a photo
# that is not of the species) are reported and left out.
chosen_photos <- function(picks, candidates) {
  picks |>
    filter(!omit, !is.na(photo_id)) |>
    pmap(\(speciesKey, name, photo_id, rotate, ...) {
      own <- filter(candidates, speciesKey == !!speciesKey, photo_id == !!photo_id)
      photo <- if (nrow(own) > 0) {
        select(own[1, ], photo_id, image_url, licence, attribution, user, observation_url)
      } else {
        taxon_id <- unique(na.omit(candidates$taxon_id[candidates$speciesKey == speciesKey]))
        tryCatch(
          inaturalist_photo(photo_id, taxon_id),
          error = \(e) {
            message("PICK TO REDO, ", name, ": ", conditionMessage(e))
            NULL
          }
        )
      }
      if (!is.null(photo)) bind_cols(tibble(speciesKey = speciesKey, name = name, rotate = rotate), photo)
    }) |>
    list_rbind()
}
