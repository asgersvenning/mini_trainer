# Embedding structure figures: polar coordinates of the images and class means, and the text snippets
# that go with the figures. Depends on family.R and projection.R.

# Polar coordinates of the drawn images and the class means of one dataset, all heads stacked.
# `runs`: family_runs() of the dataset; `wrong`: keys of the species with known wrong labels (not
# used for the plane or the class means); `reference_head`: the head whose class-mean configuration the
# others are rotated to (and whose test images are drawn); `weighting` ("species" or "genus") and
# `robust`: the plane fit; `class_mean_method` and `class_mean_k`: where a species is drawn, see
# class_means() and the settings in embedding_structure.R; `split_shown` and `cap`: the images drawn
# (at most `cap` per species of that split, the same ones in every head).
polar_layers <- function(
  runs,
  wrong,
  reference_head,
  weighting,
  robust,
  class_mean_method,
  class_mean_k,
  split_shown,
  cap
) {
  valid <- map(runs, without_species, wrong)
  genus <- valid[[reference_head]]$species$genusKey
  plane_weights <- switch(
    weighting,
    species = rep(1, length(genus)),
    genus = balanced_weights(genus)
  )
  # The plane is fitted on the plain class means (noise averaged out) of the valid species, while the
  # positions drawn are those of `class_mean_method`, for all species per head, the wrong ones included
  is_wrong <- map(runs, \(d) {
    as.character(d$species$speciesKey) %in% as.character(wrong)
  })
  species_positions <- \(d, fit, R) {
    if (class_mean_method != "polar") {
      return(polar_placement(
        fit,
        class_means(d, class_mean_method, class_mean_k),
        R
      ))
    }
    # "polar": the median angle to the centre of the species' images, in the direction of their mean
    # projected vector (the centre of an arc of images, not the mean of its points)
    tibble(speciesKey = d$images$speciesKey, polar_placement(fit, d$embeddings, R)) |>
      summarise(
        theta = median(theta),
        angle = atan2(sum(in_plane * sin(angle)), sum(in_plane * cos(angle))),
        .by = speciesKey
      ) |>
      slice(match(d$species$speciesKey, speciesKey)) |>
      select(theta, angle)
  }
  planes <- imap(valid, \(d, head) {
    fit_means <- class_means(d)
    fit <- tangent_pca(fit_means, plane_weights, robust = robust)
    planar <- project(fit, fit_means)
    list(
      fit = fit,
      planar = planar,
      radius = median(sqrt(rowSums(planar^2)))
    )
  })
  reference <- planes[[reference_head]]$planar / planes[[reference_head]]$radius
  rotations <- map(planes, \(p) {
    procrustes_rotation(p$planar / p$radius, reference, plane_weights)
  })

  set.seed(1) # the same images in every run
  shown_ids <- runs[[reference_head]]$images |>
    filter(split == split_shown) |>
    slice_sample(n = cap, by = speciesKey) |>
    pull(image_id)

  imap(runs, \(d, head) {
    plane <- planes[[head]]
    R <- rotations[[head]]
    row <- match(shown_ids, d$images$image_id)
    # class means of the wrong species: not drawn as points, only for the limits and the photos
    gone <- is_wrong[[head]]
    positions <- species_positions(d, plane$fit, R) |>
      mutate(n_images = as.vector(table(d$images$speciesKey)[as.character(d$species$speciesKey)]))
    bind_rows(
      if (any(gone)) {
        bind_cols(
          tibble(level = rep("wrong class mean", sum(gone))),
          positions[gone, ],
          select(d$species[gone, ], speciesKey, genusKey)
        )
      },
      bind_cols(
        tibble(level = "class mean"),
        positions[!gone, ],
        select(d$species[!gone, ], speciesKey, genusKey)
      ),
      bind_cols(
        tibble(level = "image"),
        polar_placement(plane$fit, d$embeddings[row, , drop = FALSE], R),
        select(d$images[row, ], speciesKey, genusKey)
      )
    ) |>
      mutate(
        head = head,
        wrong = as.character(speciesKey) %in% as.character(wrong)
      )
  }) |>
    list_rbind()
}

# The sentence naming the family and the genera of the legends of one dataset (`study_label`) with their
# GBIF taxon keys; `genera` and `lumped` (the genera grouped as Other) are tibbles of `name` and `key`
taxa_sentence <- function(study_label, family_key, genera, lumped) {
  str_c(
    "Taxon names in the ",
    study_label,
    " figures with their GBIF taxon keys: the family ",
    italic(taxon_names(family_key)),
    " (",
    family_key,
    ") and the genera ",
    join_and(str_c(italic(genera$name), " (", genera$key, ")")),
    if (nrow(lumped) > 0) {
      str_c(
        ". The genera grouped as Other are ",
        join_and(str_c(italic(lumped$name), " (", lumped$key, ")"))
      )
    },
    "."
  )
}

# The sentence listing the species labels of the photo figures of one dataset: `lab` has the columns
# `label`, `name` and `speciesKey`; `italic_labels` for labels that are abbreviated names
species_labels_sentence <- function(study_label, lab, italic_labels) {
  str_c(
    "Species labels in the ",
    study_label,
    " photo figures, with the full names and GBIF taxon keys of the species: ",
    str_c(
      if (italic_labels) italic(lab$label) else lab$label,
      " = ",
      italic(lab$name),
      " (",
      lab$speciesKey,
      ")",
      collapse = "; "
    ),
    "."
  )
}

# The sentence on what the embedding-structure figures of one dataset show, for the caption or
# appendix: `family_key`, the species of the family (`n_species`), those with fewer than `min_images`
# images (`n_rare`) and those without a photo (`n_without_photo`), the images drawn per species (`cap`),
# and the radial limits of the panels (`limits`: columns `head`, `lo`, `hi` in degrees)
caption_sentence <- function(
  study_label, family_key, n_species, n_rare, n_without_photo, cap, min_images, limits, ring_breaks
) {
  str_c(
    "Embedding structure of the ",
    n_species,
    " species of the family ",
    italic(taxon_names(family_key)),
    " in the ",
    study_label,
    " dataset: up to ",
    cap,
    " test images per species and the species' class means, at the angle to the family's spherical ",
    "median (rings every ",
    ring_breaks,
    " degrees). The radial axis of each panel runs from ",
    join_and(str_c(limits$lo, " to ", limits$hi, " degrees (", limits$head, ")")),
    "; images beyond the outer limit are placed on it.",
    if (n_rare > 0) {
      str_c(
        " Hollow markers (faded in the photo figures) show the ",
        n_rare,
        " species with fewer than ",
        min_images,
        " images."
      )
    },
    if (n_without_photo > 0) {
      str_c(" No photo is available for ", n_without_photo, " species.")
    }
  )
}
