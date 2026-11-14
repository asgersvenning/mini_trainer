# Data of one taxonomic family under one run: the objects the embedding-structure figure
# projects. Depends on heads.R (vocabulary, heads, layer_weights) and the ERDA accessors.

unit_rows <- function(X) X / sqrt(rowSums(X^2))

# family_data() of the heads of one backbone, as a named list, cached per family and run (the scan of
# the embeddings is slow; delete the files in data/derived/families/ to reload). A conditional head's
# prototype is the path sum over its layers, which approximates its predictions.
family_runs <- function(
  study,
  family,
  heads = c("flat", "hierarchical", "independent", "conditional"),
  backbone = "efficientnet_v2_s",
  vocab = vocabulary(study)
) {
  map(set_names(heads), \(head) {
    run <- str_c(backbone, "_", head)
    cached(derived_path("families", study, family, str_c(run, ".rds")), {
      message("Loading ", study, " ", family, " ", run)
      family_data(study, family, run, vocab, path = head == "conditional")
    })
  })
}

# The unit prototypes of any rows `species` of the vocabulary under one run (rows named by speciesKey;
# see family_data() for `path`)
species_prototypes <- function(study, species, run_id, path = FALSE) {
  H <- heads(study, run_id) |>
    filter(key %in% !!unique(unlist(species[vocab_ranks(species)]))) |>
    collect()
  prototypes_from(H, species, path = path)
}

# The points of any rows `species` of the vocabulary under each head, as a named list of unit matrices
# with rows named by speciesKey: the class means of their images (`basis` "embeddings") or their
# prototypes ("prototypes"). Cached by the species.
species_points <- function(
  study,
  species,
  basis = c("embeddings", "prototypes"),
  heads = c("flat", "hierarchical", "independent", "conditional"),
  backbone = "efficientnet_v2_s"
) {
  basis <- match.arg(basis)
  map(set_names(heads), \(head) {
    run <- str_c(backbone, "_", head)
    cached(
      derived_path("species_points", study, str_c(digest(species$speciesKey), "_", basis, "_", run, ".rds")),
      {
        message("Loading ", nrow(species), " species of ", study, " ", run, " (", basis, ")")
        path <- head == "conditional"
        if (basis == "embeddings") {
          d <- species_data(study, species, run, path)
          # species without images in the export have no class mean
          d$species <- d$species[d$species$speciesKey %in% d$images$speciesKey, ]
          class_means(d)
        } else {
          species_prototypes(study, species, run, path)
        }
      }
    )
  })
}

# A `family_data()` result without some species (their rows in species, prototypes, images and
# embeddings), e.g. those with known wrong labels
without_species <- function(d, keys) {
  species <- !as.character(d$species$speciesKey) %in% as.character(keys)
  images <- !as.character(d$images$speciesKey) %in% as.character(keys)
  list(
    species = d$species[species, ],
    prototypes = d$prototypes[species, , drop = FALSE],
    genus = d$genus,
    images = d$images[images, ],
    embeddings = d$embeddings[images, , drop = FALSE]
  )
}

# Central embedding of the images of every species of a `family_data()` result (unit rows named by
# speciesKey, in `d$species` order), computed on the unprojected embeddings on the hypersphere:
#   "mean"    the normalised mean of the embeddings: the point on the sphere that minimises the summed
#             squared straight-line (chordal) distance to them (the extrinsic mean)
#   "medoid"  the normalised mean of the `k` embeddings of the species with the smallest summed angular
#             (geodesic) distance to its other embeddings (all of them if it has fewer; `k` = 1 is the
#             medoid itself). Those are close to each other, so the mean shrinks towards the centre much
#             less than the mean of all the images, which noise averages towards the family centre.
#             Needs the n x n angles of a species' n images.
class_means <- function(d, method = c("mean", "medoid"), k = 3) {
  if (match.arg(method) == "mean") {
    return(unit_rows(rowsum(d$embeddings, d$images$speciesKey)[d$species$speciesKey, ]))
  }
  rows <- split(seq_len(nrow(d$embeddings)), d$images$speciesKey)[
    as.character(d$species$speciesKey)
  ]
  centres <- map(rows, \(i) {
    X <- d$embeddings[i, , drop = FALSE]
    nearest <- order(rowSums(acos(pmin(pmax(tcrossprod(X), -1), 1))))
    colMeans(X[nearest[seq_len(min(k, nrow(X)))], , drop = FALSE])
  })
  unit_rows(`rownames<-`(do.call(rbind, centres), d$species$speciesKey))
}

# Everything needed to look at a family's species in the embedding space of one run:
#   species     taxonomy rows of the family, in vocabulary order
#   prototypes  unit species prototypes (species layer of the head), rows named by speciesKey
#   genus       unit genus-layer weights of the family's genera (NULL for heads without one)
#   images      image_id, class_id, speciesKey, genusKey, split of the family's images
#   embeddings  unit image embeddings, rows aligned with `images`
#
# `path = TRUE` uses the effective prototype of a conditional head (species layer plus ancestor layers,
# see head_prototypes). Hierarchical heads store only species weights, so `genus` is NULL for them.
family_data <- function(study, family, run_id, vocab = vocabulary(study), path = FALSE) {
  species_data(study, vocab[vocab$familyKey == family, ], run_id, path)
}

# family_data() for any rows `species` of the vocabulary (e.g. species of several families)
species_data <- function(study, species, run_id, path = FALSE) {
  genus_keys <- unique(species$genusKey)
  # head rows of the family's taxa at every rank (lazily; lepi heads have 12,632 species)
  H <- heads(study, run_id) |>
    filter(key %in% !!unique(unlist(species[vocab_ranks(species)]))) |>
    collect()

  images <- images(study) |>
    filter(class_id %in% !!species$class_id) |>
    select(image_id, class_id, split) |>
    collect() |>
    left_join(select(species, class_id, speciesKey, genusKey), by = "class_id")

  E <- embeddings(study, run_id) |>
    filter(image_id %in% !!images$image_id) |>
    select(image_id, matches("^e\\d+$")) |>
    collect()
  E <- E[match(images$image_id, E$image_id), ]
  stopifnot(!anyNA(E$image_id))

  list(
    species = species,
    prototypes = prototypes_from(H, species, path = path),
    genus = if (any(H$rank == 1)) {
      unit_rows(layer_weights(H, 1)[genus_keys, , drop = FALSE])
    },
    images = images,
    embeddings = unit_rows(as.matrix(select(E, -image_id)))
  )
}
