source(here::here("shared.R"))

# Which species lie far from the rest in each head's embedding space? Per species and per head, the
# similarity of the species' mean image embedding (the mean direction of its images) to the centre of
# the family's species, in the full
# space (no projection involved), as the z-score of the similarity against uniformly random directions:
#   Z(theta, D) = sqrt(D - 2) * (acos(-theta) - pi / 2)
# where theta is the cosine similarity and D the dimension of the embedding (so Z is about
# sqrt(D - 2) * asin(theta): standard normal for a random direction, whatever the dimension). It has the
# same scale in every head. A low Z is far from the centre; the angular distance is kept as well.
#
# The target species (marked in the table and the figure, e.g. species with known wrong labels) are
# chosen by key at any rank with a white list (`include`) and a black list (`exclude`): a species is
# a target if it is in an included taxon and in no excluded one. Both are lists named by the rank
# column of the taxonomy (speciesKey, genusKey, familyKey, ...) with the keys as character vectors, e.g.
#   include <- list(genusKey = "8331787")                      # the whole genus ...
#   exclude <- list(speciesKey = c("5123118", "5123137"))      # ... except two of its species
# The options: taxa(study, "family"), taxa(study, "genus", filter = list(family = family)), ... (keys, names,
# species and training-image counts of the taxa; taxa(study, "species", ...) for the species)
# One figure per dataset (the family of each is the one of embedding_structure.R). `include` is set in the
# loop below: by default the species with known wrong labels (known_taxonomy_errors(); none in Pl@ntNet,
# where the figure then shows the family's curve and the outside species only).
families <- cfg$example_families # helpers/config.R
exclude <- list()
# Species outside the family (in the same syntax as `include`; empty: any species of the dataset
# outside the family): `outside_n` of them, drawn at random, are placed on the family's curve by their Z
# against the centre of the family's species (the centre does not include them), at the number of
# family species with a lower Z. `outside_n <- 0`: none.
outside <- list()
outside_n <- 50
# The centre of the class means: the spherical median (robust: a few distant species do not move it),
# or spherical_mean for the mean
centre_of <- spherical_median

for (study in names(families)) {
  family <- families[[study]]
  family_name <- taxon_names(family)
  include <- list(speciesKey = known_taxonomy_errors(study)$speciesKey)

  runs <- family_runs(study, family) # slow to load the first time (cached)
  species <- runs[[1]]$species

  # TRUE for the species of the taxonomy rows `species` that are in any taxon of the list `taxa`
  in_taxa <- function(species, taxa) {
    stopifnot(names(taxa) %in% vocab_ranks(species))
    Reduce(
      `|`,
      imap(taxa, \(keys, rank) species[[rank]] %in% keys),
      rep(FALSE, nrow(species))
    )
  }
  targets <- species$speciesKey[
    in_taxa(species, include) & !in_taxa(species, exclude)
  ]
  message(length(targets), " target species of ", nrow(species))

  # Species names: the taxonomy's, or GBIF's for the species without one (the lepidoptera export has
  # none; looked up once, see vocab_names())
  species_names <- tibble(
    speciesKey = species$speciesKey,
    name = vocab_names(species, "speciesKey")
  )

  outlyingness <- imap(runs, \(d, head) {
    means <- class_means(d)
    similarity <- drop(means %*% centre_of(means))
    tibble(
      head = head,
      speciesKey = d$species$speciesKey,
      genusKey = d$species$genusKey,
      familyKey = d$species$familyKey,
      distance = acos(pmin(similarity, 1)),
      z = sqrt(ncol(means) - 2) * (acos(-similarity) - pi / 2)
    )
  }) |>
    list_rbind() |>
    mutate(rank = rank(z), .by = head) |> # 1 = lowest Z = furthest from the centre
    left_join(species_names, by = "speciesKey") |>
    mutate(
      head = factor(head, names(runs)),
      target = speciesKey %in% targets,
      size = if_else(target, 1.6, 0.8)
    )

  # The outside species: their Z against the family's centre, and their place on the family's curve
  set.seed(1)
  vocab <- vocabulary(study)
  candidates <- vocab[
    vocab$familyKey != family &
      (if (length(outside) == 0) TRUE else in_taxa(vocab, outside)),
  ]
  outsiders <- candidates[
    sample.int(nrow(candidates), min(outside_n, nrow(candidates))),
  ]
  outside_z <- if (nrow(outsiders) == 0) {
    NULL
  } else {
    outsider_names <- tibble(
      speciesKey = outsiders$speciesKey,
      genusKey = outsiders$genusKey,
      familyKey = outsiders$familyKey,
      name = vocab_names(outsiders, "speciesKey")
    )
    outside_points <- species_points(study, outsiders, "embeddings")
    imap(runs, \(d, head) {
      centre <- centre_of(class_means(d))
      family_z <- filter(outlyingness, head == !!head)$z
      similarity <- drop(outside_points[[head]] %*% centre)
      tibble(
        head = head,
        speciesKey = rownames(outside_points[[head]]),
        distance = acos(pmin(similarity, 1)),
        z = sqrt(length(centre) - 2) * (acos(-similarity) - pi / 2),
        rank = map_dbl(z, \(zz) sum(family_z < zz) + 0.5)
      )
    }) |>
      list_rbind() |>
      left_join(outsider_names, by = "speciesKey") |>
      mutate(head = factor(head, names(runs)))
  }

  cat("\nThe ten species furthest from the centre, per head:\n")
  outlyingness |>
    slice_min(z, n = 10, by = head) |>
    arrange(head, z) |>
    select(head, name, speciesKey, genusKey, distance, z, target) |>
    print(n = Inf)

  cat("\nTarget species (rank 1 = furthest of", nrow(species), "):\n")
  outlyingness |>
    filter(target) |>
    select(name, head, distance, z, rank) |>
    arrange(name, head) |>
    print(n = Inf)

  if (!is.null(outside_z)) {
    cat(
      "\nOutside species: their Z against the family's centre, and the share below the family's median Z:\n"
    )
    outside_z |>
      left_join(
        outlyingness |> summarise(median_family_z = median(z), .by = head),
        by = "head"
      ) |>
      summarise(
        n = n(),
        median_z = median(z),
        min_z = min(z),
        max_z = max(z),
        below_family_median = mean(z < first(median_family_z)),
        .by = head
      ) |>
      print()
  }

  path <- derived_path(
    "outlyingness",
    str_c(study, "_", family, ".csv")
  )
  ensure_dir(dirname(path))
  write_csv(
    bind_rows(
      mutate(outlyingness, group = "family"),
      if (!is.null(outside_z)) mutate(outside_z, group = "outside")
    ),
    path
  )

  # the family's species (those with known wrong labels apart) and the outside species, on one scale
  types <- c(family_name, str_c(family_name, " (incorrect)"), "Other")
  head_label <- \(x) factor(as.character(x), names(runs), label_sets$head[names(runs)])
  plt_data <- bind_rows(
    mutate(outlyingness, type = if_else(target, types[2], types[1])),
    if (!is.null(outside_z)) mutate(outside_z, type = types[3])
  ) |>
    select(head, rank, z, type) |>
    mutate(type = factor(type, types), head = head_label(head))
  labelled <- outlyingness |>
    filter(target | rank <= 8 | (max(rank) - rank) <= 2) |>
    mutate(head = head_label(head), left = rank < nrow(species) / 2)

  figure <- plt_data |>
    ggplot(aes(rank, z, colour = type)) +
    geom_line(
      data = filter(plt_data, type != "Other"),
      color = "gray75",
      linewidth = 0.25
    ) +
    geom_point(aes(size = type, shape = type)) +
    geom_text(
      data = labelled,
      aes(rank + if_else(left, 1.5, -1.5), z, label = name, hjust = if_else(left, 0, 1)),
      inherit.aes = FALSE,
      size = 2.5,
      vjust = 0.5,
      family = plot_font
    ) +
    scale_colour_manual(
      values = set_names(c("grey40", "firebrick", "royalblue"), types),
      breaks = intersect(types, plt_data$type), # no legend entry for a type that has no species
      guide = guide_legend(override.aes = list(size = 5))
    ) +
    scale_size_manual(values = set_names(c(0.8, 1.6, 0.5), types), guide = "none") +
    scale_shape_manual(values = set_names(c(19, 19, 4), types), guide = "none") +
    scale_y_continuous(expand = expansion(0, 1)) +
    coord_cartesian(clip = "off") +
    facet_wrap(vars(head), nrow = 1) +
    labs(
      x = "Rank (1 = farthest from the center)",
      y = "Similarity to the family center (Z)",
      color = NULL
    ) +
    theme(legend.position = "bottom")
  save_figure(
    figure,
    str_c("outlyingness_", cfg$study_dataset[[study]]),
    width = 10,
    height = 7
  )
}
