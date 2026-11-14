source(here::here("shared.R"))

# A representative ("typical") family per dataset for inspecting the embedding space: enough
# species and genera, and a distribution of training and test frequencies per species and of
# genus sizes that is close to the dataset's. Each family gets three distances to the dataset's
# distribution, a mean absolute difference between quantile functions (a 1-D Wasserstein-type
# distance) in units of the dataset's SD, of
#   d_train  log10(training images + 1) per species
#   d_test   log10(test images + 1) per species
#   d_genus  log(species per genus), over the genera of the family
# and their sum is the score (smaller is more typical). Only families with at least `min_species`
# species and `min_genera` genera and `min_test_images` test images are considered.
studies_used <- c("gefion-plantnet", "gefion-global-lepi")
min_species <- 15
min_genera <- 4
min_test_images <- 200
max_species <- 100 # readability of the figure; typicality hardly discriminates in large datasets
shortlist <- 10

# Mean absolute difference between the quantile functions of `x` and `ref` over a grid of
# probabilities, in SDs of `ref`
quantile_distance <- function(x, ref, p = seq(0.05, 0.95, by = 0.05)) {
  mean(abs(quantile(x, p, names = FALSE) - quantile(ref, p, names = FALSE))) /
    sd(ref)
}

# One row per species: taxonomy keys, training and test images
species_table <- function(study) {
  test_counts <- images(study) |>
    filter(split == "test") |>
    count(class_id, name = "test_count") |>
    collect()
  vocabulary(study) |>
    select(class_id, speciesKey, genusKey, familyKey, train_count) |>
    left_join(test_counts, by = "class_id") |>
    mutate(test_count = coalesce(test_count, 0L))
}

# Families ranked by how typical they are, with their distances
rank_families <- function(species) {
  genera <- count(species, familyKey, genusKey, name = "n")
  reference <- list(
    train = log10(species$train_count + 1),
    test = log10(species$test_count + 1),
    genus = log(genera$n)
  )
  per_species <- species |>
    summarise(
      n_species = n(),
      n_genera = n_distinct(genusKey),
      test_images = sum(test_count),
      d_train = quantile_distance(log10(train_count + 1), reference$train),
      d_test = quantile_distance(log10(test_count + 1), reference$test),
      .by = familyKey
    )
  per_genus <- genera |>
    summarise(
      d_genus = quantile_distance(log(n), reference$genus),
      .by = familyKey
    )
  per_species |>
    left_join(per_genus, by = "familyKey") |>
    filter(
      n_species >= min_species,
      n_species <= max_species,
      n_genera >= min_genera,
      test_images >= min_test_images
    ) |>
    mutate(score = d_train + d_test + d_genus) |>
    arrange(score)
}

# The training and test frequency distributions of the top families against the dataset's
plot_candidates <- function(species, ranking, top = 3) {
  families <- head(ranking$familyKey, top)
  bind_rows(
    mutate(species, group = "dataset"),
    species |>
      filter(familyKey %in% families) |>
      mutate(group = familyKey)
  ) |>
    pivot_longer(
      c(train_count, test_count),
      names_to = "split",
      names_pattern = "(.*)_count",
      values_to = "images"
    ) |>
    ggplot(aes(
      log10(images + 1),
      color = group,
      linewidth = group == "dataset"
    )) +
    stat_ecdf() +
    scale_linewidth_manual(
      values = c(`TRUE` = 1.5, `FALSE` = 0.6),
      guide = "none"
    ) +
    facet_wrap(~split) +
    labs(x = "log10(images + 1) per species", y = "ECDF", color = NULL)
}

for (study in studies_used) {
  species <- species_table(study)
  ranking <- rank_families(species)
  cat(
    "\n",
    study,
    ":",
    n_distinct(species$familyKey),
    "families,",
    nrow(species),
    "species;",
    nrow(ranking),
    "eligible\n"
  )
  print(head(ranking, shortlist), n = Inf)
  print(plot_candidates(species, ranking) + ggtitle(study))
}
