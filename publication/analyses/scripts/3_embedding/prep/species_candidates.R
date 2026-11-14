source(here::here("shared.R"))

# Candidate photos per species for the photo version of the embedding-structure figure (step 1 of 3, see
# the README). Candidates are iNaturalist observations that are research grade, not captive, and have a
# photo licensed CC0, CC BY or CC BY-SA (non-commercial licences conflict with open-access publication).
# The species' iNaturalist taxon is found from its GBIF key, never from its name (names are ambiguous).
# Of the `n_pool` most voted observations the `n_keep` best are kept (most identifications agreeing with
# the observation's taxon, then faves, then number of identifications), and the first `max_photos`
# photos of each are the candidates, with their licence and attribution.
#
# Nothing is downloaded: data/derived/species_images/<study>/candidates.html links the photos on
# iNaturalist's servers. The table is cached in candidates.rds next to it; delete it to query again.
families <- cfg$example_families # helpers/config.R
studies_used <- c("gefion-plantnet", "gefion-global-lepi")
n_pool <- 100
n_keep <- 30
max_photos <- 3

# Candidate photos of one taxon: one row per photo with the observation's quality signals
inaturalist_candidates <- function(taxon_id) {
  observations <- inaturalist_observations(
    taxon_id,
    per_page = n_pool,
    quality_grade = "research",
    captive = "false",
    order_by = "votes",
    order = "desc"
  )
  rows <- map(observations, \(o) {
    photos <- o$photos |>
      keep(\(p) or_na(p$license_code) %in% open_licences) |>
      head(max_photos)
    map(seq_along(photos), \(i) {
      bind_cols(
        tibble(observation_id = num_or_na(o$id), photo_index = i),
        inaturalist_photo_row(o, photos[[i]]),
        tibble(
          verifications = num_or_na(o$num_identification_agreements),
          identifications = num_or_na(o$identifications_count),
          faves = num_or_na(o$faves_count)
        )
      )
    }) |>
      list_rbind()
  }) |>
    list_rbind()
  if (nrow(rows) == 0) {
    return(rows)
  }
  best <- rows |>
    distinct(observation_id, verifications, faves, identifications) |>
    arrange(
      desc(coalesce(verifications, 0)),
      desc(coalesce(faves, 0)),
      desc(coalesce(identifications, 0))
    ) |>
    slice_head(n = n_keep) |>
    mutate(rank = row_number())
  inner_join(best[c("observation_id", "rank")], rows, by = "observation_id") |>
    arrange(rank, photo_index)
}

# The candidate rows of one species
species_candidates <- function(speciesKey, name) {
  message(name)
  taxon <- inaturalist_taxon_of(speciesKey)
  if (is.na(taxon$taxon_id) && !is.na(name)) {
    message("  no iNaturalist observation in GBIF: matching the name (check by eye)")
    taxon <- inaturalist_taxon(name)
    taxon$taxon_note <- str_c(coalesce(taxon$taxon_note, ""), " [matched by name]")
  }
  if (!is.na(taxon$taxon_note)) {
    message("  ", taxon$taxon_note)
  }
  if (is.na(taxon$taxon_id)) {
    message("  no iNaturalist taxon")
    return(tibble(speciesKey = speciesKey, name = name, taxon_id = NA_real_, taxon_note = taxon$taxon_note))
  }
  found <- inaturalist_candidates(taxon$taxon_id)
  if (nrow(found) == 0) {
    message("  no open-licence research-grade photo")
    found <- tibble(observation_id = NA_real_)
  }
  found |>
    mutate(
      speciesKey = speciesKey,
      name = name,
      taxon_id = taxon$taxon_id,
      taxon_note = taxon$taxon_note,
      .before = 1
    )
}

candidates_study <- function(study) {
  family <- families[[study]] # not inside filter(): the taxonomy has its own `study` column
  species <- vocabulary(study) |>
    filter(familyKey == family) |>
    select(speciesKey) |>
    mutate(name = taxon_names(speciesKey))
  path <- derived_path("species_images", study, "candidates.rds")
  candidates <- cached(path, pmap(species, species_candidates) |> list_rbind())

  # The cache is per study: drop species of other families (an earlier `families` setting) and query
  # the species it lacks, with the name-match fallback of species_candidates()
  candidates <- filter(candidates, speciesKey %in% species$speciesKey)
  missing <- filter(species, !speciesKey %in% candidates$speciesKey)
  if (nrow(missing) > 0) {
    candidates <- bind_rows(candidates, pmap(missing, species_candidates) |> list_rbind())
    saveRDS(candidates, path)
  }

  # Audit of the taxa of an existing table: species whose iNaturalist taxon by key differs from the
  # stored one (e.g. a name match to a different organism) are queried again and replaced, the others
  # are left as they are (a new query could reorder their candidates). Their picks must be redone.
  by_key <- species |>
    mutate(taxon_by_key = map_dbl(speciesKey, \(k) inaturalist_taxon_of(k)$taxon_id)) |>
    left_join(distinct(candidates, speciesKey, taxon_id), by = "speciesKey") |>
    filter(!is.na(taxon_by_key), is.na(taxon_id) | taxon_id != taxon_by_key)
  if (nrow(by_key) > 0) {
    cat("\nSpecies whose stored iNaturalist taxon differs from the one by key (replaced):\n")
    print(select(by_key, speciesKey, name, stored = taxon_id, by_key = taxon_by_key), n = Inf)
    candidates <- bind_rows(
      filter(candidates, !speciesKey %in% by_key$speciesKey),
      pmap(select(by_key, speciesKey, name), species_candidates) |> list_rbind()
    )
    saveRDS(candidates, path)
  }
  candidates
}

write_candidate_sheet <- function(candidates, study) {
  card <- \(image_url, observation_url, rank, verifications, faves, licence) {
    str_c(
      "<div class='card'><a href='",
      html_escape(observation_url),
      "'><img loading='lazy' src='",
      html_escape(image_url),
      "'></a><div class='meta'>#",
      rank,
      " &#10003;",
      verifications,
      " &#9733;",
      faves,
      " ",
      html_escape(licence),
      "</div></div>"
    )
  }
  rows <- candidates |>
    arrange(name) |>
    nest(.by = c(speciesKey, name)) |>
    pmap_chr(\(speciesKey, name, data) {
      with_photos <- filter(data, !is.na(image_url))
      note <- data$taxon_note[[1]]
      str_c(
        "<h3>",
        html_escape(name),
        " <small>",
        speciesKey,
        ": ",
        n_distinct(with_photos$observation_id),
        " observations, ",
        nrow(with_photos),
        " photos</small>",
        if (!is.na(note)) str_c("<br><small class='note'>", html_escape(note), "</small>"),
        "</h3><div class='row'>",
        str_c(
          pmap_chr(
            with_photos[c(
              "image_url",
              "observation_url",
              "rank",
              "verifications",
              "faves",
              "licence"
            )],
            card
          ),
          collapse = ""
        ),
        "</div>"
      )
    })
  path <- derived_path("species_images", study, "candidates.html")
  writeLines(
    c(
      "<!doctype html><meta charset='utf-8'><title>Candidate pictures</title>",
      "<style>body{font:14px sans-serif;margin:1em 2em}.row{display:flex;gap:6px;flex-wrap:wrap}",
      ".card{width:150px}img{width:150px;height:150px;object-fit:cover;background:#eee}",
      ".meta{font-size:11px;color:#444}h3{margin:1.2em 0 .3em}small{font-weight:normal;color:#666}</style>",
      "<p>#rank, &#10003; verifications (agreeing identifications), &#9733; faves, licence. ",
      "Click a photo for its iNaturalist observation.</p>",
      rows
    ),
    path
  )
  path
}

for (study in studies_used) {
  candidates <- candidates_study(study)
  cat("\n", study, "\n", sep = "")
  print(
    candidates |>
      summarise(
        observations = n_distinct(observation_id, na.rm = TRUE),
        photos = sum(!is.na(image_url)),
        min_verifications = suppressWarnings(min(verifications, na.rm = TRUE)),
        .by = c(speciesKey, name, taxon_note)
      ),
    n = Inf
  )
  message("Candidate sheet: ", write_candidate_sheet(candidates, study))
}
