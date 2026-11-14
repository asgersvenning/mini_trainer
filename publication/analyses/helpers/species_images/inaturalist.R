# iNaturalist: taxa, observations and their photos (api.inaturalist.org/v1), and the link from GBIF
# species keys to iNaturalist taxa. Depends on web.R (get_json, query, key_string, or_na, num_or_na).

# Photo licences that allow use in an open-access publication (no non-commercial or no-derivatives)
open_licences <- c("cc0", "cc-by", "cc-by-sa")

# GBIF's copy of iNaturalist's research-grade observations: every occurrence has the GBIF backbone
# `speciesKey` and, as `taxonID`, the iNaturalist taxon id
inaturalist_dataset <- "50c9509d-22c7-4a22-a47d-8c48425ef4a7"

# iNaturalist taxon id of a GBIF species key, by identifiers only (see `inaturalist_dataset`): from
# the observations identified to exactly that species (not to a subspecies). NA id if there are none.
inaturalist_taxon_of <- function(speciesKey) {
  key <- key_string(speciesKey)
  occurrences <- get_json(
    str_c(
      "https://api.gbif.org/v1/occurrence/search?",
      query(datasetKey = inaturalist_dataset, speciesKey = key, limit = "50")
    )
  )$results
  ids <- occurrences |>
    keep(\(o) num_or_na(o$taxonKey) == as.numeric(key)) |>
    map_chr(\(o) or_na(o$taxonID))
  ids <- ids[!is.na(ids)]
  tibble(
    taxon_id = if (length(ids) == 0) NA_real_ else as.numeric(names(sort(table(ids), decreasing = TRUE))[1]),
    taxon_note = if (length(unique(ids)) > 1) "GBIF occurrences disagree on the iNaturalist taxon" else NA_character_
  )
}

# FALLBACK, by name (ambiguous: only for species without observations in GBIF): the iNaturalist
# taxon of a species name: the active taxon with exactly that name or, if the name is an
# inactive taxon that iNaturalist has swapped or merged (e.g. Angelica heterocarpa, replaced by
# Angelica sylvestris var. villosa), the single taxon that replaced it, with a `taxon_note` saying so
# (the replacement may be a different rank or a wider concept: check these species by eye). NA id if
# there is no taxon, or no single replacement.
inaturalist_taxon <- function(name) {
  taxon <- \(id, note = NA_character_) tibble(taxon_id = id, taxon_note = note)
  find <- \(active) {
    hits <- get_json(str_c(
      "https://api.inaturalist.org/v1/taxa?",
      query(rank = "species", is_active = active, per_page = "5", q = name)
    ))
    keep(hits$results, \(r) str_to_lower(or_na(r$name)) == str_to_lower(name))
  }
  active <- find("true")
  if (length(active) > 0) {
    return(taxon(num_or_na(active[[1]]$id)))
  }
  inactive <- find("false")
  if (length(inactive) == 0) {
    return(taxon(NA_real_))
  }
  replaced <- get_json(str_c("https://api.inaturalist.org/v1/taxa/", inactive[[1]]$id))$results[[1]]
  replacements <- unlist(replaced$current_synonymous_taxon_ids)
  if (length(replacements) != 1) {
    return(taxon(
      NA_real_,
      str_c("inactive on iNaturalist, ", length(replacements), " replacement taxa")
    ))
  }
  target <- get_json(str_c("https://api.inaturalist.org/v1/taxa/", replacements))$results[[1]]
  taxon(
    num_or_na(target$id),
    str_c("iNaturalist has replaced ", name, " with ", or_na(target$name))
  )
}

# A page of observations (as lists) of a taxon that have a photo with an open licence; `...` are
# further query parameters (quality_grade, captive, order_by, ...)
inaturalist_observations <- function(taxon_id, per_page = 200, page = 1, ...) {
  get_json(str_c(
    "https://api.inaturalist.org/v1/observations?",
    query(
      taxon_id = key_string(taxon_id),
      photo_license = str_c(open_licences, collapse = ","),
      photos = "true",
      per_page = as.character(per_page),
      page = as.character(page),
      ...
    )
  ))$results
}

# One row for a photo `p` of observation `o` (as lists from the API): the photo's id, the medium-size
# image, its licence and attribution string (no creator name for CC0), the observer's iNaturalist
# username and the observation's page
inaturalist_photo_row <- function(o, p) {
  tibble(
    photo_id = num_or_na(p$id),
    image_url = str_replace(or_na(p$url), "square", "medium"),
    licence = or_na(p$license_code),
    attribution = or_na(p$attribution),
    user = or_na(o$user$login),
    observation_url = or_na(o$uri)
  )
}

# One row (inaturalist_photo_row) for an iNaturalist photo (by its id) of a taxon, looked up among the
# taxon's observations with open-licence photos. Stops if the photo is not found there.
inaturalist_photo <- function(photo_id, taxon_id) {
  for (page in 1:10) {
    results <- inaturalist_observations(taxon_id, page = page)
    for (o in results) {
      for (p in o$photos) {
        if (num_or_na(p$id) == photo_id && or_na(p$license_code) %in% open_licences) {
          return(inaturalist_photo_row(o, p))
        }
      }
    }
    if (length(results) < 200) break
  }
  stop(
    "photo ", key_string(photo_id), " not found among the open-licence photos of iNaturalist taxon ",
    key_string(taxon_id), " (wrong species, or not CC0/CC BY/CC BY-SA)",
    call. = FALSE
  )
}
