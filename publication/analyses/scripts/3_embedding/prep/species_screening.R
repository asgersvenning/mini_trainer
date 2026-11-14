source(here::here("shared.R"))

# Screening of the candidate photos (step 2 of 3, see the README): every candidate photo is described by a
# vision model (Gemini, needs the environment variable GEMINI_API_KEY) under a fixed rubric (below, one
# per group of species), and one photo per species is selected by a rule fixed in advance: the photo
# whose weakest criterion is best, then the highest mean of the criteria, quality, verifications and
# faves. The model describes only what is visible and is never asked to identify the species.
#
# The answers are cached per photo (data/derived/species_images/<study>/screening/<model>_<rubric hash>/,
# resumable; delete a photo's file to redo it), with the model, the rubric's hash and the raw response,
# since model output is not deterministic. Next to the candidates it writes
#   selected.csv   the selected photo per species with licence and attribution
#   selected.html  review sheet with the best photos per species: select, rotate, give another iNaturalist
#                  photo or omit, then export the picks
#   chosen.csv     the manual picks with licence and attribution (for species_thumbnails.R)
# The picks are data, in data/species_picks/<study>.csv (the export of the review sheet); the sheet
# starts from that file.
studies_used <- c("gefion-plantnet", "gefion-global-lepi")
rubric_of <- c(`gefion-plantnet` = "plants", `gefion-global-lepi` = "moths")
gemini_settings <- list(
  model = "gemini-3.1-flash-lite", # or "gemini-3.8-flash"
  # For models that think: "low", "medium" or "high" (gemini-3.8-flash cannot turn thinking off:
  # "minimal" returns an error). NULL leaves the model's default, e.g. for gemini-3.1-flash-lite.
  thinking_level = NULL,
  temperature = 0, # NULL for the model's default, if its documentation recommends that
  batch_size = 64, # photos downloaded and requested together; each answer is stored as it arrives
  parallel_requests = 32 # requests in flight at once; raise it if your quota allows (429s are retried)
)
min_quality <- 3
photos_per_species <- Inf # cap while testing, e.g. 10

flag <- \(description) list(type = "BOOLEAN", description = description)
choice <- \(values) list(type = "STRING", enum = as.list(values))
quality <- list(type = "INTEGER", minimum = 1, maximum = 5)
# A graded answer as a value in [0, 1]: `scale` maps answers to values, any other answer is 0
grade <- \(x, scale) unname(coalesce(scale[as.character(x)], 0))
schema_of <- \(properties) {
  list(
    type = "OBJECT",
    properties = properties,
    required = as.list(names(properties))
  )
}

rubrics <- list(
  plants = list(
    prompt = str_c(
      "You are screening photographs for a figure that shows what a plant species looks like. ",
      "Describe only what is visible in the photograph; do not try to identify the species. ",
      "Answer in the given JSON schema. 'individual': one plant is clearly the main subject. ",
      "'natural_environment': the plant grows outdoors in a natural or semi-natural setting (not ",
      "potted, indoors, in a studio, a herbarium sheet or an illustration). 'flowers' and 'leaves': ",
      "clearly_visible means large and sharp enough in the frame to see their shape and colour; ",
      "partial_or_small means only a few, tiny, partly hidden or blurred ones; none means absent. ",
      "'subject_size': the share of the frame that the plant covers: dominant is more than about ",
      "half, moderate is about a quarter to a half, small is less than about a quarter. ",
      "'viewpoint': the camera direction relative to the plant. 'overlap': few_or_none means the ",
      "plant is clearly separate from its surroundings, with at most a few leaves of other plants ",
      "touching it; some means other plants cover part of it; dense means it is mixed up with or ",
      "partly hidden by surrounding vegetation such as grass. 'quality': 1 (poor) to 5 (excellent) ",
      "for sharpness and exposure."
    ),
    schema = schema_of(list(
      is_plant = flag("the main subject is a plant"),
      individual = flag("one individual plant is the clear main subject"),
      natural_environment = flag(
        "growing in a natural or semi-natural outdoor setting"
      ),
      flowers = choice(c("clearly_visible", "partial_or_small", "none")),
      leaves = choice(c("clearly_visible", "partial_or_small", "none")),
      whole_plant_visible = flag("the whole plant is in the frame"),
      subject_size = choice(c("dominant", "moderate", "small")),
      viewpoint = choice(c(
        "slightly_above",
        "level_side",
        "directly_overhead",
        "from_below",
        "macro_closeup",
        "unclear"
      )),
      overlap = choice(c("few_or_none", "some", "dense")),
      quality = quality
    )),
    # Selection rules can be changed freely without new requests: `qualifies` (hard requirements),
    # `criteria` (what is scored) and `minor` only read the cached answers, and the cache is keyed
    # by the prompt and schema alone.
    qualifies = \(d) d$is_plant & d$quality >= min_quality,
    # criteria that only count in the mean (tie-break), not in the minimum
    minor = c("overlap", "individual", "whole_plant_visible"),
    criteria = \(d) {
      tibble(
        individual = as.numeric(d$individual),
        natural_environment = as.numeric(d$natural_environment),
        flowers = grade(
          d$flowers,
          c(none = 0, partial_or_small = 0.5, clearly_visible = 1)
        ),
        leaves = grade(
          d$leaves,
          c(none = 0, partial_or_small = 0.5, clearly_visible = 1)
        ),
        whole_plant_visible = as.numeric(d$whole_plant_visible),
        subject_size = grade(
          d$subject_size,
          c(small = 0, moderate = 0.5, dominant = 1)
        ),
        viewpoint = grade(
          d$viewpoint,
          c(slightly_above = 1, level_side = 0.5, directly_overhead = 0.5)
        ),
        overlap = grade(d$overlap, c(dense = 0, some = 0.5, few_or_none = 1))
      )
    },
    describe = \(d) {
      str_c(
        "flowers ",
        d$flowers,
        ", leaves ",
        d$leaves,
        ", ",
        d$subject_size,
        " subject, ",
        d$viewpoint,
        ", ",
        d$overlap,
        " overlap"
      )
    }
  ),
  moths = list(
    prompt = str_c(
      "You are screening photographs for a figure that shows what a moth or butterfly species looks ",
      "like. Describe only what is visible in the photograph; do not try to identify the species. ",
      "Answer in the given JSON schema. 'adult': the adult stage (not an egg, caterpillar or pupa). ",
      "'individual': one insect is clearly the main subject. 'view': the camera direction relative to ",
      "the insect; top_down means the upper side of the wings is seen from directly above. ",
      "'wings_spread': the wings are open or flat so that their upper surface is visible. ",
      "'background': minimal means plain, uniform or evenly blurred; cluttered means busy or distracting. ",
      "'quality': 1 (poor) to 5 (excellent) for sharpness, exposure and how well the insect fills the frame."
    ),
    schema = schema_of(list(
      is_lepidopteran = flag("the main subject is a moth or butterfly"),
      adult = flag("the insect is an adult"),
      individual = flag("a single individual is the clear main subject"),
      view = choice(c(
        "top_down",
        "side",
        "front",
        "underside",
        "oblique",
        "unclear"
      )),
      wings_spread = flag("the wings are open or flat, upper surface visible"),
      background = choice(c("minimal", "moderate", "cluttered")),
      quality = quality
    )),
    qualifies = \(d) {
      d$is_lepidopteran & d$adult & d$individual & d$quality >= min_quality
    },
    criteria = \(d) {
      tibble(
        view = grade(d$view, c(top_down = 1, oblique = 0.5)),
        background = grade(
          d$background,
          c(minimal = 1, moderate = 0.5, cluttered = 0)
        ),
        wings_spread = as.numeric(d$wings_spread)
      )
    },
    describe = \(d) str_c(d$view, " view, ", d$background, " background")
  )
)

for (study in studies_used) {
  rubric <- rubrics[[rubric_of[[study]]]]
  candidates_file <- derived_path("species_images", study, "candidates.rds")
  if (!file.exists(candidates_file)) {
    stop(
      "Run species_candidates.R for ",
      study,
      " first (it writes ",
      candidates_file,
      ")",
      call. = FALSE
    )
  }
  all_candidates <- readRDS(candidates_file)
  candidates <- filter(all_candidates, !is.na(image_url))
  # Species without any open-licence candidate (kept out of the screening, but recorded)
  no_candidates <- all_candidates |>
    distinct(speciesKey, name, taxon_id) |>
    filter(!speciesKey %in% candidates$speciesKey)
  write_csv(no_candidates, derived_path("species_images", study, "no_candidates.csv"))
  if (nrow(no_candidates) > 0) {
    message(
      "No open-licence candidate (written to no_candidates.csv): ",
      str_c(no_candidates$name, collapse = ", ")
    )
  }
  screening <- candidates |>
    slice_head(n = photos_per_species, by = speciesKey) |>
    screen_photos(study, candidates = _, rubric, gemini_settings)

  joined <- inner_join(
    candidates,
    select(screening, -model, -rubric_sha, -response),
    by = "photo_id"
  )
  ranked <- joined[rubric$qualifies(joined), ]
  criteria <- rubric$criteria(ranked)
  major <- setdiff(names(criteria), rubric$minor)
  ranked$score <- apply(criteria[major], 1, min) # the weakest major criterion decides ...
  ranked$mean_score <- rowMeans(criteria) # ... and the mean over all (incl. minor) breaks ties
  ranked$description <- rubric$describe(ranked)
  ranked <- arrange(
    ranked,
    speciesKey,
    desc(score),
    desc(mean_score),
    desc(quality),
    desc(coalesce(verifications, 0)),
    desc(coalesce(faves, 0))
  )
  selected <- slice_head(ranked, n = 1, by = speciesKey)
  write_csv(
    select(
      selected,
      speciesKey,
      name,
      photo_id,
      image_url,
      licence,
      attribution,
      user,
      observation_url,
      score,
      mean_score,
      quality
    ),
    derived_path("species_images", study, "selected.csv")
  )
  cat(
    "\n",
    study,
    ": ",
    nrow(selected),
    " of ",
    n_distinct(all_candidates$speciesKey),
    " species have a qualifying photo (",
    nrow(no_candidates),
    " without any open-licence candidate)\n",
    sep = ""
  )
  print(count(selected, score))
  picks <- read_picks(study)
  if (!is.null(picks)) {
    # picks of species outside the candidates (an earlier family) are ignored
    stale <- !picks$speciesKey %in% as.numeric(all_candidates$speciesKey)
    if (any(stale)) {
      message("Ignoring ", sum(stale), " picks of species not in the candidates (an earlier family)")
    }
    picks <- if (all(stale)) NULL else picks[!stale, ]
  }
  message("Selection sheet: ", write_selection_sheet(ranked, study, picks))
  if (!is.null(picks)) {
    chosen <- chosen_photos(picks, candidates)
    undecided <- setdiff(
      key_string(unique(candidates$speciesKey)),
      key_string(picks$speciesKey[picks$omit | !is.na(picks$photo_id)])
    )
    if (length(undecided) > 0) {
      message(length(undecided), " species have no pick yet")
    }
    write_csv(chosen, derived_path("species_images", study, "chosen.csv"))
    message("Chosen photos: ", nrow(chosen), " species, written to chosen.csv")
  }
}
