# Screening of candidate photos with a vision model (Gemini): one request per photo under a rubric,
# every answer stored on its own so that a run can be resumed. `settings` is a list with `model`,
# `thinking_level`, `temperature`, `batch_size` and `parallel_requests` (set in species_screening.R).
# Depends on gemini.R and web.R.

# Hash of what the model is asked and how it must answer (the prompt and the response schema): it
# names the cache folder, so a changed rubric never reuses old answers. The model and the thinking level
# are in the folder name; the temperature is not (change `temperature` together with the folder).
rubric_hash <- \(rubric) {
  digest(list(rubric$prompt, rubric$schema), algo = "sha256")
}

# The request for one downloaded photo under a rubric (gemini_request() is in gemini.R)
photo_request <- \(file, rubric, settings) {
  gemini_request(
    file,
    rubric$prompt,
    rubric$schema,
    settings$model,
    settings$temperature,
    settings$thinking_level
  )
}

# One response -> one row of answers, with the model, the prompt's hash and the raw response
photo_answer <- function(response, rubric, settings) {
  answer <- gemini_text(response)
  as_tibble(fromJSON(answer$text)) |>
    mutate(
      model = settings$model,
      rubric_sha = rubric_hash(rubric),
      response = answer$text,
      # token usage: the cost, and whether the model spends time thinking
      prompt_tokens = answer$prompt_tokens,
      output_tokens = answer$output_tokens,
      thought_tokens = answer$thought_tokens
    )
}

# ---- One queue for all photos of all species -----------------------------------------------------
# Every photo is stored on its own as soon as its answer arrives (screening/<version>/<photo id>.rds),
# so an interrupted run loses at most the batch in flight and a rerun only does what is missing. The
# queue is worked off in batches of `batch_size` photos: within a batch the photos are downloaded
# concurrently and their requests run `parallel_requests` at a time.

screening_dir <- \(study, rubric, settings) {
  derived_path(
    "species_images",
    study,
    "screening",
    str_c(
      settings$model,
      if (!is.null(settings$thinking_level)) str_c("-", settings$thinking_level),
      "_",
      str_sub(rubric_hash(rubric), 1, 8)
    )
  )
}

# One batch: downloads the photos, runs the requests, parses and stores every answer. Failures are
# reported and left unstored (a rerun retries them). Returns the number of answers stored.
process_batch <- function(batch, rubric, dir, settings) {
  files <- download_photos(batch$image_url)
  requests <- map(files, \(file) {
    if (is.na(file)) NULL else photo_request(file, rubric, settings)
  })
  unlink(files[!is.na(files)])
  sent <- !map_lgl(requests, is.null)
  if (!any(sent)) {
    return(0L)
  }
  responses <- req_perform_parallel(
    requests[sent],
    max_active = settings$parallel_requests,
    on_error = "continue",
    progress = FALSE
  )
  stored <- map2_lgl(
    responses,
    batch$photo_id[sent],
    \(response, photo_id) {
      answer <- tryCatch(
        {
          if (!inherits(response, "httr2_response")) {
            stop(conditionMessage(response), call. = FALSE)
          }
          photo_answer(response, rubric, settings)
        },
        error = \(e) {
          message(
            "  failed: photo ",
            key_string(photo_id),
            " (",
            conditionMessage(e),
            ")"
          )
          NULL
        }
      )
      if (is.null(answer)) {
        return(FALSE)
      }
      saveRDS(
        mutate(answer, photo_id = photo_id, .before = 1),
        file.path(dir, str_c(key_string(photo_id), ".rds"))
      )
      TRUE
    }
  )
  sum(stored)
}

# The answers for all candidate photos of a study, screening the ones not stored yet. The bar shows
# the photos done of those to do, the species completed of all species, the species last worked on,
# the rate and the ETA.
screen_photos <- function(study, candidates, rubric, settings) {
  dir <- ensure_dir(screening_dir(study, rubric, settings))
  stored_file <- \(photo_id) file.path(dir, str_c(key_string(photo_id), ".rds"))
  pending <- \(d) d[!file.exists(stored_file(d$photo_id)), ]
  todo <- pending(arrange(candidates, name, rank, photo_index))
  n_species <- n_distinct(candidates$speciesKey)
  species_done <- \() n_species - n_distinct(pending(candidates)$speciesKey)
  message(
    study,
    ": ",
    nrow(candidates),
    " photos of ",
    n_species,
    " species, ",
    nrow(candidates) - nrow(todo),
    " already screened, ",
    nrow(todo),
    " to do"
  )

  if (nrow(todo) > 0) {
    cli_progress_bar(
      total = nrow(todo),
      format = str_c(
        "{cli::pb_bar} {cli::pb_percent} | photos {cli::pb_current}/{cli::pb_total} | ",
        "species {cli::pb_extra$species_done}/{cli::pb_extra$species_total} | ",
        "{cli::pb_extra$current} | {cli::pb_rate} | ETA {cli::pb_eta}"
      ),
      extra = list(
        species_done = species_done(),
        species_total = n_species,
        current = todo$name[1]
      ),
      clear = FALSE
    )
    processed <- 0
    stored <- 0
    for (rows in split(
      seq_len(nrow(todo)),
      ceiling(seq_len(nrow(todo)) / settings$batch_size)
    )) {
      batch <- todo[rows, ]
      stored <- stored + process_batch(batch, rubric, dir, settings)
      if (stored == 0) {
        stop(
          "The first batch of photos failed: see the messages above",
          call. = FALSE
        )
      }
      processed <- processed + nrow(batch)
      cli_progress_update(
        set = processed,
        extra = list(
          species_done = species_done(),
          species_total = n_species,
          current = batch$name[nrow(batch)]
        ),
        force = TRUE
      )
    }
    cli_progress_done()
  }

  files <- stored_file(candidates$photo_id)
  unscreened <- !file.exists(files)
  if (any(unscreened)) {
    message(
      sum(unscreened),
      " photos could not be screened (rerun to retry them)"
    )
  }
  map(files[!unscreened], readRDS) |> list_rbind()
}
