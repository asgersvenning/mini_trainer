source(here::here("shared.R"))

# Thumbnails of the chosen photos (step 3 of 3, see the README). The figure shows them at roughly 50-200
# px, so the whole photo would show a speck of plant in a meadow. Per species Gemini (needs the
# environment variable GEMINI_API_KEY) is asked for the bounding box of the focal organism's identifying
# features (plants: flowers and a representative group of leaves; moths: the whole insect) in the upright
# photo; it sees only the photo, never the species name. The thumbnail is the square around that box,
# with a margin and a minimum size, padded with white only where cropping to the photo would cut off too
# much of the box (see square_around()).
#
# Photos are fetched in iNaturalist's "large" size (1024 px) and kept, rotated upright, in source/. Writes
# in data/derived/species_images/<study>/thumbnails/:
#   thumbnails/<speciesKey>.jpg  the thumbnail
#   thumbnails.csv               chosen.csv plus the crop and its `coverage` (share of the model's box
#                                inside the square; < 1: features were cut off)
#   attribution.csv              per species the creator, licence, source link and modifications
#   review.html                  per species the photo with the model's box (red) and the square (green),
#                                and the thumbnail at 200, 100 and 50 px: check that the features survive
#   answers_<model>_<prompt hash>/<photo id>.rds   the model's answers (resumable)
# and, for the manuscript, results/sections/photo_credits.tex (the appendix section on the photograph
# credits) and results/values/species_photo_contributors.tex (the contributors alone, for the
# acknowledgements).
studies_used <- c("gefion-plantnet", "gefion-global-lepi")
gemini_model <- "gemini-3.1-flash-lite" # or "gemini-3.8-flash" (then thinking_level <- "low")
thinking_level <- NULL
temperature <- 0
parallel_requests <- 8
source_size <- "large"
thumbnail_px <- 400
box_margin <- 0.1 # added around the box, as a share of its larger side
# a square that cuts off more of the box than this is replaced by a padded one: plants may lose a
# little of their width, but not the moths' wings
min_coverage <- c(`gefion-plantnet` = 0.8, `gefion-global-lepi` = 0.99)
min_side <- 0.3 # smallest square, as a share of the photo's shorter side
# Manual corrections of the model's box after reviewing review.html: per species key, the factor by
# which the box is grown around its centre (> 1 when the box is too tight)
box_grow <- c(
  `3033266` = 2.2 # Anemone ranunculoides
)

prompts <- list(
  `gefion-plantnet` = str_c(
    "This photograph shows a plant. The focal plant is the single most prominent, most central plant. ",
    "Find the part of the photograph that best shows the focal plant's identifying botanical features: ",
    "its flowers or inflorescences together with a representative group of its leaves. That part will ",
    "be shown as a tiny thumbnail (50 to 200 pixels wide), so the features must stay as large and ",
    "recognisable as possible. Return the tightest bounding box that contains the focal plant's ",
    "flowers and a representative group of its leaves, excluding as much background, other plants and ",
    "empty space as possible. If the plant already fills most of the frame, return a box covering most ",
    "of the photograph. 'box_2d' is [ymin, xmin, ymax, xmax], each normalised to 0-1000. 'features': a ",
    "short phrase naming what is inside the box."
  ),
  `gefion-global-lepi` = str_c(
    "This photograph shows a moth or butterfly. The focal insect is the single most prominent, most ",
    "central one. Find the part of the photograph that best shows its identifying features: the whole ",
    "insect, with all of its wings (their shape and pattern), its body and antennae. That part will be ",
    "shown as a tiny thumbnail (50 to 200 pixels wide), so the insect must stay as large and ",
    "recognisable as possible. Return the tightest bounding box that contains the whole focal insect, ",
    "excluding as much background and empty space as possible. If the insect already fills most of ",
    "the frame, return a box covering most of the photograph. 'box_2d' is [ymin, xmin, ymax, xmax], ",
    "each normalised to 0-1000. 'features': a short phrase naming what is inside the box."
  )
)
schema <- list(
  type = "OBJECT",
  properties = list(
    box_2d = list(
      type = "ARRAY",
      items = list(type = "INTEGER"),
      minItems = 4,
      maxItems = 4
    ),
    features = list(type = "STRING")
  ),
  required = list("box_2d", "features")
)
# Name of the folder of the model's answers: the model, the thinking level and the prompt's hash
version_of <- \(prompt) {
  str_c(
    gemini_model,
    if (!is.null(thinking_level)) str_c("-", thinking_level),
    "_",
    str_sub(digest(list(prompt, schema), algo = "sha256"), 1, 8)
  )
}

# Photos as upright files in `dir`/source: paths (NA where the download failed) and sizes
prepare_sources <- function(chosen, dir) {
  path <- file.path(
    ensure_dir(file.path(dir, "source")),
    str_c(key_string(chosen$photo_id), ".jpg")
  )
  todo <- !file.exists(path)
  if (any(todo)) {
    urls <- str_replace(
      chosen$image_url[todo],
      "/medium\\.",
      str_c("/", source_size, ".")
    )
    files <- download_photos(urls)
    pmap(list(files, path[todo], chosen$rotate[todo]), \(file, to, rotate) {
      if (!is.na(file)) {
        image_read(file) |>
          image_orient() |>
          image_background("white") |> # the corners left by a rotation that is not a multiple of 90
          image_rotate(rotate) |>
          image_write(to, format = "jpeg", quality = 92)
        unlink(file)
      }
    })
  }
  path[!file.exists(path)] <- NA_character_
  path
}

# The model's box for every photo, requesting the ones not stored yet (one file per photo)
photo_boxes <- function(chosen, sources, dir, prompt) {
  dir <- ensure_dir(file.path(dir, str_c("answers_", version_of(prompt))))
  stored <- file.path(dir, str_c(key_string(chosen$photo_id), ".rds"))
  todo <- !file.exists(stored) & !is.na(sources)
  if (any(todo)) {
    message("Requesting ", sum(todo), " boxes from ", gemini_model)
    requests <- map(sources[todo], \(file) {
      gemini_request(
        file,
        prompt,
        schema,
        gemini_model,
        temperature,
        thinking_level
      )
    })
    responses <- req_perform_parallel(
      requests,
      max_active = parallel_requests,
      on_error = "continue",
      progress = TRUE
    )
    walk2(responses, which(todo), \(response, i) {
      tryCatch(
        {
          if (!inherits(response, "httr2_response")) {
            stop(conditionMessage(response), call. = FALSE)
          }
          answer <- gemini_text(response)
          parsed <- fromJSON(answer$text)
          saveRDS(
            tibble(
              photo_id = chosen$photo_id[i],
              ymin = parsed$box_2d[1],
              xmin = parsed$box_2d[2],
              ymax = parsed$box_2d[3],
              xmax = parsed$box_2d[4],
              features = parsed$features,
              model = gemini_model,
              response = answer$text,
              prompt_tokens = answer$prompt_tokens,
              output_tokens = answer$output_tokens,
              thought_tokens = answer$thought_tokens
            ),
            stored[i]
          )
        },
        error = \(e) {
          message("  failed: ", chosen$name[i], " (", conditionMessage(e), ")")
        }
      )
    })
  }
  map(stored[file.exists(stored)], readRDS) |> list_rbind()
}

# The square (pixels in the upright photo) for a box (0-1000 per axis). Without padding: the box's
# larger side plus `box_margin`, at least `min_side` of the photo's shorter side and at most the shorter
# side, centred on the box and moved inside the photo. If that square covers less than `min_coverage`
# of the box area or is shorter than the box, the thumbnail is instead padded: the square is as large as the box's larger side,
# the whole box is inside it and, on the axis where it is larger than the photo, the photo is centred
# with white on both sides. `coverage` is the share of the box area inside the square.
square_around <- function(d, min_coverage) {
  x0 <- d$xmin / 1000 * d$width
  x1 <- d$xmax / 1000 * d$width
  y0 <- d$ymin / 1000 * d$height
  y1 <- d$ymax / 1000 * d$height
  short <- pmin(d$width, d$height)
  longest <- pmax(x1 - x0, y1 - y0)
  place <- \(centre, side, extent) {
    ifelse(
      side >= extent,
      (extent - side) / 2,
      pmin(pmax(centre - side / 2, 0), extent - side)
    )
  }
  coverage <- \(side, left, top) {
    pmax(pmin(x1, left + side) - pmax(x0, left), 0) *
      pmax(pmin(y1, top + side) - pmax(y0, top), 0) /
      ((x1 - x0) * (y1 - y0))
  }
  crop_side <- pmax(pmin(longest * (1 + box_margin), short), min_side * short)
  crop_left <- place((x0 + x1) / 2, crop_side, d$width)
  crop_top <- place((y0 + y1) / 2, crop_side, d$height)
  # padded if the plain square would cut off too much of the box or any of its height
  padded <- coverage(crop_side, crop_left, crop_top) < min_coverage |
    crop_side < y1 - y0
  side <- ifelse(padded, pmax(longest, crop_side), crop_side)
  left <- place((x0 + x1) / 2, side, d$width)
  top <- place((y0 + y1) / 2, side, d$height)
  mutate(
    d,
    left = round(left),
    top = round(top),
    side = round(side),
    padded = padded,
    coverage = round(coverage(side, left, top), 2)
  )
}

# The thumbnails: the part of the square inside the photo, scaled, on a white square canvas
write_thumbnails <- function(d, dir) {
  ensure_dir(file.path(dir, "thumbnails"))
  pmap(d, \(speciesKey, source, width, height, side, left, top, ...) {
    x0 <- max(left, 0)
    y0 <- max(top, 0)
    w <- min(left + side, width) - x0
    h <- min(top + side, height) - y0
    scale <- thumbnail_px / side
    piece <- image_read(source) |>
      image_crop(geometry_area(w, h, x0, y0)) |>
      image_resize(str_c(round(w * scale), "x", round(h * scale), "!"))
    image_blank(thumbnail_px, thumbnail_px, "white") |>
      image_composite(
        piece,
        offset = geometry_point(
          round((x0 - left) * scale),
          round((y0 - top) * scale)
        )
      ) |>
      image_write(
        file.path(dir, "thumbnails", str_c(speciesKey, ".jpg")),
        format = "jpeg",
        quality = 92
      )
  })
  invisible(d)
}

write_review_sheet <- function(d, dir) {
  pct <- \(x, of) str_c(round(100 * x / of, 2), "%")
  rows <- pmap_chr(
    d,
    \(
      speciesKey,
      name,
      source,
      width,
      height,
      xmin,
      xmax,
      ymin,
      ymax,
      left,
      top,
      side,
      coverage,
      features,
      rank,
      ...
    ) {
      thumb <- str_c("thumbnails/", speciesKey, ".jpg")
      box <- function(l, t, w, h, colour) {
        str_c(
          "<div style='position:absolute;box-sizing:border-box;border:2px solid ",
          colour,
          ";left:",
          pct(l, width),
          ";top:",
          pct(t, height),
          ";width:",
          pct(w, width),
          ";height:",
          pct(h, height),
          "'></div>"
        )
      }
      str_c(
        "<h3>",
        html_escape(name),
        " <small>coverage ",
        coverage,
        "; ",
        html_escape(features),
        "</small></h3><div class='row'><div class='full'><img src='source/",
        basename(source),
        "'>",
        box(
          xmin / 1000 * width,
          ymin / 1000 * height,
          (xmax - xmin) / 1000 * width,
          (ymax - ymin) / 1000 * height,
          "red"
        ),
        box(left, top, side, side, "lime"),
        "</div>",
        str_c(
          "<img src='",
          thumb,
          "' width='",
          c(200, 100, 50),
          "'>",
          collapse = ""
        ),
        "</div>"
      )
    }
  )
  path <- file.path(dir, "review.html")
  writeLines(
    c(
      "<!doctype html><meta charset='utf-8'><title>Thumbnail review</title>",
      "<style>body{font:14px sans-serif;margin:1em 2em}.row{display:flex;gap:12px;align-items:flex-end}",
      ".full{position:relative;width:400px}.full img{width:100%;display:block}",
      "h3{margin:1.2em 0 .3em}small{font-weight:normal;color:#666}</style>",
      "<p>Red: the model's box. Green: the thumbnail's square. Right: the thumbnail at 200, 100 and 50 px.</p>",
      rows
    ),
    path
  )
  path
}

# ---- Attribution ----------------------------------------------------------------------------------
# The photos are used under CC0, CC BY or CC BY-SA. CC BY and CC BY-SA ask for: the creator, the
# licence (with a link), a link to the source, and a note that the material was changed. iNaturalist's
# `attribution` string gives the creator as "(c) <name>, some rights reserved (CC BY)"; for CC0 ("no rights
# reserved") there is no name, and the observer's iNaturalist username is credited instead (CC0 does not
# require it).
licences <- tribble(
  ~licence   , ~licence_name  , ~licence_url                                         ,
  "cc0"      , "CC0 1.0"      , "https://creativecommons.org/publicdomain/zero/1.0/" ,
  "cc-by"    , "CC BY 4.0"    , "https://creativecommons.org/licenses/by/4.0/"       ,
  "cc-by-sa" , "CC BY-SA 4.0" , "https://creativecommons.org/licenses/by-sa/4.0/"
)

# One row per thumbnail with the attribution fields and a `credit` line:
#   <creator> (<licence>), <observation url>; cropped, resized[, rotated, padded with white]
attribution_table <- function(d) {
  d |>
    left_join(licences, by = "licence") |>
    mutate(
      creator = if_else(
        str_detect(attribution, "^no rights"),
        user,
        str_remove(
          str_remove(attribution, "^\\(c\\)\\s*"),
          ",?\\s*some rights reserved.*$"
        )
      ),
      modifications = str_c(
        "cropped, resized",
        if_else(rotate != 0, ", rotated", ""),
        if_else(padded, ", padded with white", "")
      ),
      credit = str_c(
        creator,
        " (",
        licence_name,
        ", ",
        licence_url,
        "), ",
        observation_url,
        "; ",
        modifications
      )
    ) |>
    select(
      speciesKey,
      name,
      creator,
      licence_name,
      licence_url,
      observation_url,
      modifications,
      credit
    )
}

# The appendix section on the photograph credits (results/sections/photo_credits.tex; in the manuscript
# \input{results/sections/photo_credits} or \resultsection{photo_credits}, label sec:licenses): the
# licenses with their links, then per dataset and creator the species whose photograph the creator
# supplied, each with its iNaturalist observation number. For CC0 photos the creator is the observer's
# iNaturalist username (see above). `attributions` is a list of attribution tables named by the dataset's
# label.
photo_credits_section <- function(attributions) {
  used_licenses <- list_rbind(attributions) |>
    distinct(licence_name, licence_url) |>
    arrange(licence_name)
  per_dataset <- map2_chr(
    attributions,
    names(attributions),
    \(attribution, label) {
      credits <- attribution |>
        mutate(
          creator = tex_escape(creator),
          species = str_c(
            italic(name),
            " (\\href{",
            observation_url,
            "}{",
            str_extract(observation_url, "[0-9]+$"),
            "})"
          )
        ) |>
        arrange(creator, name) |>
        summarise(species = join_and(species), .by = c(creator, licence_name))
      str_c(
        "\\paragraph{",
        label,
        ".} ",
        str_c(
          credits$creator,
          " (",
          credits$licence_name,
          "): ",
          credits$species,
          collapse = "; "
        ),
        "."
      )
    }
  )
  c(
    "\\section{Photograph credits}\\label{sec:licenses}",
    str_c(
      "Each photograph in the photo figures is a crop of an iNaturalist photograph, resized (some ",
      "rotated or padded with white), used under the licenses ",
      join_and(str_c(
        used_licenses$licence_name,
        " (\\url{",
        used_licenses$licence_url,
        "})"
      )),
      ". The photographs were supplied by the following iNaturalist contributors, given by name or, ",
      "for CC0 photographs, by iNaturalist username, with the species and the iNaturalist observation ",
      "number in parentheses."
    ),
    per_dataset
  )
}

# The creators of the photos of one dataset as a list (for a thank-you in the acknowledgements):
# usernames of CC0 photos and names of the others, alphabetically
photo_contributors_sentence <- function(attribution) {
  join_and(sort(unique(tex_escape(attribution$creator)), method = "radix"))
}

attributions <- list()
photo_contributors <- list()
for (study in studies_used) {
  chosen_file <- derived_path("species_images", study, "chosen.csv")
  if (!file.exists(chosen_file)) {
    message(
      study,
      ": no ",
      chosen_file,
      " yet (species_screening.R writes it once the picks are set); the tracked thumbnails are kept"
    )
    next
  }
  chosen <- read_csv(chosen_file, show_col_types = FALSE)
  dir <- ensure_dir(derived_path("species_images", study, "thumbnails"))

  chosen$source <- prepare_sources(chosen, dir)
  sizes <- map(chosen$source[!is.na(chosen$source)], \(f) {
    as_tibble(image_info(image_read(f))[c("width", "height")])
  }) |>
    list_rbind() |>
    mutate(source = chosen$source[!is.na(chosen$source)])
  boxes <- photo_boxes(chosen, chosen$source, dir, prompts[[study]])

  d <- chosen |>
    inner_join(sizes, by = "source") |>
    inner_join(
      select(boxes, photo_id, ymin, xmin, ymax, xmax, features),
      by = "photo_id"
    ) |>
    mutate(grow = coalesce(unname(box_grow[key_string(speciesKey)]), 1)) |>
    mutate(
      centre_y = (ymin + ymax) / 2,
      centre_x = (xmin + xmax) / 2,
      ymin = pmax(centre_y - (centre_y - ymin) * grow, 0),
      ymax = pmin(centre_y + (ymax - centre_y) * grow, 1000),
      xmin = pmax(centre_x - (centre_x - xmin) * grow, 0),
      xmax = pmin(centre_x + (xmax - centre_x) * grow, 1000)
    ) |>
    select(-centre_y, -centre_x, -grow) |>
    square_around(min_coverage[[study]])
  missing_species <- setdiff(chosen$name, d$name)
  if (length(missing_species) > 0) {
    message(
      "No thumbnail (rerun to retry): ",
      str_c(missing_species, collapse = ", ")
    )
  }
  write_thumbnails(d, dir)
  write_csv(select(d, -source), file.path(dir, "thumbnails.csv"))
  attribution <- attribution_table(d)
  write_csv(attribution, file.path(dir, "attribution.csv"))
  # the copies that the figures and the credits use (tracked): the current thumbnails and both tables
  out <- thumbnail_dir(study)
  unlink(out, recursive = TRUE)
  ensure_dir(out)
  thumbs <- file.path(dir, "thumbnails", str_c(d$speciesKey, ".jpg"))
  file.copy(thumbs[file.exists(thumbs)], out)
  file.copy(file.path(dir, c("thumbnails.csv", "attribution.csv")), out)
  print(count(attribution, licence_name))
  message("Review sheet: ", write_review_sheet(d, dir))
  print(select(d, name, side, coverage, features), n = Inf)
}

# The credits of both datasets in one section, from the tracked attribution tables
for (study in studies_used) {
  file <- file.path(thumbnail_dir(study), "attribution.csv")
  if (file.exists(file)) {
    attribution <- read_csv(file, col_types = cols(.default = col_character()))
    attributions[[label_sets$dataset[[cfg$study_dataset[[study]]]]]] <- attribution
    photo_contributors[[study]] <- photo_contributors_sentence(attribution)
  }
}
if (length(attributions) > 0) {
  section <- photo_credits_section(attributions)
  writeLines(section)
  save_section(section, "photo_credits")
  do.call(
    save_results,
    c(
      list("species_photo_contributors"),
      as.list(set_names(
        photo_contributors,
        cfg$study_dataset[names(photo_contributors)]
      ))
    )
  )
}
