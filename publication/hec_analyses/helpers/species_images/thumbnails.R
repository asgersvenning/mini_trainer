# The thumbnails of the species photos of one study, tracked in data/species_thumbnails/<study>/ (one
# <speciesKey>.jpg per species, thumbnails.csv and attribution.csv; written by species_thumbnails.R),
# so that the photo figures and the credits need neither the photo pipeline nor an API key
thumbnail_dir <- function(study) here("data", "species_thumbnails", study)

# The thumbnail files of one study, one row per species that has one (none if there are no thumbnails)
thumbnail_files <- function(study) {
  csv <- file.path(thumbnail_dir(study), "thumbnails.csv")
  if (!file.exists(csv)) {
    return(tibble(study = character(), speciesKey = character(), file = character()))
  }
  current <- read_csv(csv, col_types = cols(.default = col_character()))$speciesKey
  files <- file.path(thumbnail_dir(study), str_c(current, ".jpg"))
  tibble(
    study = study,
    speciesKey = current[file.exists(files)],
    file = files[file.exists(files)]
  )
}

# Copies of the photos `files`, shrunk to `pixels` in width (a photo covers a small part of a panel,
# and the full-size photos made the PDFs tens of megabytes) and with a border in `colours` (a share
# `width` of the photo's width), in the temporary directory. `faded` photos are mixed with `fade`
# percent white
framed_thumbnails <- \(
  files,
  colours,
  faded = rep(FALSE, length(files)),
  width = 0.04,
  pixels = 128,
  fade = 50
) {
  dir <- ensure_dir(file.path(tempdir(), "framed_thumbnails"))
  paths <- file.path(
    dir,
    str_c(map_chr(str_c(files, colours, faded, pixels, fade), digest), ".png")
  )
  for (i in which(!file.exists(paths))) {
    photo <- image_resize(image_read(files[i]), str_c(pixels, "x"))
    if (faded[i]) {
      photo <- image_colorize(photo, fade, "white") # mixed with white (PDFs need no alpha)
    }
    border <- max(2, round(width * image_info(photo)$width))
    image_write(
      image_border(photo, colours[i], str_c(border, "x", border)),
      paths[i],
      format = "png"
    )
  }
  paths
}
