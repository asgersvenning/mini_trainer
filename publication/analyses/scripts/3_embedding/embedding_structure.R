source(here::here("shared.R"))

# Embedding structure figures. Plane fit: class means without the known wrong species, genus weights,
# spherical median and spherical PCA; images: `cap` per species, the same in every column; rotation:
# orthogonal Procrustes to the flat head's class-mean configuration. The coordinates: the radius is the
# exact angle (degrees) between a point and the family's spherical median (0 at the centre, so the
# centre is a fixed, meaningful origin; 90 = unrelated), and only the angle is taken from the plane
# (polar_placement()): the direction of the point's tangent vector projected onto it. Every panel ends
# at its own farthest class mean, those of the wrong species included (rounded up to `ring_step`
# degrees); images farther out are drawn on the edge. The panels therefore have different radial
# scales. Black-edged dots: class means; small dots: images; red: images of the species with known
# wrong labels (known_taxonomy_errors()), which take no part in the plane or the class means. The
# genera with the most species get the colours of `genus_palette` (as many as it has); the others are
# lumped into grey "Other".
# Main figure: heads as columns, datasets as rows. Appendix: photos at the class means (and, smaller
# and framed in red, at the class mean of each wrong species), one figure per dataset; the photos come
# from prep/ (see the README). The coordinates and the text snippets are in helpers/embeddings/structure.R.
families <- cfg$example_families # helpers/config.R
study_labels <- set_names(
  unname(label_sets$dataset[cfg$study_dataset[names(families)]]),
  names(families)
)
split_shown <- "test"
cap <- 30 # images drawn per species
reference_head <- "flat" # the other heads are rotated to its class-mean configuration
weighting <- "genus" # weight of the class means in the plane fit: "species" or "genus"
robust <- TRUE
# Where a species is drawn: "mean", "medoid" (see class_means()) or "polar": the median angle to the
# family centre of its images, in the direction of their mean projected vector (the centre of an arc
# of images, not the mean of its points). The plane itself is always fitted on the plain means.
class_mean_method <- "polar"
class_mean_k <- 3
ring_step <- 5 # degrees; a panel's outer edge is rounded up to a multiple of it
ring_breaks <- 15 # degrees between the rings drawn
squish_margin <- 0.03 # share of the radial axis between the squish limit and its end
thumbnail_size <- 0.15 # width of a photo as a share of the panel
# Species labels under the photos: "none", "shorthand" (F. vulgaris), "code" (FVU) or "number" (01)
photo_labels <- "shorthand"
photo_label_size <- 1.5 # mm
photo_label_offset <- 1 # distance below the centre of a photo, in half-widths of the photo
# A label that overlaps an earlier one (wrong species first, then the outermost) by more than this
# share of the smaller one's area is left out; `photo_panel_mm` is the width of a panel in the saved
# figure, for the size of the labels relative to the panel
photo_label_max_overlap <- 0.25
photo_panel_mm <- 105
# Photos are drawn at about 1/15 of a panel; more pixels only make the PDFs large (full-size photos
# made them 40-70 MB)
thumbnail_pixels <- 128
error_colour <- "#E41A1C" # "Outlier": the species with wrong labels
error_thumbnail_scale <- 0.8
# Species with fewer images than this have an imprecise position: hollow in the scatter plot, smaller
# (`rare_thumbnail_scale`) and faded (`rare_fade` percent white) in the photo figures
min_images <- 4
rare_thumbnail_scale <- 0.7
rare_fade <- 50
# Colours of "Other" and of the genera: Kelly's colours of maximum contrast without the reds, greys
# and near-black, plus a teal and an aquamarine. Every colour is at least 22 CIEDE2000 from every other
# (the light blue is 16 from the grey); a colour-blind reader will unfortunately confuse some pairs.
other_colour <- "#BFBFBF"
genus_palette <- c(
  "#F3C300", # yellow
  "#875692", # purple
  "#008856", # green
  "#0067A5", # blue
  "#654522", # brown
  "#E68FAC", # pink
  "#F38400", # orange
  "#A1CAF1", # light blue
  "#8DB600", # yellow-green
  "#2B99A3", # teal
  "#19EDB8" # aquamarine
)

data <- imap(families, \(family, study) family_runs(study, family))

points <- imap(data, \(runs, study) {
  polar_layers(
    runs,
    wrong = known_taxonomy_errors(study)$speciesKey,
    reference_head = reference_head,
    weighting = weighting,
    robust = robust,
    class_mean_method = class_mean_method,
    class_mean_k = class_mean_k,
    split_shown = split_shown,
    cap = cap
  ) |>
    mutate(study = study)
}) |>
  list_rbind() |>
  mutate(
    head = factor(
      head,
      names(data[[1]]),
      label_sets$head[names(data[[1]])]
    ),
    study = factor(study, names(families)),
    angle_deg = theta * 180 / pi,
    # limits of a panel (degrees): its farthest and nearest class mean (those of the wrong species
    # included), rounded up and down; the images beyond them are squished onto them
    panel_squish = min(
      90,
      ring_step * ceiling((max(angle_deg[level != "image"])) / ring_step)
    ),
    # whether any image lies beyond the squish limit (then it is drawn as a dashed ring)
    any_squished = any(level == "image" & angle_deg > panel_squish),
    panel_min = ring_step * floor(min(angle_deg[level != "image"]) / ring_step),
    # where the radial axis starts: 10 degrees inside the nearest class mean, or 0. If it is not 0, the
    # centre is cut off and a black disc up to `panel_min` marks it
    panel_lo = max(0, panel_min - 10),
    # the end of the radial axis: the squish limit plus a margin of the same share of the axis in every
    # panel (so the margin is the same size in the plot, but a different number of degrees)
    panel_max = panel_lo + (panel_squish - panel_lo) / (1 - squish_margin),
    .by = c(study, head)
  ) |>
  mutate(
    angle_deg = pmin(pmax(angle_deg, panel_lo), panel_squish),
    # the same position on the unit disk (axis start at the centre, outer edge at 1), for the photos
    radius = (angle_deg - panel_lo) / (panel_max - panel_lo),
    x = radius * sin(angle),
    y = radius * cos(angle)
  )

# Points of one dataset with the genus grouping, drawing attributes and colours of the plots
polar_data <- function(s) {
  d <- points |> filter(study == s)
  # the genera that get a colour: the ones with the most species (then the most drawn points), as many
  # as the palette has colours (wrong species not counted); the rest are "Other"
  coloured <- d |>
    filter(!wrong) |>
    summarise(
      n_species = n_distinct(speciesKey),
      n_points = n(),
      .by = genusKey
    ) |>
    arrange(desc(n_species), desc(n_points)) |>
    slice_head(n = length(genus_palette)) |>
    pull(genusKey) |>
    as.character()
  genus_names <- set_names(taxon_names(coloured), coloured)
  d <- d |>
    mutate(
      genus_grp = if_else(
        as.character(genusKey) %in% coloured,
        unname(genus_names[as.character(genusKey)]),
        "Other"
      )
    ) |>
    mutate(
      colour_grp = if_else(wrong, "Outlier", genus_grp),
      other = colour_grp == "Other",
      size_image = if_else(other, 0.2, 0.3),
      size_mean = if_else(other, 1, 2),
      alpha_image = if_else(other, 0.15, 0.3),
      # species with few images: their position is imprecise (images only in the class-mean rows)
      rare = coalesce(n_images < min_images, FALSE)
    ) |>
    arrange(colour_grp == "Outlier", !other) # drawn last: outliers on top, "Other" behind the genera
  genera <- sort(setdiff(unique(d$colour_grp), c("Other", "Outlier")))
  stopifnot(length(genera) <= length(genus_palette))
  list(
    data = d,
    breaks = c(genera, "Other", "Outlier"),
    palette = c(
      set_names(genus_palette[seq_along(genera)], genera),
      Other = other_colour,
      Outlier = error_colour
    )
  )
}

# The frame of one panel (`d`: its rows of `points`): the rings every `ring_breaks` degrees (the end of
# the axis is not a ring), the black disc that marks a cut-off centre (only if the axis does not start
# at 0; the line is the seam) and the dashed limit that the images beyond it are squished onto (only if
# there are any). Returns the layers and the rings (radius on the unit disk and degrees, for the axis).
panel_frame <- function(d) {
  lo <- d$panel_lo[1]
  hi <- d$panel_max[1]
  on_disk <- \(degrees) (degrees - lo) / (hi - lo) # radius on the unit disk of the panel
  rings <- tibble(
    degrees = seq(
      ring_breaks * (floor(lo / ring_breaks) + 1),
      hi,
      by = ring_breaks
    )
  ) |>
    mutate(radius = on_disk(degrees))
  layers <- list(
    geom_circle(
      data = rings,
      aes(x0 = 0, y0 = 0, r = radius),
      inherit.aes = FALSE,
      colour = "grey90",
      linewidth = 0.3
    ),
    if (lo > 0) {
      list(
        geom_circle(
          data = tibble(radius = on_disk(d$panel_min[1])),
          aes(x0 = 0, y0 = 0, r = radius),
          inherit.aes = FALSE,
          fill = "gray35",
          colour = "black"
        ),
        annotate(
          "text",
          x = 0,
          y = 0,
          label = str_glue("≤ {d$panel_min[1]}°"),
          color = "white",
          fontface = "bold",
          family = "Latin Modern Math"
        )
      )
    },
    if (d$any_squished[1]) {
      geom_circle(
        data = tibble(radius = on_disk(d$panel_squish[1])),
        aes(x0 = 0, y0 = 0, r = radius),
        inherit.aes = FALSE,
        linetype = "dashed",
        colour = "grey40",
        linewidth = 0.3
      )
    }
  )
  list(layers = layers, rings = rings)
}

# One row (dataset) of the main figure: one plot per head, so that every panel has its own radial scale
polar_row <- function(s, first) {
  pd <- polar_data(s)
  heads <- levels(pd$data$head)
  panels <- map(heads, \(h) {
    d <- filter(pd$data, head == h)
    frame <- panel_frame(d)
    ggplot(d, aes(x, y, colour = colour_grp)) +
      frame$layers +
      geom_point(
        data = \(d) filter(d, level == "image"),
        aes(size = size_image, alpha = alpha_image),
        show.legend = FALSE
      ) +
      geom_point(
        data = \(d) filter(d, level == "class mean", !rare),
        aes(size = size_mean, fill = colour_grp),
        shape = 21,
        colour = "black"
      ) +
      # draws nothing, but gives every colour group a legend key (some have no filled point)
      geom_point(
        data = \(d) {
          d |>
            distinct(colour_grp, other) |>
            mutate(x = NA_real_, y = NA_real_, size_mean = if_else(other, 1, 2))
        },
        aes(size = size_mean, fill = colour_grp),
        shape = 21,
        colour = "black",
        na.rm = TRUE
      ) +
      # species with fewer than `min_images` images: hollow, in the genus colour
      geom_point(
        data = \(d) filter(d, level == "class mean", rare),
        aes(size = size_mean),
        shape = 21,
        fill = "white",
        stroke = 0.8,
        show.legend = FALSE
      ) +
      scale_colour_manual(values = pd$palette, breaks = pd$breaks) +
      # every colour in the data is in the legend (also the genera that have only hollow points)
      scale_fill_manual(
        values = pd$palette,
        breaks = intersect(pd$breaks, pd$data$colour_grp),
        limits = intersect(pd$breaks, pd$data$colour_grp)
      ) +
      scale_size_identity() +
      scale_alpha_identity() +
      # Cartesian on the unit disk (coord_polar() leaves a wide margin around the circle); the ring
      # labels are the degrees, on the vertical axis where the rings cross it
      coord_fixed(xlim = c(-1, 1), ylim = c(-1, 1), expand = FALSE) +
      scale_y_continuous(
        breaks = frame$rings$radius,
        labels = str_c(frame$rings$degrees, "°")
      ) +
      facet_wrap(vars(head)) +
      labs(
        x = NULL,
        y = if (h == heads[1]) {
          str_c(study_labels[s], "\nAngle to family spherical median")
        },
        colour = "Genus",
        fill = "Genus"
      ) +
      guides(fill = guide_legend(override.aes = list(size = 5))) +
      theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_rect(
          colour = "grey40",
          fill = NA,
          linewidth = 0.3
        ),
        axis.title.y = element_text(size = rel(0.9)),
        strip.text = if (first) element_text() else element_blank(),
        strip.background = if (first) element_rect() else element_blank()
      )
  })
  wrap_plots(panels, nrow = 1, guides = "collect") &
    theme(legend.position = "bottom")
}

rows <- imap(names(families), \(s, i) polar_row(s, first = i == 1))
figure <- wrap_plots(rows, ncol = 1)
save_figure(
  figure,
  "embedding_structure",
  width = 10,
  height = 3 * length(rows),
  scale = 1.5
)

# Taxon names to GBIF keys ------------------------------------------------------------------------

# Text for the caption or appendix, one sentence per dataset: the family and the genera named in the
# legends, each with the GBIF taxon key used in the backend, and the genera lumped into "Other".
# Saved to results/values/embedding_taxa.tex; in LaTeX, e.g.
# \result[embedding_taxa, plantnet] and \result[embedding_taxa, global_lepi].
taxa_text <- map_chr(set_names(names(families)), \(s) {
  d <- polar_data(s)$data
  genera <- d |>
    filter(!colour_grp %in% c("Other", "Outlier")) |>
    distinct(name = colour_grp, key = genusKey) |>
    arrange(name)
  lumped <- d |>
    filter(colour_grp == "Other") |>
    distinct(key = genusKey) |>
    mutate(name = taxon_names(as.character(key))) |>
    arrange(name)
  taxa_sentence(study_labels[[s]], families[[s]], genera, lumped)
})
writeLines(taxa_text)
do.call(
  save_results,
  c(
    list("embedding_taxa"),
    as.list(set_names(taxa_text, cfg$study_dataset[names(taxa_text)]))
  )
)

# Photos ------------------------------------------------------------------------------------------

# The thumbnail files (species_thumbnails.R), one row per species that has one
thumbnails <- imap(families, \(family, study) thumbnail_files(study)) |>
  list_rbind()

# The species with a photo in the photo figures of dataset `s` (the class means with a thumbnail and
# the wrong species), their names, and their labels in the style `style`
species_label_table <- function(s, style = photo_labels) {
  pd <- polar_data(s)
  keys <- pd$data |>
    filter(level != "image") |>
    pull(speciesKey) |>
    as.character() |>
    unique() |>
    intersect(filter(thumbnails, study == s)$speciesKey)
  table <- tibble(speciesKey = keys, name = taxon_names(keys)) |>
    arrange(name)
  stopifnot(!anyNA(table$name))
  table$label <- switch(
    style,
    shorthand = shorthand_labels(table$name),
    code = code_labels(table$name),
    number = number_labels(table$name)
  )
  stopifnot(!anyDuplicated(table$label))
  table
}

# Photos of a dataset in the layout of the main figure, 2 x 2 heads (one plot per head, each with its
# own radial scale, rings, cut-off disc and squish limit): a framed photo in the genus colour at every
# class mean with a photo, and a smaller red one at the class mean of each wrong species. NULL if the
# dataset has no photos.
# `labels`: NULL, or the output of species_label_table(), whose labels are written under the photos.
photo_figure <- function(
  s,
  thumbnail_scale = 0.5,
  labels = NULL
) {
  pd <- polar_data(s)
  # the photos are at the class means, those of the wrong species included
  shown <- pd$data |>
    filter(level != "image") |>
    mutate(speciesKey = as.character(speciesKey)) |>
    inner_join(
      select(filter(thumbnails, study == s), -study),
      by = "speciesKey"
    )
  if (nrow(shown) == 0) {
    return(NULL)
  }
  shown <- shown |>
    mutate(
      colour = if_else(wrong, error_colour, unname(pd$palette[colour_grp])),
      framed = framed_thumbnails(
        file,
        colour,
        faded = rare,
        pixels = thumbnail_pixels,
        fade = rare_fade
      ),
      size = thumbnail_size *
        thumbnail_scale *
        if_else(wrong, error_thumbnail_scale, 1) *
        if_else(rare, rare_thumbnail_scale, 1)
    ) |>
    arrange(wrong, desc(rare)) # drawn last is on top: low-frequency species behind, wrong ones in front
  # only the colours that some photo frame has are in the legend (any genus, "Other" or "Outlier"
  # without a photo is left out), in the order of the scatter plot's legend
  legend_breaks <- intersect(pd$breaks, shown$colour_grp)
  if (!is.null(labels)) {
    shown <- shown |>
      left_join(select(labels, speciesKey, label), by = "speciesKey") |>
      mutate(
        label_y = y - size * photo_label_offset,
        show_label = visible_labels(
          x,
          label_y,
          label,
          order(!wrong, -(x^2 + y^2)),
          label_size = photo_label_size,
          panel_mm = photo_panel_mm,
          max_overlap = photo_label_max_overlap
        ),
        .by = head
      )
  }
  heads <- levels(pd$data$head)
  panels <- map(heads, \(h) {
    d <- filter(pd$data, head == h)
    frame <- panel_frame(d)
    ggplot(filter(shown, head == h), aes(x, y)) +
      frame$layers +
      geom_point(
        aes(x, y, fill = colour_grp),
        inherit.aes = FALSE,
        shape = 21,
        colour = "black",
      ) +
      {
        if (!is.null(labels)) {
          geom_text(
            data = \(d) filter(d, show_label),
            aes(x, label_y, label = label),
            inherit.aes = FALSE,
            vjust = 1,
            size = photo_label_size,
            colour = "grey20",
            family = plot_font,
            fontface = if (photo_labels == "shorthand") "italic" else "plain"
          )
        }
      } +
      geom_image(aes(x, y, image = framed, size = size), inherit.aes = FALSE) +
      scale_fill_manual(
        values = pd$palette,
        breaks = legend_breaks,
        limits = legend_breaks
      ) +
      guides(fill = guide_legend(override.aes = list(size = 5))) +
      scale_size_identity() +
      coord_fixed(xlim = c(-1, 1), ylim = c(-1, 1), expand = FALSE) +
      scale_y_continuous(
        breaks = frame$rings$radius,
        labels = str_c(frame$rings$degrees, "°")
      ) +
      facet_wrap(vars(head)) +
      labs(
        x = NULL,
        y = if (h %in% heads[c(TRUE, FALSE)]) {
          str_c(study_labels[s], "\nAngle to family spherical median")
        },
        fill = "Genus"
      ) +
      theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line = element_blank(),
        panel.grid = element_blank(),
        panel.border = element_rect(
          colour = "grey40",
          fill = NA,
          linewidth = 0.3
        ),
        axis.title.y = element_text(size = rel(0.9))
      )
  })
  wrap_plots(panels, ncol = 2, guides = "collect") &
    theme(legend.position = "bottom")
}

# Species labels of the photo figures and the text that lists them with their names and GBIF keys
# (results/values/embedding_species_labels.tex; \result[embedding_species_labels, plantnet])
species_labels <- if (photo_labels != "none") {
  map(set_names(names(families)), species_label_table)
}
if (!is.null(species_labels)) {
  label_text <- map_chr(set_names(names(families)), \(s) {
    species_labels_sentence(
      study_labels[[s]],
      species_labels[[s]],
      italic_labels = photo_labels == "shorthand"
    )
  })
  writeLines(label_text)
  do.call(
    save_results,
    c(
      list("embedding_species_labels"),
      as.list(set_names(label_text, cfg$study_dataset[names(label_text)]))
    )
  )
}

# Numbers and the sentence for the captions (results/values/embedding_caption.tex;
# \result[embedding_caption, plantnet, text] and e.g. \result[embedding_caption, plantnet, without_photo]):
# they follow the settings above (`cap`, `min_images`, ...), so rerun after changing them
caption_values <- map(set_names(names(families)), \(s) {
  species <- polar_data(s)$data |>
    filter(level != "image") |>
    distinct(speciesKey, rare)
  limits <- points |>
    filter(study == s) |>
    distinct(head, lo = panel_lo, hi = panel_squish) |>
    arrange(head)
  values <- list(
    species = nrow(species),
    rare = sum(species$rare),
    without_photo = sum(
      !as.character(species$speciesKey) %in%
        filter(thumbnails, study == s)$speciesKey
    ),
    cap = as.integer(cap),
    min_images = as.integer(min_images)
  )
  c(
    values,
    text = do.call(
      caption_sentence,
      c(
        list(
          study_labels[[s]],
          families[[s]],
          values$species,
          values$rare,
          values$without_photo
        ),
        list(cap, min_images, limits, ring_breaks)
      )
    )
  )
})
writeLines(map_chr(caption_values, "text"))
do.call(
  save_results,
  c(
    list("embedding_caption"),
    set_names(caption_values, cfg$study_dataset[names(caption_values)])
  )
)

for (s in names(families)) {
  photos <- photo_figure(s, labels = species_labels[[s]])
  if (!is.null(photos)) {
    save_figure(
      photos,
      str_c("embedding_photos_", cfg$study_dataset[[s]]),
      width = 10,
      height = 11
    )
  }
}
