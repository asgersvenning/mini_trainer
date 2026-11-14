# Canonical order and display labels (id = label) for categorical variables, so
# every figure and table orders and names them the same way. Edit here only.

label_sets <- list(
  head = c(
    flat = "Flat",
    hierarchical = "Hierarchical",
    independent = "Independent",
    conditional = "Conditional",
    autoregressive = "Autoregressive"
  ),
  backbone = c(
    efficientnet_b0 = "EfficientNetB0",
    efficientnet_v2_s = "EfficientNetV2-Small",
    efficientnet_v2_m = "EfficientNetV2-Medium",
    efficientnet_v2_l = "EfficientNetV2-Large",
    vit_l_16 = "ViT-Large-16",
    vit_h_14 = "ViT-Huge-14",
    bioclip2 = "BioCLIP-2"
  ),
  dataset = c(
    plantnet = "Pl@ntNet300K",
    global_lepi = "Global Lepidoptera",
    flemming = "Expert labeled AMI"
  ),
  level = c(
    species = "Species",
    genus = "Genus",
    family = "Family",
    order = "Order",
    class = "Class",
    phylum = "Phylum",
    kingdom = "Kingdom"
  ),
  metric = c(
    accuracy = "Macro Accuracy",
    recall = "Macro Recall",
    precision = "Macro Precision",
    f1 = "Macro F1",
    micro_accuracy = "Micro Accuracy",
    micro_recall = "Micro Recall",
    micro_precision = "Micro Precision",
    micro_f1 = "Micro F1",
    theilU = "Theil's U",
    coverage = "Coverage",
    in_vocab = "In Vocabulary"
  ),
  # names(cluster_methods()) (helpers/embeddings/alignment.R)
  cluster_method = c(
    average = "average linkage",
    complete = "complete linkage",
    pam = "PAM",
    `hdbscan (flat)` = "HDBSCAN"
  )
)

# Ordered factor with display labels, e.g. relabel(df$head, "head").
# Unknown ids are an error, so a new value must be added to `label_sets` first.
relabel <- function(x, set) {
  labels <- label_sets[[set]]
  unknown <- setdiff(unique(as.character(x)), names(labels))
  if (length(unknown)) {
    stop("Not in label_sets$", set, ": ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  factor(x, levels = names(labels), labels = labels)
}

# Rank names of a study's replicate tables as an ordered factor (species first, "Key" suffix
# removed): level k of the tables is `resampling(study)$ranks[k + 1]`.
# Usage: mutate(level = rank_factor(level, "plantnet")).
rank_factor <- function(level, study) {
  ranks <- resampling(study)$ranks
  fct_relabel(factor(ranks[level + 1], ranks), \(x) str_remove(x, "Key$"))
}

# ---- Labels of species in figures -------------------------------------------------------------------

# Labels of species from their names (`names`: GBIF canonical names, "Genus epithet", sorted
# alphabetically). All three styles are unique within `names`.
# - shorthand: "F. vulgaris"; the genus is abbreviated to the fewest letters that make the labels of
#   the whole set unique ("Ad. alba" and "Am. alba" if two genera share an initial and an epithet)
shorthand_labels <- \(names) {
  genus <- str_extract(names, "^\\S+")
  epithet <- str_remove(names, "^\\S+\\s+")
  letters_kept <- rep(1L, length(names))
  repeat {
    label <- str_c(str_sub(genus, 1, letters_kept), ". ", epithet)
    clash <- label %in% label[duplicated(label)]
    if (!any(clash)) {
      return(label)
    }
    letters_kept[clash] <- letters_kept[clash] + 1L
  }
}
# - code: three capital letters, e.g. "FVU": the first letter of the genus and the first two of the
#   epithet; if taken (by an earlier name in the alphabet), the next free combination of the first
#   letter of the genus with two letters of the epithet in order (FVL, FVG, ...), and then with the
#   genus' second and third letters
code_labels <- \(names) {
  genus <- toupper(str_extract(names, "^[A-Za-z]+"))
  epithet <- toupper(str_remove_all(
    str_remove(names, "^\\S+\\s+"),
    "[^A-Za-z]"
  ))
  codes <- character(length(names))
  for (i in seq_along(names)) {
    g <- strsplit(genus[i], "")[[1]]
    e <- c(strsplit(epithet[i], "")[[1]], "X", "X")[
      1:max(2, min(6, nchar(epithet[i])))
    ]
    pairs <- apply(combn(e, 2), 2, str_c, collapse = "")
    candidates <- setdiff(
      as.vector(outer(pairs, g[seq_len(min(3, length(g)))], \(p, l) {
        str_c(l, p)
      })),
      codes[seq_len(i - 1)]
    )
    stopifnot(length(candidates) > 0)
    codes[i] <- candidates[1]
  }
  codes
}
# - number: the rank in the alphabet, zero-padded to the same width for all
number_labels <- \(names) {
  sprintf(str_c("%0", nchar(length(names)), "d"), seq_along(names))
}

# Which labels to draw: TRUE for the labels at (`x`, `y`) (top centre, panel units: the panel is 2
# wide) that do not overlap a label that was kept before them (in the order of `priority`, the
# first is kept) by more than `max_overlap` of the smaller one's area. Sizes are estimated from the
# font size `label_size` (mm), the width of a panel in the saved figure `panel_mm` (mm) and the number
# of characters (half the font size each).
visible_labels <- \(x, y, label, priority, label_size, panel_mm, max_overlap) {
  mm_per_unit <- panel_mm / 2
  width <- str_length(label) * 0.5 * label_size / mm_per_unit
  height <- label_size / mm_per_unit
  keep <- rep(FALSE, length(x))
  for (i in priority) {
    kept <- which(keep)
    shared <- pmax(
      0,
      pmin(x[i] + width[i] / 2, x[kept] + width[kept] / 2) -
        pmax(x[i] - width[i] / 2, x[kept] - width[kept] / 2)
    ) *
      pmax(0, pmin(y[i], y[kept]) - pmax(y[i] - height, y[kept] - height))
    keep[i] <- all(
      shared <= max_overlap * height * pmin(width[i], width[kept])
    )
  }
  keep
}
