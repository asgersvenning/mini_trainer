# Vocabulary, image labels, embeddings and head weights of the embedding studies
# (e.g. "gefion-plantnet"). Depends on the ERDA accessors (data.R) and `derived_path` (io.R).
#
# Head weights are the pinned `heads` snapshot (data/snapshots.json), heads/<study>/<run_id>.parquet
# (columns: study, run_id, rank, class_index, key, bias, w0, w1, ...): effective classifier weights per
# rank. Flat and hierarchical heads have a species layer only; conditional and independent heads
# have one layer per rank (ancestor weights are not added to the species rows).

# Ranks, species upwards, and their taxonomy columns (`rank_keys`: those the datasets have, rank 0 to 4)
rank_ranks <- c("species", "genus", "family", "order", "class", "phylum", "kingdom")
rank_keys <- str_c(rank_ranks[1:5], "Key")

# The rank keys a taxonomy has, species upwards (lepidoptera stop at family)
vocab_ranks <- function(vocab) rank_keys[rank_keys %in% names(vocab)]

# Species vocabulary (taxonomy) in class order, collected. The taxonomy files are read over the
# network if they are not downloaded, which is slow and repeated for every query of the lazy
# `taxonomy(study)`; so the collected table is kept in data/derived/vocabulary/ (per pinned folder).
vocabulary <- function(study) {
  cached(
    derived_path("vocabulary", str_c(folder_for("evidence", study), "_", study, ".rds")),
    taxonomy(study) |>
      arrange(class_id) |>
      collect()
  )
}

# GBIF names of taxon keys (canonical names), each looked up once and kept in
# data/derived/taxon_names.rds. NA for a key GBIF does not know or a failed lookup (not kept, so it is
# tried again on the next call).
taxon_names <- function(keys) {
  path <- derived_path("taxon_names.rds")
  known <- if (file.exists(path)) readRDS(path) else character()
  new <- setdiff(unique(as.character(keys)), names(known))
  if (length(new) > 0) {
    # silent when all names are known; otherwise a progress bar, and the names found so far are
    # saved every 100 lookups (an interrupted run keeps them)
    ensure_dir(dirname(path))
    cli_progress_bar(
      "Looking up taxon names at GBIF",
      total = length(new),
      format = "{cli::pb_spin} {cli::pb_name} {cli::pb_bar} {cli::pb_current}/{cli::pb_total} | {cli::pb_rate} | ETA {cli::pb_eta}"
    )
    found <- character()
    for (i in seq_along(new)) {
      found[new[i]] <- or_na(get_json(str_c("https://api.gbif.org/v1/species/", new[i]), 0)$canonicalName)
      if (i %% 100 == 0) {
        saveRDS(c(known, found[!is.na(found)]), path)
      }
      cli_progress_update()
    }
    cli_progress_done()
    saveRDS(c(known, found[!is.na(found)]), path)
    known <- c(known, found)
  }
  unname(known[as.character(keys)])
}

# A rank name ("family", " Genus ", "familyKey") as its taxonomy column ("familyKey"). Stops on an
# unknown rank, or one that the taxonomy of the study does not have.
rank_column <- function(rank, vocab) {
  rank <- str_remove(str_to_lower(str_trim(rank)), "key$")
  if (length(rank) != 1 || !rank %in% rank_ranks) {
    stop("Unknown rank '", rank, "': use one of ", str_c(rank_ranks, collapse = ", "), call. = FALSE)
  }
  column <- str_c(rank, "Key")
  if (!column %in% names(vocab)) {
    stop(
      "The taxonomy has no ", rank, " ranks; it has ",
      str_c(str_remove(vocab_ranks(vocab), "Key$"), collapse = ", "),
      call. = FALSE
    )
  }
  column
}

# Taxon names of the rows of `vocab` at one rank column: the species' scientific names (GBIF names for
# the species that have none, as in the lepidoptera export), the GBIF names of the higher taxa
vocab_names <- function(vocab, column) {
  if (column == "speciesKey") {
    nm <- vocab$scientificName
    missing <- is.na(nm)
    if (any(missing)) {
      nm[missing] <- taxon_names(vocab$speciesKey[missing])
    }
    nm
  } else {
    taxon_names(vocab[[column]])
  }
}

# Keys or names of a filter entry as clean character: trimmed, no missing or empty values
clean_values <- function(x) {
  x <- unlist(x)
  x <- if (is.numeric(x)) format(x, scientific = FALSE, trim = TRUE) else as.character(x)
  x <- str_squish(x)
  x[!is.na(x) & x != ""]
}

# The taxa of a study at one rank, with their name, number of species and training images: the
# options for choosing a family or a target. Columns: key, name, n_species, train_images.
#   target_rank  the rank of the taxa listed: "species" (default), "genus", "family", "order", "class",
#                "phylum" or "kingdom" (the study's taxonomy has the ranks it has: the lepidoptera stop
#                at family). Case and spaces do not matter, and "familyKey" is accepted.
#   filter       which species to take the taxa from: a list named by rank, e.g.
#                list(family = "8863", genus = c("Adaina", "1859082")). The values are keys or names
#                (names are GBIF's, case-insensitive; the names of the species are the taxonomy's);
#                the values of one rank are alternatives (or), the ranks together are all required
#                (and). A missing rank, or an empty entry, means no restriction on that rank.
# Values that match nothing are reported, as they are probably typos.
taxa <- function(
  study,
  target_rank = c("species", "genus", "family", "order", "class", "phylum", "kingdom"),
  filter = list()
) {
  vocab <- vocabulary(study)
  column <- rank_column(target_rank[[1]], vocab)
  filter <- as.list(filter %||% list())
  for (entry in seq_along(filter)) {
    values <- clean_values(filter[[entry]])
    if (length(values) == 0) {
      next
    }
    filter_column <- rank_column(names(filter)[entry], vocab)
    is_key <- str_detect(values, "^[0-9]+$")
    keys_hit <- vocab[[filter_column]] %in% values[is_key]
    names_hit <- rep(FALSE, nrow(vocab))
    if (!all(is_key)) {
      names_here <- str_to_lower(vocab_names(vocab, filter_column))
      names_hit <- names_here %in% str_to_lower(values[!is_key])
      unmatched <- setdiff(str_to_lower(values[!is_key]), names_here)
    } else {
      unmatched <- character()
    }
    unmatched <- c(unmatched, setdiff(values[is_key], vocab[[filter_column]]))
    if (length(unmatched) > 0) {
      message("No ", names(filter)[entry], " matches: ", str_c(unmatched, collapse = ", "))
    }
    vocab <- vocab[keys_hit | names_hit, ]
  }
  if (nrow(vocab) == 0) {
    return(tibble(key = character(), name = character(), n_species = integer(), train_images = numeric()))
  }
  out <- vocab |>
    summarise(
      n_species = n(),
      train_images = sum(train_count),
      .by = all_of(column)
    ) |>
    arrange(desc(n_species), desc(train_images))
  out$name <- vocab_names(vocab[match(out[[column]], vocab[[column]]), ], column)
  select(out, key = all_of(column), name, n_species, train_images)
}

# Species whose labels in the datasets are known to be wrong (data/known_taxonomy_errors.csv: the
# labelled and the correct genus and family, and where the error is documented), optionally of one
# study. The labels are used as they are everywhere; this is the place to look species up, e.g. to mark
# them in a figure.
known_taxonomy_errors <- function(study = NULL) {
  errors <- read_csv(
    here("data", "known_taxonomy_errors.csv"),
    col_types = cols(.default = col_character())
  )
  if (is.null(study)) errors else errors[errors$study == study, ]
}

# Lazy table of the head weights of one run.
heads <- function(study, run_id) {
  parquet_table(data_source(
    folder_for("heads", study),
    paste0("heads/", study, "/", run_id, ".parquet")
  ))
}

# Weight matrix (classes x dim) of one rank layer, rows named by taxon key
layer_weights <- function(H, r) {
  Hr <- filter(H, rank == r)
  `rownames<-`(as.matrix(select(Hr, starts_with("w"))), Hr$key)
}

# Species prototypes of a head, in vocabulary order and unit length: either its own
# species layer, or (path = TRUE) the sum of the layers of the species and all its
# ancestors (the effective prototype of a conditional head).
head_prototypes <- function(study, run_id, vocab, path = FALSE) {
  prototypes_from(head_rows(study, run_id, vocab), vocab, path)
}

# The collected head rows of the taxa of `vocab` (filtered lazily, so a family costs little of a large
# head). Collect once when several prototypes of one run are needed.
head_rows <- function(study, run_id, vocab) {
  needed <- unique(unlist(vocab[vocab_ranks(vocab)]))
  heads(study, run_id) |>
    filter(key %in% !!needed) |>
    collect()
}

# The same from collected head rows `H` (must contain the taxa of `vocab`)
prototypes_from <- function(H, vocab, path = FALSE) {
  keys <- if (path) vocab_ranks(vocab) else rank_keys[1]
  W <- imap(keys, \(col, r) {
    layer_weights(H, r - 1)[vocab[[col]], , drop = FALSE]
  }) |>
    Reduce(`+`, x = _)
  W / sqrt(rowSums(W^2))
}
