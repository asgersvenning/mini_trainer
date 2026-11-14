# Access to the ERDA snapshots pinned in data/snapshots.json (folder names, roles,
# studies, manifest hashes). Depends on `cfg` and the path helpers (io.R).
#
# Lazy: choosing a study or run costs only small metadata files. A parquet file is
# read straight from ERDA over HTTPS (DuckDB range requests; only the requested
# columns and rows are transferred) unless a verified local copy exists in
# data/raw/. Use download_data() for files you read repeatedly or need whole.
#
# Typical use:
#   runs("plantnet-16")                                   # lazy table
#   scores("plantnet-16", "full_seed42", rank = 0) |> select(image_id, c0) |> collect()
#   replicates("plantnet-16") |> filter(setting == "zero") |> collect()

# --- Session cache ----------------------------------------------------------

.session <- new.env()
once <- function(key, expr) {
  if (!exists(key, envir = .session, inherits = FALSE)) assign(key, expr, envir = .session)
  get(key, envir = .session)
}

# --- Pinned index and manifests ----------------------------------------------

snapshot_index <- function() once("index", fromJSON(cfg$snapshots, simplifyVector = FALSE))

snapshot <- function(folder) {
  hit <- Find(function(s) s$name == folder, snapshot_index()$snapshots)
  if (is.null(hit)) stop("Folder not in ", cfg$snapshots, ": ", folder, call. = FALSE)
  hit
}

# The single folder of a given role ("evidence", "replicates", "checkpoints") that provides `study`.
folder_for <- function(role, study) {
  hits <- Filter(function(s) s$role == role && study %in% unlist(s$studies), snapshot_index()$snapshots)
  if (length(hits) != 1) {
    stop("Expected one '", role, "' folder for study '", study, "', found ", length(hits), call. = FALSE)
  }
  hits[[1]]$name
}

file_url <- function(folder, path) {
  base <- if (is.null(cfg$base_url)) snapshot_index()$read_base_url else cfg$base_url
  paste(sub("/$", "", base), folder, utils::URLencode(path), sep = "/")
}

# A folder's manifest as a tibble (path, bytes, sha256), checked against the pinned hash.
folder_manifest <- function(folder) {
  once(paste0("manifest:", folder), {
    local <- raw_path(folder, "manifest.json")
    if (!file.exists(local)) {
      download_file(file_url(folder, "manifest.json"), local, snapshot(folder)$manifest_sha256)
    }
    files <- fromJSON(local, simplifyVector = FALSE)$files
    tibble(
      path = names(files),
      bytes = map_dbl(files, "bytes"),
      sha256 = map_chr(files, "sha256")
    )
  })
}

manifest_entry <- function(folder, path) {
  entry <- folder_manifest(folder)
  entry <- entry[entry$path == path, ]
  if (nrow(entry) != 1) stop("Not in manifest of ", folder, ": ", path, call. = FALSE)
  entry
}

# --- Files ------------------------------------------------------------------

sha256_of <- function(file) digest(file, algo = "sha256", file = TRUE)

download_file <- function(url, dest, sha256) {
  if (!startsWith(url, "https://")) stop("Downloads require HTTPS: ", url, call. = FALSE)
  ensure_dir(dirname(dest))
  part <- paste0(dest, ".part")
  on.exit(unlink(part))
  utils::download.file(url, part, mode = "wb", quiet = TRUE)
  if (sha256_of(part) != sha256) stop("Checksum mismatch: ", url, call. = FALSE)
  file.rename(part, dest)
}

# Local path of a file; downloads it (once, verified) if missing. Large files need allow_large = TRUE.
data_file <- function(folder, path, allow_large = FALSE) {
  local <- raw_path(folder, path)
  if (file.exists(local)) return(local)
  entry <- manifest_entry(folder, path)
  if (entry$bytes > cfg$max_download_mb * 1e6 && !allow_large) {
    stop(sprintf("%s is %.0f MB (limit %d MB). Use allow_large = TRUE to download it.",
                 path, entry$bytes / 1e6, cfg$max_download_mb), call. = FALSE)
  }
  download_file(file_url(folder, path), local, entry$sha256)
  local
}

# Download every file of a folder (optionally only paths matching `pattern`).
download_data <- function(folder, pattern = NULL, allow_large = FALSE) {
  paths <- folder_manifest(folder)$path
  if (!is.null(pattern)) paths <- grep(pattern, paths, value = TRUE)
  for (path in paths) data_file(folder, path, allow_large)
  invisible(paths)
}

# Where to read a file from: the local copy if present, else its URL (or a
# download when cfg$remote is FALSE).
data_source <- function(folder, path) {
  manifest_entry(folder, path)
  local <- raw_path(folder, path)
  if (file.exists(local)) return(normalizePath(local, winslash = "/"))
  if (cfg$remote) return(file_url(folder, path))
  normalizePath(data_file(folder, path), winslash = "/")
}

# On demand: compare downloaded files of a folder with its manifest.
verify_data <- function(folder) {
  manifest <- folder_manifest(folder)
  local <- raw_path(folder, manifest$path)
  manifest$status <- vapply(seq_along(local), function(i) {
    if (!file.exists(local[i])) "missing"
    else if (sha256_of(local[i]) == manifest$sha256[i]) "ok"
    else "corrupt"
  }, character(1))
  manifest
}

# --- Lazy parquet tables ------------------------------------------------------

db <- function() {
  once("db", {
    con <- dbConnect(duckdb())
    dbExecute(con, "INSTALL httpfs; LOAD httpfs;")
    # bounded memory and parallelism; no insertion order saves buffering in large scans
    dbExecute(con, sprintf("SET memory_limit = '%s'", cfg$duckdb_memory_limit))
    dbExecute(con, sprintf("SET threads = %d", cfg$duckdb_threads))
    dbExecute(con, "SET preserve_insertion_order = false")
    con
  })
}

# One lazy (dbplyr) table over parquet files or URLs; columns are unioned by name.
parquet_table <- function(sources) {
  files <- paste(dbQuoteString(db(), sources), collapse = ", ")
  tbl(db(), sql(sprintf("SELECT * FROM read_parquet([%s], union_by_name = true)", files)))
}

# --- Evidence (tidy snapshots; catalog.csv has one row per file) ---------------

catalog <- function(folder) {
  once(paste0("catalog:", folder), read_csv(data_file(folder, "catalog.csv"), show_col_types = FALSE))
}

# Lazy table of all `kind` files of `study`, optionally filtered on catalog columns,
# e.g. evidence("scores", "plantnet-16", run_id = "full_seed42", rank = 0).
# Files without a study in the catalog (runs.parquet) are folder-level.
evidence <- function(kind, study, ...) {
  folder <- folder_for("evidence", study)
  files <- catalog(folder)
  files <- files[files$kind == kind & (is.na(files$study) | files$study == study), ]
  filters <- list(...)
  for (column in names(filters)) files <- files[files[[column]] %in% filters[[column]], ]
  if (nrow(files) == 0) stop("No '", kind, "' files for ", study, " match the filters.", call. = FALSE)
  parquet_table(vapply(files$path, function(path) data_source(folder, path), character(1)))
}

runs        <- function(study) filter(evidence("runs", study), study == !!study)
images      <- function(study) evidence("images", study)
taxonomy    <- function(study) evidence("taxonomy", study)
classes     <- function(study) evidence("classes", study)
learning    <- function(study) evidence("learning", study)
# Scores (c<j>) and embeddings (e<j>) are stored as float16. DuckDB reads them as
# floating point; check_float16() guards against a reader that returns the raw
# 16-bit codes instead (as R's arrow does), which would give integers like 38009.
check_float16 <- function(table) {
  first <- table |> select(any_of(paste0(c("c", "e"), rep(0:7, each = 2)))) |> head(1) |> collect()
  values <- unlist(first)
  if (is.integer(values) || !all(is.finite(values)) || max(abs(values)) > 1000) {
    stop("float16 columns look like raw 16-bit codes, not numbers; cast them to float32 when reading.", call. = FALSE)
  }
  table
}

scores      <- function(study, run_id, rank = 0L) check_float16(evidence("scores", study, run_id = run_id, rank = rank))
embeddings  <- function(study, run_id) check_float16(evidence("embeddings", study, run_id = run_id))

# --- Bootstrap replicates ---------------------------------------------------

# Lazy table <study>/<table>.parquet of the replicates folder (table: "replicates",
# "thresholds" or "weights"). Replicate 0 is the 90% reporting partition, not the
# full test set (see resampling.json).
replicates <- function(study, table = "replicates") {
  folder <- folder_for("replicates", study)
  parquet_table(data_source(folder, paste0(study, "/", table, ".parquet")))
}

# Lazy table <study>/predictions/<run_id>.parquet of the replicates folder: the per-image predictions
# of one run (columns: see `head(predictions(study, run_id)) |> collect()`).
predictions <- function(study, run_id) {
  folder <- folder_for("replicates", study)
  parquet_table(data_source(folder, paste0(study, "/predictions/", run_id, ".parquet")))
}

# <study>/resampling.json of the replicates folder as a list: setup, settings and
# `ranks` (level k of the replicate tables is ranks[k + 1]; absent for the legacy campaign).
resampling <- function(study) {
  folder <- folder_for("replicates", study)
  fromJSON(data_file(folder, paste0(study, "/resampling.json")))
}

# --- Discovery ----------------------------------------------------------------
# What is available? studies() -> contents(study) -> schema(kind, study).

# One row per (study, role, folder, status), from the pinned index (no network).
# `status` is "current" unless the index marks the folder "legacy" (an older campaign
# kept for comparison).
studies <- function() {
  bind_rows(lapply(snapshot_index()$snapshots, function(s) {
    tibble(study = unlist(s$studies), role = s$role, folder = s$name,
           status = if (is.null(s$status)) "current" else s$status)
  })) |>
    arrange(study, role)
}

# Every file of a study across folders: path, kind, run_id, rank, role, folder, bytes
# and whether a verified local copy exists. Evidence rows come from catalog.csv;
# other folders have no catalog, so their `kind` is the first path component after
# the study (replicates.parquet -> "replicates", predictions/<run>.parquet -> "predictions").
contents <- function(study) {
  folders <- studies()[studies()$study == study, ]
  if (nrow(folders) == 0) stop("Unknown study: ", study, call. = FALSE)
  bind_rows(lapply(seq_len(nrow(folders)), function(i) {
    folder <- folders$folder[i]
    manifest <- folder_manifest(folder)
    files <- if (folders$role[i] == "evidence") {
      cat <- catalog(folder)
      as_tibble(cat[is.na(cat$study) | cat$study == study, c("path", "kind", "run_id", "rank")])
    } else {
      paths <- grep(paste0("^", study, "/"), manifest$path, value = TRUE)
      tibble(path = paths, kind = sub(paste0("^", study, "/([^/.]+).*"), "\\1", paths))
    }
    files$role <- folders$role[i]
    files$folder <- folder
    files$bytes <- manifest$bytes[match(files$path, manifest$path)]
    files$cached <- file.exists(raw_path(folder, files$path))
    files
  }))
}

# Column names and Arrow types of an evidence table kind (schemas.json). `kind = NULL`
# lists the available kinds.
schema <- function(kind = NULL, study) {
  schemas <- fromJSON(data_file(folder_for("evidence", study), "schemas.json"), simplifyVector = FALSE)
  if (is.null(kind)) return(names(schemas))
  if (is.null(schemas[[kind]])) {
    stop("Unknown kind '", kind, "'. Available: ", paste(names(schemas), collapse = ", "), call. = FALSE)
  }
  tibble(column = names(schemas[[kind]]), type = unlist(schemas[[kind]]))
}
