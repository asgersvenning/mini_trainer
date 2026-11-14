# Paths, caching and provenance. Depends on `cfg` (config.R).
#
# Layers:  raw     <- downloaded from ERDA, immutable        (raw_path)
#          derived <- regenerable intermediate results       (derived_path)
#          results <- final figures/tables used in the paper (result_path)
# Scripts read raw/derived and write derived/results, never the reverse.
# Access to the ERDA data itself is in data.R.

raw_path     <- function(...) file.path(cfg$data_root, "raw", ...)
derived_path <- function(...) file.path(cfg$data_root, "derived", ...)
result_path  <- function(...) file.path(cfg$results_root, ...)

# Create a directory (and parents) if needed; returns it.
ensure_dir <- function(dir) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  dir
}

# Compute `expr` once and store it at `path` (.rds). Delete the file to recompute.
cached <- function(path, expr) {
  if (file.exists(path)) return(readRDS(path))
  value <- expr
  ensure_dir(dirname(path))
  saveRDS(value, path)
  value
}
