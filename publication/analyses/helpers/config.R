# Project configuration: plain values only, no logic.
# Machine-specific overrides go in `config.local.R` (git-ignored) at the project
# root; it must return a list, e.g. list(data_root = "D:/data"). See
# `config.local.example.R`.

cfg <- list(
  # Pinned index of the ERDA folders (names, roles, studies, manifest hashes).
  snapshots = here("data", "snapshots.json"),
  # Override of the read-only ERDA URL in the index; NULL = use the index's.
  base_url = NULL,
  # TRUE: read parquet straight from ERDA when there is no local copy.
  # FALSE: download (and verify) files to data/raw/ first.
  remote = TRUE,
  # Downloads above this size need allow_large = TRUE.
  max_download_mb = 500L,
  # DuckDB limits for the lazy parquet reads (DuckDB defaults to 80% of the RAM and all cores,
  # which a scan of large embedding files can exhaust). Raise them in config.local.R if you can.
  duckdb_memory_limit = "4GB",
  duckdb_threads = 4L,
  # Layers (see README): raw/ (downloaded, immutable), derived/ (regenerable),
  # results/ (final figures and tables).
  data_root = here("data"),
  results_root = here("results"),
  # Figure export defaults (inches / dpi of the PNG preview), and decimals for
  # numeric LaTeX macros.
  fig_width = 7,
  fig_height = 4.5,
  fig_dpi = 300,
  digits = 2L,
  seed = 1L,
  # The math font (shipped in data/fonts/) for the arrows of the win/loss axis titles (helpers/ggplot.R)
  math_font_file = here("data", "fonts", "latinmodern-math.ttf"),
  # Optional: your contact (e.g. an email address) for the user agent of the web requests
  # (iNaturalist, GBIF; helpers/species_images/web.R)
  contact = NULL,
  # The family per study shown in the embedding figures and the photo pipeline (chosen with
  # scripts/3_embedding/family_selection.R and the support of its species).
  example_families = c(`gefion-plantnet` = "2410", `gefion-global-lepi` = "8863"),
  # Ranks left out of the calibration analyses (calibration_curves.R, calibration_vs_flat.R): the
  # Pl@ntNet classes are few and extremely imbalanced, so nearly every image has a confidence near 1
  calibration_skip_ranks = list(`gefion-plantnet` = "class"),
  # The dataset (id of label_sets$dataset) that a study is drawn from: the main-comparison and
  # embedding studies, and the ablation studies used in the paper (`plantnet` is the full dataset,
  # `lepi-512` and `lepi-1513` are subsets of 512 and 1513 species of Global Lepidoptera).
  study_dataset = c(
    `gefion-plantnet` = "plantnet",
    `gefion-global-lepi` = "global_lepi",
    `gefion-flemming` = "flemming",
    plantnet = "plantnet",
    `lepi-512` = "global_lepi",
    `lepi-1513` = "global_lepi"
  )
)

local_config <- here("config.local.R")
if (file.exists(local_config)) {
  cfg <- modifyList(cfg, source(local_config, local = TRUE)$value)
}
