# HEC analyses (Hierarchically Explicit Classification)

Analysis code for "The Role of Hierarchically Explicit Deep Learning Models in Species Classification". R, run from the RStudio project `HEC_Analyses.Rproj` or from this folder (the project root is found by `here` through the `.here` file; `Rscript run_all.R` runs everything, see "Running everything").

## Provenance

The analyses read the immutable ERDA snapshots pinned in `data/snapshots.json`, a selection from the producer registry `publication/experiments/erda-snapshots.json` of `mini_trainer` (which records each snapshot's export and training commits, manifest hashes and, for the bootstrap replicates, the `mini_metrics` version, 0.4.1). How the evidence was produced is described in `publication/experiments/README.md`. Run the analyses from this folder (`renv` and `here` both depend on it).

## Setup

1. Open the project in R and run `renv::restore()` (the packages and versions are pinned in `renv.lock`; `helpers/packages.R` lists what the code imports). Without `renv`, install `box`, `here` and `progressr`, plus the packages imported in `helpers/packages.R`.
2. Install the CMU Serif font and register it once with `extrafont::font_import(pattern = "CMU")`. The math font (Latin Modern Math) is shipped in `data/fonts/` and registered automatically.
3. Optional: copy `config.local.example.R` to `config.local.R` (git-ignored) to override entries of `cfg` in `helpers/config.R`, e.g. `data_root` (keep large data outside OneDrive) and `base_url` (read-only ERDA link of the snapshot).

## Writing an analysis script

Every script, at any depth, starts with

```r
source(here::here("shared.R"))
```

`shared.R` checks `box`, attaches `here`, then sources the files listed in `helpers/scripts` in that order (packages, config, data access, export, plot theme, labels and the analysis helpers) and sets the seed (`cfg$seed`). Add a helper by creating the file and listing it there (paths are relative to `helpers/`, subfolders allowed).

## Helpers

| Path | Content |
|---|---|
| `helpers/packages.R` | Every package function used in the project, imported by name with `box` (add new imports here) |
| `helpers/{config,io,data,export,tables,labels}.R` | Configuration, paths and caching, ERDA access, manuscript export, comparison-table and win/loss summary helpers, label sets |
| `helpers/ggplot.R` | The plot theme and the fixed colours: `head_colours` (one colour per model in every figure; edit there only) and the win/loss colours in `helpers/tables.R` |
| `helpers/calibration.R` | Per-image correctness and confidence of the saved predictions, for the calibration scripts |
| `helpers/compare_models.R` | Winners and losers among competing models from paired bootstrap resamples (`compare_models()`): the contender and clear-winner rules and their Dirichlet uncertainty model |
| `helpers/embeddings/` | Head weights and embeddings of the embedding studies (`heads.R`, `family.R`), cosine/z-score distances (`geometry.R`), taxonomic alignment of clusters (`alignment.R`), spherical median, tangent PCA and polar placement (`projection.R`), and the coordinates and text snippets of the embedding-structure figures (`structure.R`) |
| `helpers/species_images/` | The species-image pipeline: web and iNaturalist access (`web.R`, `inaturalist.R`), the Gemini client (`gemini.R`), the picks (`picks.R`), the screening of the candidates (`screening.R`), the review sheet (`selection_sheet.R`) and the thumbnails (`thumbnails.R`) |
| `helpers/ablation/` | Run metadata and the factorial design (`design.R`), the replicate tables pooled over studies (`replicates.R`), the dot matrix and row axis shared by the ablation figures (`figures.R`) |

## Scripts

`scripts/` has one analysis per script, grouped by the part of the paper it feeds and numbered in reading order. Every script starts with `source(here::here("shared.R"))` and a few settings. Plots stay in the scripts; computations that several scripts need live in the helpers. `docs/` has notes on the data (e.g. the known taxonomy errors).

| Script | Writes to `results/` (manuscript) |
|---|---|
| `1_main_comparison.R` | `tables/main_comparison_<setting>.tex` (main backbone, datasets as row groups, ranks nested), `tables/appendix_comparison_<dataset>_<setting>.tex` (other backbones), `tables/appendix_win_loss.tex`, `figures/main_comparison_wins_losses`, `values/main_comparison_win_loss` |
| `2_ablation/ablation_win_loss.R` | `figures/ablation_win_loss_<group>[_zero]` |
| `2_ablation/ablation_vs_standard.R` | `figures/ablation_vs_standard[_zero]` |
| `2_ablation/hierarchy_contrast.R` | `values/hierarchy_contrast`: the accuracy effect of the hierarchy (hierarchical minus flat) in the architecture comparison and in each ablation seed, and their difference with an interval |
| `3_embedding/taxonomic_alignment.R` | `figures/composite_taxonomic_alignment`, `values/taxonomic_alignment` (the clustering methods, the number and size of the subsamples, and the median and 95% interval of the AMI gain over the flat head per dataset, rank and head): do clusters of the species prototypes align with the taxonomy (the quantitative claim over all families) |
| `3_embedding/embedding_structure.R` | `figures/embedding_structure`, `figures/embedding_photos_<dataset>`, `values/embedding_taxa`, `values/embedding_species_labels`, `values/embedding_caption`: the embedding structure of one example family per dataset |
| `3_embedding/outlyingness.R` | `figures/outlyingness_<dataset>`: species far from the rest of a head's embedding space |
| `3_embedding/family_selection.R` | none (diagnostic for choosing the example family) |
| `4_calibration/calibration_curves.R` | `figures/calibration_<dataset>`: reliability curves (accuracy minus confidence) per backbone, head and rank of the architecture-comparison models; prints the expected calibration error |
| `4_calibration/calibration_vs_flat.R` | `values/calibration_error`: calibration error (root of the debiased mean squared gap) per dataset, head and rank group (species, higher ranks), its difference to the flat head with a delta-method interval, paired over the test images, and how the backbones split; `tables/calibration_verdicts` (heads against flat over all datasets and backbones) and `tables/appendix_calibration_<dataset>` (calibration error of every head per rank and backbone, marked better or worse than flat) |

The photo version of the embedding-structure figure needs one open-licence photo per species, made in three steps in `3_embedding/prep/` (each writes to `data/derived/species_images/<study>/`; the family per study is `cfg$example_families` in `helpers/config.R`). Run them in this order before `embedding_structure.R`:

1. `species_candidates.R`: iNaturalist candidates (research grade, CC0, CC BY or CC BY-SA) with a `candidates.html` sheet; species without any are listed in `no_candidates.csv` and get no photo.
2. `species_screening.R`: a vision model screens the candidates; you pick one per species in `selected.html` and save the exported picks as `data/species_picks/<study>.csv` (tracked; the picks of an earlier family are ignored), which writes `chosen.csv`.
3. `species_thumbnails.R`: crops each chosen photo to its focal organism (`box_grow` there corrects a box by hand after checking `review.html`) and writes the thumbnails with `thumbnails.csv` and `attribution.csv`, copied to the tracked `data/species_thumbnails/<study>/` that `embedding_structure.R` and the credits read (so a fresh clone makes the photo figures without the pipeline or an API key), and the appendix section on the photograph credits (`sections/photo_credits`, label `sec:licenses`: per dataset and creator the species whose image they supplied; `values/species_photo_contributors`: the creators alone, for an acknowledgement; CC0 photos are credited to the observer's iNaturalist username).

## Running everything

`Rscript run_all.R` (from the project root) runs the scripts above in reading order (not `family_selection.R`, a diagnostic), each in its own environment, with a progress bar and a table of the time and outcome of every script at the end. The options are at the top of the file: `do_recompute` (delete the caches of the expensive computations listed in `recompute_caches`: alignment, calibration, families, species points), `dry_run`, `do_photos` (the photo candidates and screening, off by default since they need a Gemini key and manual picks; `species_thumbnails.R` always runs and only fetches what is missing), `skip` (script names or groups), `stop_on_error` and `verbose`. It needs the `progressr` and `cli` packages.

## Attribution

Asger Svenning has full attribution for all code by default; files need no header. A file gets an agent attribution header only when an AI agent made a meaningful contribution to it, i.e. wrote substantial new functionality or logic. Formatting, non-functional refactoring (moving, renaming, splitting code), minor changes and basic data wrangling do not count. The header goes at the top of the file:

```r
# Design & Idea: Asger Svenning
# 09-10-2026
# Written by <model name> (<effort>)
```

The date is DD-MM-YYYY. Agents follow this rule when they create or edit files here.

## Data layers

| Layer | Path | Content | Git |
|---|---|---|---|
| raw | `data/raw/<folder>/` | Files downloaded from ERDA, immutable (`data_file()`, `download_data()`) | ignored |
| derived | `data/derived/` | Regenerable intermediate results (`cached()`) | ignored |
| inputs | `data/snapshots.json`, `data/species_picks/`, `data/species_thumbnails/`, `data/known_taxonomy_errors.csv`, `data/fonts/` | The pinned data index, the photo picks and thumbnails, the known label errors and the math font | tracked |
| results | `results/` | Figures, tables and values used in the manuscript | tracked |

Scripts read raw or derived and write derived or results, never the reverse. Deleting `derived/` and `results/` and rerunning the scripts rebuilds them from ERDA and the tracked inputs (the photo pipeline's own files in `derived/species_images/` excepted).

## Reading the ERDA data

`data/snapshots.json` (tracked) pins the ERDA folders used: their role (`evidence`, `replicates`, `checkpoints`, `heads`), studies and manifest SHA-256. Each folder's `manifest.json` is checked against that hash, so results do not depend on a moving branch. Update the file when a new folder is published.

`docs/studies.md` describes every study id (dataset, design, seeds, which ERDA folder).

Accessors in `helpers/data.R` return lazy DuckDB tables (`filter`, `select`, then `collect()`):

```r
runs("plantnet-16")
scores("plantnet-16", "full_seed42", rank = 0) |> select(image_id, c0) |> collect()
replicates("plantnet-16") |> filter(setting == "zero") |> collect()
```

To see what exists, use `studies()` (study, role, folder; no network), `contents("plantnet-16")` (every file with kind, run, rank, size and whether it is cached) and `schema("scores", "plantnet-16")` (columns and types; `schema(study = ...)` lists the kinds). For example `contents("plantnet-16") |> count(role, kind)`.

With `cfg$remote = TRUE` (default) a parquet file with no local copy is read directly over HTTPS, transferring only the requested columns. `download_data(folder, pattern)` caches files in `data/raw/` (verified by SHA-256) for repeated or full reads; downloads over `cfg$max_download_mb` need `allow_large = TRUE`. Remote reads are not checksum-verified. Replicate 0 in the bootstrap tables is the 90% reporting partition, not the full test set.

## Exporting to the manuscript

`results/` is the whole Overleaf payload: upload it as-is.

| Function | Writes | In LaTeX |
|---|---|---|
| `save_figure(plot, name)` | `figures/<name>.pdf` (and a PNG preview in `figure_previews/`, untracked) | `\resultgraphic[width=\linewidth]{name}` |
| `save_table(tabular, name, caption)` | `tables/<name>.tex` (a `tabular` only) and the generated caption as a value | `\resulttable{name}` inside your own `table` environment; `\resultcaption{name}` for the caption text; `\autoresulttable{name}[placement][label]` for the whole table (default label `tab:name`) |
| `save_results(name, ...)` | `values/<name>.tex` (nested values) | `\result[name, key, ...]` |
| `save_section(lines, name)` | `sections/<name>.tex` (a whole section) | `\resultsection{name}` |

Manuscript preamble: `\input{results/setup}` (a copy of `latex/setup.tex`, refreshed by every export). Never type numbers or tables into the manuscript by hand. Compile the manuscript with XeLaTeX or LuaLaTeX: the photo credits contain non-Latin names.

```r
save_results("experiment_1", accuracy = list(flat = 46.45, independent = 46.06))
# LaTeX: \result[experiment_1, accuracy, flat]
```

A hyphen in a `\result` key stands for an underscore (`\result[embedding-caption, global-lepi, text]`), which avoids the editor's underscore warning.
