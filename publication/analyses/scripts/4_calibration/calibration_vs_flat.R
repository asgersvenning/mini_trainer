source(here::here("shared.R"))

# Are the non-flat heads better or worse calibrated than the flat head? For every model and rank, the
# calibration error in equal-width confidence bins, as the root of the debiased mean squared gap:
#   theta = sum over bins of (share of images in the bin) * (gap^2 - variance of the gap estimate),
#   gap of a bin = mean over its images of (1 if the prediction is right, else 0) - confidence,
# the variance of the gap estimate being the variance of (right - confidence) in the bin divided by its
# number of images. Subtracting it makes theta unbiased, which the absolute-value ECE is not: it is
# inflated by the noise of every bin, differently for each head (see calibration_curves.R for the ECE).
# Per dataset, head and rank group (species; the ranks above species) theta is averaged over the
# backbones and ranks, and its root is the calibration error reported (same unit as the ECE).
#
# The uncertainty of the difference to the flat head is from the influence function of theta: an image i in
# a bin with gap g and residual r_i = (right - confidence) changes theta by psi_i = 2 g r_i - g^2 - theta
# if it is counted slightly more. The variance of an estimate is the variance of psi_i over the images
# divided by their number, and for the difference between two heads on the same images that of
# psi_i(head) - psi_i(flat), so the pairing is built in.
#  - Better or worse (the verdict): from the interval of the difference of theta itself, which is always
#    defined (the squared scale needs no root).
#  - The interval of the difference of the calibration errors (the roots) is by the delta method, which
#    divides by the root of theta and is only reliable where theta is far from 0: its relative error is
#    about 1 / (8 z^2) for z = theta / its standard error, under 1% for z >= `min_z_theta`. For smaller z of
#    either head the interval is withheld ("interval unavailable"); the estimate is the difference of the
#    roots, a theta at or below 0 counting as 0, which does not mean that the head is exactly calibrated.
# Per backbone and per rank the same is done; the counts of backbones where a head is better, worse or
# inconclusive are saved with the pooled results.
studies_used <- c("gefion-global-lepi", "gefion-plantnet")
n_bins <- 30
min_z_theta <- 4

# theta of one model and rank from the confidence bin (1 to n_bins) and residual (right - confidence) of
# every image, and the influence psi_i of every image on it
calibration_error <- function(bin, residual) {
  n <- tabulate(bin, n_bins)
  per_bin <- \(x) {
    s <- numeric(n_bins)
    sums <- rowsum(x, bin)
    s[as.integer(rownames(sums))] <- as.vector(sums)
    s
  }
  gap <- per_bin(residual) / pmax(n, 1)
  variance <- (per_bin(residual^2) - n * gap^2) / pmax(n - 1, 1)
  use <- n >= 2
  theta <- sum(n[use] / length(bin) * (gap[use]^2 - variance[use] / n[use]))
  list(theta = theta, psi = 2 * gap[bin] * residual - gap[bin]^2 - theta)
}

# TRUE for the rows of `cells` in the same cell of `by` as row i
in_cell_of <- \(cells, i, by) {
  Reduce(`&`, map(by, \(k) cells[[k]] == cells[[k]][i]))
}

# One unit per run and rank (all ranks, in the order of the images, the same in every unit): theta and the
# influence of every image. Slow (the predictions of every run are read), so cached per study and number
# of bins in data/derived/calibration/ (delete to recompute).
calibration_units <- function(study) {
  truth <- calibration_truth(study)
  image_ids <- sort(unique(truth$image_id))
  map(pull(runs(study), run_id), \(run_id) {
    split(run_calibration(study, run_id, truth), ~rank) |>
      map(\(r) {
        stopifnot(setequal(r$image_id, image_ids))
        r <- r[match(image_ids, r$image_id), ]
        c(
          list(run_id = run_id, rank = r$rank[1]),
          calibration_error(
            confidence_bin(r$confidence, n_bins),
            as.numeric(r$correct) - r$confidence
          )
        )
      })
  }) |>
    list_flatten()
}

compare_heads <- function(study) {
  units <- cached(
    derived_path("calibration", str_c(study, "_", n_bins, "_units.rds")),
    calibration_units(study)
  ) |>
    keep(\(u) !u$rank %in% cfg$calibration_skip_ranks[[study]])
  n_images <- length(units[[1]]$psi)
  info <- tibble(
    head = map_chr(units, \(u) run_head(u$run_id)),
    backbone = map_chr(units, \(u) run_backbone(u$run_id)),
    rank = map_chr(units, "rank"),
    group = if_else(map_chr(units, "rank") == "species", "species", "higher")
  )

  # The calibration error of every head and the difference to the flat head, pooled over the backbones
  # and ranks of each cell of `by` ("group", or also "backbone"): theta and psi are the means over the
  # cell's units
  root <- \(p) sqrt(max(p$theta, 0))
  root_psi <- \(p) {
    if (p$theta > 0) p$psi / (2 * root(p)) else rep(NA_real_, length(p$psi))
  }
  se_of <- \(psi) stats::sd(psi) / sqrt(n_images)
  z <- stats::qnorm(0.975)
  pool <- \(by) {
    cells <- distinct(info, across(all_of(c(by, "head"))))
    in_cell <- \(cell) {
      Reduce(`&`, map(names(cell), \(k) info[[k]] == cell[[k]]))
    }
    cells <- mutate(
      cells,
      p = map(seq_len(nrow(cells)), \(i) {
        u <- units[in_cell(cells[i, ])]
        list(
          theta = mean(map_dbl(u, "theta")),
          psi = reduce(map(u, "psi"), `+`) / length(u)
        )
      }),
      flat = map(seq_len(nrow(cells)), \(i) {
        p[[which(in_cell_of(cells, i, by) & head == "flat")]]
      })
    )
    cells |>
      mutate(
        mean = map_dbl(p, root) * 100,
        estimate = map2_dbl(p, flat, \(a, f) root(a) - root(f)) * 100,
        verdict = map2_chr(p, flat, \(a, f) {
          d <- a$theta - f$theta
          se <- se_of(a$psi - f$psi)
          case_when(
            d - z * se > 0 ~ "worse",
            d + z * se < 0 ~ "better",
            .default = "inconclusive"
          )
        }),
        z_theta = map_dbl(p, \(a) a$theta / se_of(a$psi)),
        z_flat = map_dbl(flat, \(f) f$theta / se_of(f$psi)),
        se = map2_dbl(p, flat, \(a, f) se_of(root_psi(a) - root_psi(f))) * 100,
        reliable = pmin(z_theta, z_flat) >= min_z_theta,
        lower = if_else(reliable, estimate - z * se, NA_real_),
        upper = if_else(reliable, estimate + z * se, NA_real_)
      ) |>
      select(
        all_of(by),
        head,
        mean,
        estimate,
        lower,
        upper,
        verdict,
        z_theta,
        z_flat
      )
  }
  list(
    pooled = pool("group"),
    by_backbone = pool(c("group", "backbone")),
    by_rank = pool(c("rank", "backbone"))
  )
}

results <- map(set_names(studies_used), compare_heads)

for (study in studies_used) {
  cat(
    "\n",
    study,
    ": calibration error (%) of the heads, and the difference to the flat head ",
    "(percentage points, 95% interval)\n",
    sep = ""
  )
  results[[study]]$pooled |>
    filter(head != "flat") |>
    print(n = Inf)
}
# For curiosity, not saved: the same per backbone
for (study in studies_used) {
  cat("\n", study, ": per backbone\n", sep = "")
  results[[study]]$by_backbone |>
    filter(head != "flat") |>
    mutate(backbone = relabel(backbone, "backbone")) |>
    arrange(group, backbone, head) |>
    print(n = Inf)
}
# Where the delta-method interval is withheld or close to it (z = theta / its standard error of either
# head, see the header), per rank and backbone
for (study in studies_used) {
  cat(
    "\n",
    study,
    ": cells per rank with z below ",
    2 * min_z_theta,
    " (interval withheld below ",
    min_z_theta,
    ")\n",
    sep = ""
  )
  results[[study]]$by_rank |>
    filter(head != "flat", pmin(z_theta, z_flat) < 2 * min_z_theta) |>
    arrange(rank, backbone, head) |>
    print(n = Inf)
}

# For the manuscript, per dataset, rank group and head (\result[calibration_error, plantnet, higher,
# hierarchical, difference]): `error` (%), and for the heads other than flat the `difference` to flat
# (percentage points with the 95% interval), its `verdict` and `backbone_verdicts`, how the backbones split
# (e.g. "2 better and 1 worse"); `backbones` of a dataset is their number
number <- \(x) str_c("$", sprintf("%.2f", x), "$")
# The difference with its interval as text, or "interval unavailable"
difference_text <- \(estimate, lower, upper) {
  if_else(
    is.na(lower),
    str_c(number(estimate), " (interval unavailable)"),
    str_c(number(estimate), " (", number(lower), " to ", number(upper), ")")
  )
}
# The verdicts of the backbones as text, e.g. "2 better and 1 worse"
backbone_split <- \(verdicts) {
  counts <- table(factor(verdicts, c("better", "worse", "inconclusive")))
  join_and(str_c(counts[counts > 0], " ", names(counts)[counts > 0]))
}
by_dataset <- map(
  set_names(studies_used, cfg$study_dataset[studies_used]),
  \(study) {
    pooled <- results[[study]]$pooled
    by_backbone <- results[[study]]$by_backbone
    c(
      map(set_names(unique(pooled$group)), \(g) {
        cells <- filter(pooled, group == g)
        map(set_names(cells$head), \(h) {
          cell <- filter(cells, head == h)
          error <- sprintf("%.2f", cell$mean)
          if (h == "flat") {
            return(list(error = error))
          }
          list(
            error = error,
            difference = difference_text(cell$estimate, cell$lower, cell$upper),
            verdict = cell$verdict,
            backbone_verdicts = backbone_split(by_backbone$verdict[
              by_backbone$group == g & by_backbone$head == h
            ])
          )
        })
      }),
      list(backbones = n_distinct(by_backbone$backbone))
    )
  }
)
# How the backbones split over all datasets and rank groups together, per head (\result[calibration_error,
# overall, hierarchical]): the pattern over all these cases is clearer than any single cell
overall <- map(results, "by_backbone") |>
  list_rbind() |>
  filter(head != "flat") |>
  summarise(split = backbone_split(verdict), .by = head)
print(overall)
do.call(
  save_results,
  c(
    list("calibration_error"),
    by_dataset,
    list(overall = as.list(set_names(overall$split, overall$head)))
  )
)

# Tables for the appendix ---------------------------------------------------------------------------

# What the calibration error and the verdicts are, for the captions
calibration_note <- str_c(
  "Calibration error is the square root of the debiased mean squared accuracy--confidence gap across ",
  "confidence bins. Better/worse: the pointwise 95\\% confidence interval for the paired difference in ",
  "squared calibration error lies entirely below/above 0; inconclusive: it includes 0."
)
all_cases <- map(results, "by_backbone") |>
  list_rbind() |>
  filter(head != "flat")
verdicts_caption <- str_c(
  "Calibration of the heads compared with the flat head: the number of cases (",
  n_distinct(all_cases$backbone),
  " backbones in each of ",
  length(results),
  " datasets) in which a head is better or worse calibrated than the flat head, at the species rank ",
  "and pooled over the higher ranks. ",
  calibration_note
)
all_cases |>
  summarise(split = backbone_split(verdict), .by = c(head, group)) |>
  pivot_wider(names_from = group, values_from = split) |>
  mutate(head = relabel(head, "head")) |>
  arrange(head) |>
  select(head, higher, species) |>
  kable(
    "latex",
    escape = FALSE,
    booktabs = TRUE,
    linesep = "",
    col.names = c("Head", "Higher ranks", "Species")
  ) |>
  save_table(
    "calibration_verdicts",
    caption = verdicts_caption
  )

# The calibration error of every head per dataset, rank and backbone; the heads other than flat are marked
# better (bold) or worse (italics) than the flat head by the paired comparison of the squared errors
for (study in studies_used) {
  dataset <- cfg$study_dataset[[study]]
  table <- results[[study]]$by_rank |>
    mutate(
      error = sprintf("%.2f", mean),
      error = case_when(
        head != "flat" & verdict == "better" ~ str_c("\\textbf{", error, "}"),
        head != "flat" & verdict == "worse" ~ str_c("\\textit{", error, "}"),
        .default = error
      ),
      rank = relabel(rank, "level"),
      head = relabel(head, "head"),
      backbone = relabel(backbone, "backbone")
    ) |>
    arrange(rank, backbone, head) |>
    select(rank, backbone, head, error) |>
    pivot_wider(names_from = head, values_from = error)
  caption <- str_c(
    "Calibration error (\\%) of the heads in the ",
    label_sets$dataset[[dataset]],
    " dataset, per rank and backbone. ",
    calibration_note,
    " Bold: better calibrated than the flat head; italics: worse calibrated; plain entries (not ",
    "including the flat column): inconclusive. Nonpositive estimates of the squared calibration error ",
    "are displayed as zero; this does not establish exact calibration."
  )
  table |>
    select(backbone, everything(), -rank) |>
    kable(
      "latex",
      escape = FALSE,
      booktabs = TRUE,
      linesep = "",
      col.names = c("Backbone", names(table)[-(1:2)])
    ) |>
    pack_rows(index = col_to_index(as.character(table$rank))) |>
    save_table(str_c("appendix_calibration_", dataset), caption = caption)
}
