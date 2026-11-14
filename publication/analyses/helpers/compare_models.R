# Winners and losers among K competing models from paired bootstrap resamples.
#
# Resamples recompute the metric from fixed predictions and labels on a fixed test set. In
# resample r the winner credit of model k is c_kr = 1[s_kr = max_j s_jr] / #{j : s_jr = max}
# (ties share the credit; scores are rounded first), and the win share theta_k = E[c_kr | data]
# is estimated by w_k / N. The shares sum to one, so under exchangeability each model has the
# chance share 1/K. Loss shares are defined in the same way with the minimum.
#
# Uncertainty model (a fixed convention): theta | w ~ Dirichlet(w_1 + 1, ..., w_K + 1), i.e. a
# uniform prior on the simplex. It captures the Monte Carlo error of the N resamples, not the
# uncertainty of the test data. With ties (fractional credits) it is a working approximation: it
# respects the sum-to-one constraint, but the accumulated credits do not carry the variability
# caused by the ties.
#
#   Winner ("contender"):  p_k = P(theta_k >= 1/K | w) > alpha, with the marginal
#                          theta_k | w ~ Beta(w_k + 1, N - w_k + K - 1).
#   Clear winner:          q_k = P(max_{j != k} theta_j < 1/K | w) > 1 - alpha, i.e. the model is
#                          jointly established as the only eligible contender. This implies that
#                          the model is a winner, that every rival is not, and that at most one
#                          model is a clear winner (alpha < 1/2).

# ---- Joint probability by simulation ---------------------------------------------------------

# P(all components except `k` of theta < `t`) for theta ~ Dirichlet(`shape`), by sequential Monte
# Carlo in batches until the Wilson interval (z = `z`) for the probability excludes `level`, or
# `max_draws` is reached. Reproducible: uses its own seed and restores the RNG state.
# Returns the estimate, its standard error, the draws used, and whether the decision against
# `level` is resolved (interval excludes `level`).
simulate_rivals_below <- function(
  shape,
  k,
  t,
  level,
  z = 4,
  batch = 20000,
  max_draws = 4e6,
  seed = 1L
) {
  K <- length(shape)
  had_seed <- exists(".Random.seed", envir = globalenv())
  old_seed <- if (had_seed) get(".Random.seed", envir = globalenv())
  on.exit(
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv())) {
      rm(".Random.seed", envir = globalenv())
    },
    add = TRUE
  )
  set.seed(seed)

  hits <- 0
  draws <- 0
  repeat {
    g <- matrix(rgamma(batch * K, rep(shape, batch)), batch, K, byrow = TRUE)
    rivals <- g[, -k, drop = FALSE] / rowSums(g)
    top <- rivals[cbind(seq_len(batch), max.col(rivals, ties.method = "first"))]
    hits <- hits + sum(top < t)
    draws <- draws + batch

    est <- hits / draws
    centre <- (est + z^2 / (2 * draws)) / (1 + z^2 / draws)
    half <- z * sqrt(est * (1 - est) / draws + z^2 / (4 * draws^2)) / (1 + z^2 / draws)
    resolved <- (centre - half > level) || (centre + half < level)
    if (resolved || draws >= max_draws) {
      break
    }
  }
  list(
    est = est,
    se = sqrt(est * (1 - est) / draws),
    draws = draws,
    resolved = resolved
  )
}

# ---- Classification of one set of shares -----------------------------------------------------

# Classify the K models with (possibly fractional) credit totals `w` out of `n` resamples.
# Returns one row per model:
#   p                  P(theta_k >= 1/K | w), the contender probability
#   contender          p > alpha
#   q_lower, q_upper   deterministic bounds on q_k = P(max_{j != k} theta_j < 1/K | w):
#                        max(0, 1 - sum_{j != k} p_j, P(theta_k > 1 - 1/K))
#                          <= q_k <= min(p_k, 1 - max_{j != k} p_j)
#   q_est, q_se        simulated q_k and its standard error, only where the bounds do not decide
#   clear              q_k > 1 - alpha (decided by the bounds where possible, else by simulation)
#   resolved           FALSE only if the simulation budget ran out near 1 - alpha
classify_shares <- function(w, n, alpha = 0.05, ...) {
  K <- length(w)
  stopifnot(K >= 2, all(w >= 0), all(w <= n))
  t <- 1 / K
  level <- 1 - alpha

  p <- 1 - pbeta(t, w + 1, n - w + K - 1)

  top <- sort(p, decreasing = TRUE)
  max_other <- ifelse(p == top[1], top[2], top[1]) # max of p over the other models
  q_upper <- pmin(p, 1 - max_other)
  dominant <- 1 - pbeta(1 - t, w + 1, n - w + K - 1) # P(theta_k > 1 - 1/K)
  q_lower <- pmax(0, 1 - (sum(p) - p), dominant)

  clear <- q_lower > level
  undecided <- !clear & q_upper > level
  q_est <- q_se <- rep(NA_real_, K)
  resolved <- rep(TRUE, K)
  for (k in which(undecided)) {
    sim <- simulate_rivals_below(w + 1, k, t, level, ...)
    q_est[k] <- sim$est
    q_se[k] <- sim$se
    clear[k] <- sim$est > level
    resolved[k] <- sim$resolved
  }

  contender <- p > alpha
  # The defining guarantees: a clear winner is a contender, its rivals are not, and it is unique
  stopifnot(
    sum(clear) <= 1,
    !any(clear) || (sum(contender) == 1 && contender[clear])
  )

  tibble(
    p = p,
    contender = contender,
    q_lower = q_lower,
    q_upper = q_upper,
    q_est = q_est,
    q_se = q_se,
    clear = clear,
    resolved = resolved
  )
}

# ---- Comparison of models from resample scores -----------------------------------------------

# One group of competing models: a long table with columns `model`, `replicate` and `value`,
# where replicate 0 (if present) is the point estimate and replicates >= 1 are bootstrap
# resamples of the same test set (the same draws for every model, so the comparison within a
# resample is paired). Values are rounded to `digits` decimals before the credits are assigned.
# Returns per model the estimate (`est`, NA without replicate 0), the bootstrap SD (`sd`) and mean
# (`boot_mean`), the number of resamples (`n_boot`), the credit totals `wins` and `losses` with
# shares `p_best` and `p_worst`, the flags
#   `tied` (winner) and `dominant` (clear winner), `worst` (loser) and `dominant_worst`
#   (clear loser),
# and the details of classify_shares() for both sides, prefixed `win_` and `loss_`.
compare_models <- function(g, digits = 3, alpha = 0.05, ...) {
  g <- g |>
    mutate(
      model = as.character(model),
      rounded = round(value, digits = digits)
    )
  point <- g |> filter(replicate == 0) |> pull(value, model)
  boot <- g |> filter(replicate > 0)
  stopifnot(
    !anyDuplicated(boot[c("model", "replicate")]), # one value per model and resample
    n_distinct(count(boot, model)$n) == 1
  )

  totals <- boot |>
    mutate(
      credit = (rounded == max(rounded)) / sum(rounded == max(rounded)),
      debit = (rounded == min(rounded)) / sum(rounded == min(rounded)),
      .by = replicate
    ) |>
    summarize(
      sd = sd(value),
      boot_mean = mean(value),
      wins = sum(credit),
      losses = sum(debit),
      n_boot = n(),
      .by = model
    )

  win <- classify_shares(totals$wins, totals$n_boot[1], alpha, ...) |>
    rename_with(\(x) str_c("win_", x))
  loss <- classify_shares(totals$losses, totals$n_boot[1], alpha, ...) |>
    rename_with(\(x) str_c("loss_", x))

  totals |>
    mutate(
      p_best = wins / n_boot,
      p_worst = losses / n_boot,
      est = unname(point[model])
    ) |>
    bind_cols(win, loss) |>
    mutate(
      tied = win_contender,
      dominant = win_clear,
      worst = loss_contender,
      dominant_worst = loss_clear
    )
}
