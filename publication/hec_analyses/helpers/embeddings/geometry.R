# Cosine similarity -> distance, for the clustering of head weights (alignment.R).

# z-score of a cosine similarity between random unit vectors in d dimensions
# (the angle is approximately N(pi/2, 1/(d - 2))).
cosine_to_zscore <- function(cs, d, eps = 1e-12) {
  sqrt(d - 2) * (acos(-pmin(pmax(cs, -1 + eps), 1 - eps)) - pi / 2)
}

# Chi-squared(df) quantile of the null upper-tail probability of z. df = 2 is the
# closed form -2 log Phi(z). The choice is arbitrary and only matters for methods that
# use distance values (average, McQuitty, PAM, HDBSCAN), not for single/complete linkage.
z_distance <- function(z, df = 2) {
  log_cdf <- pnorm(z, log.p = TRUE)
  if (df == 2) {
    -2 * log_cdf
  } else {
    qchisq(log_cdf, df, lower.tail = FALSE, log.p = TRUE)
  }
}
