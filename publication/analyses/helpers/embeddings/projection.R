# Projections of unit vectors (prototypes, embeddings) of one family to the plane.

# Extrinsic spherical mean of unit rows, optionally weighted
spherical_mean <- function(X, w = rep(1, nrow(X))) {
  m <- colSums(X * w)
  m / sqrt(sum(m^2))
}

# Weights that give every group the same total weight (e.g. every genus, whatever its species count)
balanced_weights <- function(groups) {
  1 / as.numeric(table(groups)[as.character(groups)])
}

# Log map at the unit vector `mu`: tangent vectors in the ambient space, length = angle to `mu`
log_map <- function(X, mu) {
  cosine <- pmin(pmax(drop(X %*% mu), -1), 1)
  tangent <- X - outer(cosine, mu)
  norm <- sqrt(rowSums(tangent^2))
  tangent * ifelse(norm < 1e-12, 0, acos(cosine) / norm)
}

# Exponential map at the unit vector `mu`: the point reached by the tangent vector `v`
exp_map <- function(mu, v) {
  length <- sqrt(sum(v^2))
  if (length < 1e-12) mu else cos(length) * mu + sin(length) * v / length
}

# Spherical (geometric) median of unit rows: the point minimising the weighted sum of angular
# distances, by Weiszfeld iterations in the tangent space. Unlike the mean, one distant point
# cannot pull it away from the bulk.
spherical_median <- function(
  X,
  w = rep(1, nrow(X)),
  iterations = 100,
  tol = 1e-10
) {
  mu <- spherical_mean(X, w)
  for (i in seq_len(iterations)) {
    V <- log_map(X, mu)
    step <- w / pmax(sqrt(rowSums(V^2)), 1e-8)
    new <- exp_map(mu, colSums(V * step) / sum(step))
    converged <- sum((new - mu)^2) < tol
    mu <- new / sqrt(sum(new^2))
    if (converged) break
  }
  mu
}

# Principal geodesic analysis: PCA of the log map of `P` at its (weighted) spherical mean, then
# `project()` any unit vectors (e.g. images) with the same basis. `w` weights the rows of `P`.
# `robust = TRUE`: the centre is the spherical median and every tangent vector enters the PCA with
# unit length (spherical PCA, Locantore et al. 1999), so that no point has more influence than its
# direction, however far out it lies. `explained` is then the share of the variance of those unit
# directions, not of the angles.
tangent_pca <- function(P, w = rep(1, nrow(P)), k = 2, robust = FALSE) {
  mu <- if (robust) spherical_median(P, w) else spherical_mean(P, w)
  V <- log_map(P, mu)
  if (robust) {
    V <- V / pmax(sqrt(rowSums(V^2)), 1e-12)
  }
  s <- svd(V * sqrt(w / sum(w)), nu = 0, nv = k)
  list(mu = mu, basis = s$v, explained = s$d^2 / sum(s$d^2))
}

project <- function(fit, X) {
  Z <- log_map(X, fit$mu) %*% fit$basis
  `colnames<-`(Z, paste0("PC", seq_len(ncol(Z))))
}

# Polar placement of the unit rows X around the centre of `fit`, for a plot with the centre in the
# middle: `theta`, the exact angle to the centre (radians; the length of the log map), and `angle`,
# the direction of the projection of the tangent vector onto the plane of `fit` (radians, clockwise
# from the top, as in coord_polar). Only `angle` is approximate: it keeps the plane's global layout
# but not all neighbourhoods. `R` is an optional orthogonal 2 x 2 matrix (rotation or reflection,
# e.g. from procrustes_rotation()) applied to the plane first; it only shifts or mirrors `angle`.
polar_placement <- function(fit, X, R = diag(2)) {
  Z <- project(fit, X) %*% R
  tibble(
    theta = acos(pmin(pmax(drop(X %*% fit$mu), -1), 1)),
    angle = atan2(-Z[, 1], -Z[, 2]),
    in_plane = sqrt(rowSums(Z^2)) # length of the projected vector (radians): how much `angle` is based on
  )
}

# Orthogonal matrix R (rotation or reflection) that brings the configuration Z closest to `ref`
# (least squares, no scaling): Z %*% R. Both are in the plane of their own tangent point.
procrustes_rotation <- function(Z, ref, w = rep(1, nrow(Z))) {
  s <- svd(crossprod(Z * w, ref))
  s$u %*% t(s$v)
}
