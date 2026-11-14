# Alignment of post-hoc clusters of species prototypes with the taxonomy.
# Inputs are distance matrices (one per head, see cosine_to_zscore() / z_distance()) and
# `taxa`, a table of taxon keys per species (one column per rank, genus to class).

# Clustering methods: (distance, ks) -> label matrix, one column per k.
cluster_methods <- function(min_pts = 5) {
  list(
    average = \(d, ks) cutree(fast_hclust(d, "average"), ks),
    complete = \(d, ks) cutree(fast_hclust(d, "complete"), ks),
    # PAM with BUILD initialisation by FastPAM (deterministic): equal quality to the original PAM at
    # a fraction of the time, and better optima than cluster::pam(variant = "faster")
    pam = \(d, ks) {
      n <- attr(d, "Size")
      sapply(ks, \(k) {
        # k = 1 and k = n have one partition each (and are edge cases of the C++ code)
        if (k < 2) {
          rep(1L, n)
        } else if (k >= n) {
          seq_len(n)
        } else {
          # maxiter: the default (0) is unlimited, and ties (the path sums have many pairs at distance
          # 0) can make the swaps cycle; 100 is the FasterPAM paper's cap
          fastpam(
            as.vector(d),
            n,
            k,
            maxiter = 100L,
            initializer = "BUILD"
          )@assignment
        }
      })
    },
    # HDBSCAN's own flat partition (no k); noise points become singletons. The same
    # partition is compared with every rank.
    "hdbscan (flat)" = \(d, ks) {
      labels <- hdbscan(d, minPts = min_pts)$cluster
      noise <- labels == 0
      labels[noise] <- max(labels) + seq_len(sum(noise))
      matrix(labels, nrow = length(labels), ncol = length(ks))
    }
  )
}

# Methods used unless a selection is given (any of names(cluster_methods()))
default_cluster_methods <- names(cluster_methods())

# AMI of each method's partitions against each rank (partition at that rank's number of taxa)
alignment_ami <- function(d, taxa, methods = default_cluster_methods) {
  method.options <- cluster_methods()
  methods <- map(methods, function(m) {
    if (!(m %in% names(method.options))) {
      stop(str_glue(
        'Selected method="{m}" is not one of the valid options: {names(method.options)}.'
      ))
    }
    method.options[[m]]
  }) |>
    set_names(methods)

  d <- as.dist(d)
  ks <- map_int(taxa, n_distinct)
  map(methods, \(method) {
    clusters <- method(d, ks)
    tibble(
      rank = names(taxa),
      ami = map2_dbl(taxa, seq_along(taxa), \(x, i) {
        AMI(clusters[, i], x)
      })
    )
  }) |>
    list_rbind(names_to = "method")
}

# The distance matrix of the rows of `W` (unit prototypes): z-score of the cosines against random
# directions of the full dimension, as a distance. Only subsets are ever needed, so no matrix of all
# species is built.
prototype_distance <- function(W) {
  z_distance(cosine_to_zscore(tcrossprod(W), ncol(W)))
}

# Rows of a random subsample of genera (`genus`: the genus of every species), drawn without
# replacement until they hold at least `size` species. Genera keep their species (sampling species
# would leave most of them alone in large datasets), except that a genus larger than `cap` is
# represented by `cap` random species, so that a few huge genera do not fill the subsample.
sample_genera <- function(genus, size, cap = 64) {
  rows <- split(seq_along(genus), genus)
  rows <- rows[sample(length(rows))] |>
    map(\(r) r[sample.int(length(r), min(cap, length(r)))])
  keep <- seq_len(which(cumsum(lengths(rows)) >= size)[1] %||% length(rows))
  sort(unlist(rows[keep], use.names = FALSE))
}

# Paired subsampling over genera (see sample_genera(), with at most 64 species per genus): the same species, about `size`, for every head
# (`prototypes`: a named list of unit matrices, rows in the order of `taxa`, whose first column is the
# genus), re-clustering each time. A fixed size makes the cost and the AMI comparable between datasets
# of different size.
alignment_bootstrap <- function(
  prototypes,
  taxa,
  n_boot = 50,
  size = 512,
  methods = default_cluster_methods
) {
  map(
    seq_len(n_boot),
    .progress = TRUE,
    function(b) {
      idx <- sample_genera(taxa[[1]], size)
      map(
        prototypes,
        \(W) {
          alignment_ami(
            prototype_distance(W[idx, , drop = FALSE]),
            taxa[idx, ],
            methods
          )
        }
      ) |>
        list_rbind(names_to = "head") |>
        mutate(b = b)
    }
  ) |>
    list_rbind()
}
