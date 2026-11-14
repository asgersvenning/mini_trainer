# Small helpers for calling web APIs (JSON over GET) and writing HTML. Depends on packages.R.

`%||%` <- \(a, b) if (is.null(a)) b else a

# First element of `x` as a character, NA if it is absent
or_na <- \(x) {
  if (is.null(x) || length(x) == 0) NA_character_ else as.character(x[[1]])
}

# Query string of a named character vector or named arguments: key=value&key=value, URL-encoded
query <- \(...) {
  args <- c(...)
  str_c(names(args), "=", map_chr(args, URLencode, reserved = TRUE), collapse = "&")
}

# A key (taxon, photo, observation id) as a plain string: whole numbers without scientific notation or
# padding (numeric keys such as 100000 would otherwise print as "1e+05"), NA stays NA
key_string <- \(x) {
  if (is.numeric(x)) ifelse(is.na(x), NA_character_, formatC(x, format = "f", digits = 0)) else as.character(x)
}

# First element of `x` as a number, NA if it is absent
num_or_na <- \(x) suppressWarnings(as.numeric(or_na(x)))

# The web APIs ask for an identifying user agent; set `contact` in config.local.R to add yours
options(
  HTTPUserAgent = str_c(
    "structured-classification (research",
    if (!is.null(cfg$contact)) str_c("; ", cfg$contact),
    ")"
  )
)

# JSON of a URL (as lists), `wait` seconds after the previous request (default: none for GBIF, 1 s for
# the other hosts, e.g. iNaturalist, which asks for at most about one request per second). On HTTP 429
# (rate limit) it retries after 5, 15 and 45 s. NULL, with a message, if it still fails.
get_json <- function(url, wait = NULL, retries = 3) {
  wait <- wait %||% (if (str_detect(url, "api\\.gbif\\.org")) 0 else 1)
  for (attempt in seq_len(retries + 1)) {
    Sys.sleep(if (attempt == 1) wait else 5 * 3^(attempt - 2))
    limited <- FALSE
    result <- tryCatch(
      withCallingHandlers(
        fromJSON(url, simplifyVector = FALSE),
        warning = \(w) {
          if (str_detect(conditionMessage(w), "429")) {
            limited <<- TRUE
          }
          invokeRestart("muffleWarning")
        }
      ),
      error = \(e) NULL
    )
    if (!is.null(result) || !limited) {
      break
    }
    message("Rate limited, retrying: ", url)
  }
  if (is.null(result)) {
    message("Failed: ", url)
  }
  result
}

html_escape <- \(x) {
  x |>
    str_replace_all("&", "&amp;") |>
    str_replace_all("<", "&lt;") |>
    str_replace_all(">", "&gt;") |>
    str_replace_all("\"", "&quot;")
}
