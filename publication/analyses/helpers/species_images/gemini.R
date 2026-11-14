# Calls to the Gemini REST API with one inline image and a structured (JSON schema) answer.
# Needs the API key in the environment variable GEMINI_API_KEY. Depends on packages.R.

# Downloads photos concurrently (libcurl fetches many URLs at once) to temporary files: the paths,
# NA for the photos that could not be downloaded
download_photos <- function(urls) {
  files <- map_chr(urls, \(url) {
    tempfile(fileext = str_c(".", tolower(tools::file_ext(sub("\\?.*$", "", url)))))
  })
  try(
    suppressWarnings(
      utils::download.file(urls, files, method = "libcurl", mode = "wb", quiet = TRUE)
    ),
    silent = TRUE
  )
  ok <- file.exists(files) & !is.na(file.size(files)) & file.size(files) > 0
  if (any(!ok)) {
    message("  download failed: ", sum(!ok), " of ", length(urls), " photos")
  }
  files[!ok] <- NA_character_
  files
}

# The request for one image file: it goes inline, base64-encoded, in the body. `temperature` and
# `thinking_level` (NULL: the model's default) are passed on as they are.
gemini_request <- function(file, prompt, schema, model, temperature = NULL, thinking_level = NULL) {
  key <- Sys.getenv("GEMINI_API_KEY")
  if (key == "") {
    stop("Set the environment variable GEMINI_API_KEY", call. = FALSE)
  }
  ext <- tolower(tools::file_ext(file))
  body <- list(
    contents = list(list(parts = list(
      list(text = prompt),
      list(inline_data = list(
        mime_type = if (ext == "png") "image/png" else "image/jpeg",
        # base64_enc() wraps its output in lines; the API rejects line breaks in base64
        data = str_remove_all(base64_enc(readBin(file, "raw", file.size(file))), "\\s")
      ))
    ))),
    generationConfig = c(
      list(responseMimeType = "application/json", responseSchema = schema),
      if (!is.null(temperature)) list(temperature = temperature),
      if (!is.null(thinking_level)) list(thinkingConfig = list(thinkingLevel = thinking_level))
    )
  )
  request(str_c(
    "https://generativelanguage.googleapis.com/v1beta/models/", model, ":generateContent"
  )) |>
    req_headers(`x-goog-api-key` = key) |>
    req_body_json(body) |>
    req_retry(max_tries = 5) |> # retries 429 (quota) and 503 with a backoff
    # put the API's own error message in the R error (httr2 only says "HTTP 400 Bad Request")
    # (cut short: the API may quote the whole request, and formatting such an error takes minutes)
    req_error(body = \(resp) str_sub(resp_body_json(resp)$error$message, 1, 300))
}

# The model's answer text and token usage of a response
gemini_text <- \(response) {
  body <- resp_body_json(response)
  list(
    text = body$candidates[[1]]$content$parts[[1]]$text,
    prompt_tokens = body$usageMetadata$promptTokenCount %||% NA,
    output_tokens = body$usageMetadata$candidatesTokenCount %||% NA,
    thought_tokens = body$usageMetadata$thoughtsTokenCount %||% 0
  )
}
