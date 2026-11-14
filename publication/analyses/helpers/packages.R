# Every function used from a package outside base R is imported here by name with `box`, so this
# file is also the list of software the analyses depend on. Add new imports here, not in the scripts.
# Exceptions: `here::here()` in the first line of every script (it finds `shared.R`), the packages
# that come with R (stats, utils, tools, grDevices), which are called as `pkg::fn()`, and
# `ggplot2::.pt`. Names that clash with base/stats are aliased (fast_hclust).
suppressPackageStartupMessages({
  box::use(
    # --- dplyr / tidyr / tibble / purrr / forcats / stringr / readr ---
    dplyr[mutate, filter, select, arrange, group_by, ungroup, summarize, summarise, across, pull,
          count, n, n_distinct, distinct, desc, row_number, rename, rename_with, transmute,
          bind_rows, bind_cols, left_join, inner_join, anti_join, if_else, case_when, coalesce,
          lead, slice, slice_head, slice_min, slice_sample, group_modify, vars, first, tbl,
          collect, slice_tail],
    tidyselect[everything, all_of, any_of, starts_with, matches, where],
    tidyr[pivot_longer, pivot_wider, unnest, nest, expand_grid, complete, fill,
          separate_wider_delim],
    tibble[tibble, tribble, as_tibble],
    purrr[map, map2, map_dbl, map_chr, map_lgl, map_int, map2_dbl, map2_chr, pmap, pmap_chr,
          imap, walk2, set_names, reduce, keep, list_rbind, list_flatten, map2_lgl],
    forcats[fct_relabel],
    stringr[str_c, str_detect, str_remove, str_remove_all, str_extract, str_replace,
            str_replace_all, str_to_sentence, str_to_lower, str_trim, str_squish, str_sub,
            str_length, str_glue],
    readr[cols, col_double, col_logical, col_character, read_csv, write_csv],

    # --- ggplot2 and extensions ---
    ggplot2[ggplot, aes, after_stat, after_scale, labs, ggtitle, xlim, ylim, theme, theme_set,
            theme_void, element_text, element_blank, element_line, element_rect, margin, rel,
            expansion, unit, arrow, annotate, facet_wrap, facet_grid, guides, guide_legend,
            coord_cartesian, coord_fixed, geom_point, geom_line, geom_col, geom_text, geom_tile,
            geom_ribbon, geom_hline, geom_vline, geom_segment, geom_path, stat_summary,
            stat_ecdf, scale_x_continuous, scale_y_continuous, scale_y_discrete,
            scale_fill_manual, scale_color_manual, scale_colour_manual, scale_fill_gradient2,
            geom_raster, scale_fill_identity, scale_alpha_identity, scale_size_identity,
            scale_shape_manual, ggsave, scale_linewidth_manual, scale_size_manual],
    ggpubr[theme_pubr],
    patchwork[wrap_plots, plot_spacer],
    ggforce[geom_circle],
    ggimage[geom_image],
    scales[label_percent, label_number, oob_squish, breaks_width, oob_keep],
    latex2exp[TeX],
    extrafont[fonts, loadfonts],
    kableExtra[kable, pack_rows, add_header_above],

    # --- embeddings: dimension reduction, clustering ---
    fastkmedoids[fastpam],
    fastcluster[fast_hclust = hclust],
    dbscan[hdbscan],
    aricode[AMI],

    # --- data / IO ---
    jsonlite[fromJSON, toJSON, base64_enc],
    httr2[request, req_headers, req_body_json, req_retry, req_error, req_perform_parallel,
          resp_body_json],
    cli[cli_progress_bar, cli_progress_update, cli_progress_done],
    magick[image_read, image_orient, image_background, image_rotate, image_info, image_crop,
           image_border, image_resize, image_colorize, image_write, image_blank, image_composite,
           geometry_area, geometry_point],
    digest[digest],
    dbplyr[sql],
    DBI[dbConnect, dbExecute, dbQuoteString],
    duckdb[duckdb],
    systemfonts[font_info, register_font]
  )
})

suppressMessages(suppressWarnings(loadfonts(quiet = TRUE)))
