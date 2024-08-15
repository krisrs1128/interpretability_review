suppressPackageStartupMessages({
  library(RcppEigen)
  library(broom)
  library(ggrepel)
  library(glmnetUtils)
  library(glue)
  library(patchwork)
  library(scico)
  library(ggdendro)
  library(tictoc)
  library(tidymodels)
})

lasso_plot <- function(coefs) {
  coef_wide <- coefs |>
    select(term, step, estimate) |>
    pivot_wider(names_from = "step", values_from = "estimate", values_fill = 0) |>
    column_to_rownames("term")
  term_order <- rownames(coef_wide)[hclust(dist(coef_wide), "single")$order]

  coefs <- coefs |>
    mutate(
      lambda = factor(step, sort(unique(step), decreasing = TRUE)),
      term = factor(term, term_order)
    )

  ggplot(coefs) +
    geom_tile(aes(factor(step), term, fill = estimate, col = estimate)) +
    scale_fill_scico(midpoint = 0, palette = "cork") +
    scale_color_scico(midpoint = 0, palette = "cork") +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(
      x = "Regularization Strength",
      y = "Feature",
      fill = expression(hat(beta)[j]),
      color = expression(hat(beta)[j])
    ) +
    theme(
      axis.text = element_blank(), 
      axis.ticks = element_blank()
    )
}

# Wrapper to run the tidymodels glmnet workflow
lasso_outputs <- function(tune_spec, wf, xy) {
  control <- control_grid(verbose = TRUE, save_workflow = TRUE)
  tic()
  lasso_grid <- tune_grid(
    wf |> add_model(tune_spec),
    vfold_cv(xy, v = 4),
    grid = grid_regular(penalty(range = c(-2.66610, -0.66610)), levels = 50),
    control = control
  )
  tdiff <- toc()

  metrics <- collect_metrics(lasso_grid) |>
    filter(.metric == "accuracy") |>
    arrange(-mean)

  final_lasso <- fit_best(lasso_grid, metric = "accuracy")
  p <- tidy(final_lasso$fit$fit$fit) |>
    lasso_plot()

  # in-sample error
  y_hat <- predict(final_lasso, xy)
  errors <- bind_cols(class = xy$class, y_hat) |>
    count(class, .pred_class)

  list(plot = p, fit = final_lasso, grid = lasso_grid, errors = errors, metrics = metrics, tdiff = tdiff)
}

#' Wrapper to run the tidymodels rpart workflow
tree_outputs <- function(tree_spec, wf, xy, metric = "accuracy") {
  control <- control_grid(verbose = TRUE, save_workflow = TRUE)
  tic()
  tree_grid <- tune_grid(
    wf |> add_model(tree_spec),
    vfold_cv(xy, v = 4),
    grid = grid_regular(cost_complexity(), levels = 10),
    control = control
  )
  tdiff <- toc()

  metrics <- collect_metrics(tree_grid) |>
    filter(.metric == metric)

  final_tree <- fit_best(tree_grid, metric = metric)
  y_hat <- predict(final_tree, xy)
  errors <- bind_cols(class = xy$class, y_hat) |>
    count(class, .pred_class)

  df <- final_tree |>
    extract_fit_engine() |>
    dendro_data()
  p <- ggplot(df$segments) +
    geom_segment(aes(x, y, xend = xend, yend = yend)) +
    geom_label(data = df$labels, aes(x, y, label = label), size = 2.5) +
    geom_label(data = df$leaf_labels, aes(x, y, label = label, fill = label), size = 3) +
    scale_fill_scico_d(palette = "lisbon") +
    scale_x_continuous(expand = c(0.1, 0.1)) +
    theme_void() +
    theme(legend.position = "none")
  list(fit = final_tree, metrics = metrics, grid = tree_grid, plot = p, errors = errors, tdiff = tdiff)
}

compare_splits <- function(xy, grid) {
  xy_split <- initial_split(xy, prop = 0.5)

  fits <- list()
  split_funs <- c(training, testing)
  for (i in seq_along(split_funs)) {
    fits[[i]] <- wf |>
      add_model(tune_spec) |>
      finalize_workflow(select_best(grid, metric = "accuracy")) |>
      fit(split_funs[[i]](xy_split))
  }

  coef_compare <- bind_rows(
    tidy(extract_fit_parsnip(fits[[1]])),
    tidy(extract_fit_parsnip(fits[[2]])),
    .id = "split"
  ) |>
    pivot_wider(names_from = split, values_from = estimate) |>
    mutate(
      term = ifelse(str_detect(term, "curvature|slope"), term, glue("original_{term}"))
    ) |>
    separate(term, c("group", "feature")) |>
    mutate(
      group = ifelse(group == "original", "original", group),
      feature = ifelse(feature == "", "Intercept", feature),
      group = factor(group, c("original", "curvature", "slope"))
    )

  p <- ggplot(coef_compare, aes(`1`, `2`, col = group)) +
    geom_vline(xintercept = 0, col = "#d3d3d3", linewidth = 1.5) +
    geom_hline(yintercept = 0, col = "#d3d3d3", linewidth = 1.5) +
    geom_point() +
    guides(color = guide_legend(override.aes = aes(label = "", size = 8))) +
    geom_text_repel(data = filter(coef_compare, pmin(abs(`1`), abs(`2`)) > 0), aes(`1`, `2`, label = feature), size = 4) +
    labs(x = "Split 1", y = "Split 2") +
    scale_color_manual("", values = c("#F2780C", "#35AAF2", "#F280CA"), drop = FALSE)

  list(plot = p, coef = coef_compare)
}
