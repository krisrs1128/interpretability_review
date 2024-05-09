error_rate <- function(spec, xy) {
  y_hat <- fit(spec, class ~ ., data = xy) |>
    predict(xy)
  errors <- bind_cols(class = xy$class, y_hat) |>
    count(class, .pred_class)
  list(errors = errors, fit = fit)
}

#' Wrapper to run the tidymodels glmnet workflow
lasso_outputs <- function(tune_spec, wf, xy) {
  control <- control_grid(verbose = TRUE, save_workflow = TRUE)
  lasso_grid <- tune_grid(
    wf |> add_model(tune_spec),
    vfold_cv(xy, v = 4),
    grid = grid_regular(penalty(range=c(-2.66610, -0.66610)), levels = 50),
    control = control
  )

  p1 <- collect_metrics(lasso_grid) |>
    filter(.metric == "accuracy") |>
    ggplot(aes(log(penalty))) +
    geom_errorbar(aes(
      ymin = mean - std_err,
      ymax = mean + std_err
    ),
    alpha = 0.5
    ) +
    geom_point(aes(y = mean), col = "#545454") +
    scale_x_reverse() +
    labs(x = expression(log(lambda)), y = "CV Accuracy") +
    facet_wrap(~ .metric)

  final_lasso <- fit_best(lasso_grid, metric = "accuracy")
  coefs <- tidy(final_lasso$fit$fit$fit)
  p2 <- ggplot(coefs) +
    geom_hline(yintercept = 0, linewidth = 1.5) +
    geom_line(aes(log(lambda), estimate, group = term), col = "#0c0c0c") +
    labs(x = expression(log(lambda)), y = "Coefficient Estimate") +
    scale_x_reverse()

  list(plot = p1 / p2, fit = final_lasso, grid = lasso_grid)
}

#' Wrapper to run the tidymodels rpart workflow
tree_outputs <- function(tree_spec, wf, xy, metric = "accuracy") {
  control <- control_grid(verbose = TRUE, save_workflow = TRUE)
  tree_grid <- tune_grid(
    wf |> add_model(tree_spec),
    vfold_cv(xy, v = 4),
    grid = grid_regular(cost_complexity(), levels = 10),
    control = control
  )

  metrics_plot <- collect_metrics(tree_grid) |>
    filter(.metric == metric) |>
    ggplot(aes(log(cost_complexity))) +
    geom_errorbar(aes(
      ymin = mean - std_err,
      ymax = mean + std_err
    ),
    alpha = 0.5
    ) +
    geom_point(aes(y = mean), col = "#545454") +
    scale_x_reverse() +
    labs(x = "Cost Complexity", y = "CV Accuracy") +
    facet_wrap(~ .metric) +
    theme(
      strip.text = element_text(size = 16),
      axis.title = element_text(size = 18),
      axis.text = element_text(size = 16)
    )

  final_tree <- fit_best(tree_grid, metric = metric)
  extract_fit_engine(final_tree) |>
    rpart.plot()
  list(fit = final_tree, grid = tree_grid, plot = metrics_plot)
}