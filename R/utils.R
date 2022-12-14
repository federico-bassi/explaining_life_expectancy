# Helper Functions


# Function to check packages already loaded into NAMESPACE
check_namespace <- function(pkgs) {

  pkgs_notloaded <- pkgs[!pkgs %in% loadedNamespaces()]
  if (length(pkgs_notloaded) == 0) {
    res <- NULL
  } else {
    res <- pkgs_notloaded
  }
  return(res)

}


# Function to install and load the specified packages
install_and_load <- function(pkgs, repos = getOption("repos")) {

  pkgs_inst <- pkgs[!pkgs %in% installed.packages()]

  if (length(pkgs_inst) == 0) {
    lapply(pkgs, library, character.only = TRUE, quietly = TRUE)
    check_res <- check_namespace(pkgs)
    if (is.null(check_res)) {
      res <- "All packages correctly installed and loaded."
    } else {
      res <- paste0(
        "Problems loading packages ",
        paste0(check_res, collapse = ", "),
        "."
      )
    }

  } else {

    inst_res <- vector("character", length(pkgs_inst))

    for (i in seq_along(pkgs_inst)) {
      inst_res_tmp <- tryCatch(
        utils::install.packages(pkgs_inst[i], dependencies = TRUE, repos = repos, quiet = TRUE),
        error = function(e) e,
        warning = function(w) w
      )
      if (!is.null(inst_res_tmp)) {
        inst_res[i] <- inst_res_tmp$message
      }
    }

    pkgs_err <- pkgs_inst[!inst_res == ""]
    if (length(pkgs_err) == 0) {
      lapply(pkgs, library, character.only = TRUE, quietly = TRUE)
      check_res <- check_namespace(pkgs)
      if (is.null(check_res)) {
        res <- "All packages correctly installed and loaded."
      } else {
        res <- paste0(
          "Problems loading packages ",
          paste0(check_res, collapse = ", "),
          "."
        )
      }
    } else {
      pkgs_noerr <- pkgs[!pkgs %in% pkgs_err]
      lapply(pkgs_noerr, library, character.only = TRUE, quietly = TRUE)
      check_res <- check_namespace(pkgs_noerr)
      if (is.null(check_res)) {
        res <- paste0(
          "Problems installing packages ",
          paste0(pkgs_err, collapse = ", "),
          "."
        )
      } else {
        res <- c(
          paste0(
            "Problems installing packages ",
            paste0(pkgs_err, collapse = ", "),
            "."
          ),
          paste0(
            "Problems loading packages ",
            paste0(check_res, collapse = ", "),
            "."
          )
        )
      }
    }

  }

  message(toupper(
    paste0(
      "\n\n\n",
      "\n==================================================================",
      "\nResults:\n ",
      res,
      "\n=================================================================="
    )
  ))
  return(invisible(res))

}

# Function to evaluate models
evaluate_model <- function(prediction_results, mode, type) {
  
  if (mode == "regression") {
    res <- prediction_results %>% 
      metrics(truth = Actual, estimate = Pred) %>% 
      dplyr::select(.metric, .estimate) %>% 
      set_names("Metric", "Estimate") %>% 
      add_column("Type" = type)
  } else {
    res_confmat <- prediction_results %>%
      conf_mat(truth = Actual, estimate = Pred)
    res_metrics <- bind_rows(
      prediction_results %>% metrics(truth = Actual, estimate = Pred),
      prediction_results %>% roc_auc(truth = Actual, estimate = Prob_Low)
    ) %>% 
      dplyr::select(.metric, .estimate) %>% 
      set_names("Metric", "Estimate") %>% 
      add_column("Type" = type)
    res_roc <- prediction_results %>%
      roc_curve(truth = Actual, Prob_Low) %>% 
      add_column("Type" = type)
    res <- list("confusion" = res_confmat$table, "metrics" = res_metrics, "roc" = res_roc)
  }
  
  return(res)
  
}


# Function to plot models
plot_model <- function(prediction_results, mode) {
  
  if (mode == "regression") {
    p <- prediction_results %>% 
      dplyr::select(-Type) %>% 
      mutate(id = 1:n()) %>% 
      ggplot(aes(x = id)) +
      geom_point(aes(y = Actual, col = "Actual")) +
      geom_point(aes(y = Pred, col = "Pred")) +
      # geom_errorbar(aes(ymin = Lower, ymax = Upper), width = .2, col = "red") +
      scale_color_manual(values = c("black", "red")) +
      labs(x = "", y = "", col = "") +
      theme_minimal()
  } else {
    p <- prediction_results$roc %>%
      ggplot(aes(x = 1 - specificity, y = sensitivity)) +
      geom_path() +
      geom_abline(lty = 3) +
      coord_equal() + 
      theme_minimal() 
  }
  
  res <- plotly::ggplotly(p)
  return(res)
  
}


# Function to fit, evaluate and plot model results on test set
calibrate_evaluate_plot <- function(
  model_fit, 
  y, 
  mode, 
  type = "testing", 
  print = TRUE
) {
  
  if (type == "testing") {
    new_data <- testing(splits)
  } else {
    new_data <- training(splits)
  }
  
  if (mode == "regression") {
    pred_res <- model_fit %>% 
      augment(new_data) %>%
      dplyr::select(all_of(y), .pred) %>% 
      set_names(c("Actual", "Pred")) %>% 
      # bind_cols(
      # 	model_fit %>% 
      # 		predict(new_data, type = "conf_int") %>%
      # 		set_names(c("Lower", "Upper")) 
      # ) %>% 
      add_column("Type" = type)
  } else {
    pred_res <- model_fit %>% 
      augment(new_data) %>%
      dplyr::select(all_of(y), contains(".pred")) %>% 
      set_names(c("Actual", "Pred", "Prob_Low", "Prob_High")) %>% 
      add_column("Type" = type)
  }
  
  pred_met <- pred_res %>% evaluate_model(mode, type )
  
  if (mode == "regression") {
    pred_plot <- pred_res %>% plot_model(mode)
  } else {
    pred_plot <- pred_met %>% plot_model(mode)
  }
  
  if (print) {
    print(pred_met)
    print(pred_plot)	
  }
  
  res <- list(
    "pred_results" = pred_res,
    "pred_metrics" = pred_met
  )
  
  return(invisible(res))
  
}


# Function to evaluate model performance on train and test
collect_results <- function(model_fit, y, mode, method) {
  
  res <- map(
    c("training", "testing"),
    ~ calibrate_evaluate_plot(model_fit, y, mode, type = ., FALSE)
  )
  
  if (mode == "regression") {
    res <- list(
      "pred_results" = map(res, "pred_results") %>% bind_rows() %>% add_column("Method" = method, .before = 1),
      "pred_metrics" = map(res, "pred_metrics") %>% bind_rows() %>% add_column("Method" = method, .before = 1)
    )
  } else {
    res <- list(
      "pred_results" = map(res, "pred_results") %>% bind_rows() %>% add_column("Method" = method, .before = 1),
      "pred_metrics" = map(res, "pred_metrics") %>% map("metrics") %>% bind_rows() %>% add_column("Method" = method, .before = 1),
      "pred_roc" = map(res, "pred_metrics") %>% map("roc") %>% bind_rows() %>% add_column("Method" = method, .before = 1)
    )
  }
  
  return(res)
  
}


# Function to evaluate stacking ensembles
calibrate_evaluate_stacks <- function(
  model_stacks,
  y, 
  mode, 
  type = "testing", 
  print = TRUE
) {
  
  if (type == "testing") {
    new_data <- testing(splits)
  } else {
    new_data <- training(splits)
  }
  
  pred_res <- predict(model_stacks, new_data, members = TRUE) %>% 
    bind_cols(
      new_data %>% dplyr::select(all_of(y)),
      .
    )  %>% 
    rename("Actual" = all_of(y)) %>% 
    rename_with(~ str_replace_all(., ".pred", "Pred"))
  
  if (mode == "regression") {
    metric <- rmse
  } else {
    pred_res <- pred_res %>% rename("Pred" = "Pred_class")
    metric <- accuracy
  }
  
  pred_met <-	colnames(pred_res) %>%
    map_dfr(metric, truth = Actual, data = pred_res) %>%
    mutate(
      Member = colnames(pred_res),
      Member = ifelse(Member == "Pred", "stack", Member)
    ) %>% 
    dplyr::slice(-1) %>% 
    dplyr::select(Member, .metric, .estimate) %>% 
    set_names("Member", "Metric", "Estimate") %>% 
    add_column("Type" = type)
  
  if (print) {
    print(pred_met)
  }
  
  res <- list(
    "pred_results" = pred_res,
    "pred_metrics" = pred_met
  )
  
  return(invisible(res))
  
}


# Mode 
stat_mode <- function(x) {
  ux <- unique(x)
  mode <- ux[which.max(tabulate(match(x, ux)))]
  return(mode)
} 


# Function to finalize and fitting models
finalizing_and_fitting <- function(workfl, param) {
  res <- finalize_workflow(workfl, param) %>% 
    last_fit(splits)
  return(res)
}


# Function to compute simple ensembles
simple_ensemble <- function(
  model_results, 
  workflows,
  y,
  mode, 
  ensemble_fun,
  print = TRUE
) {
  
  methods <- names(model_results)
  
  if (mode == "regression") {
    
    best_models <- model_results %>% map(select_best, metric = "rmse")
    wrkfls_fit_final <-	map2(workflows, best_models, finalizing_and_fitting)
    
    pred_res <- wrkfls_fit_final %>% 
      map(collect_predictions) %>% 
      map(".pred") %>% 
      bind_cols() %>% 
      rename_with(~ str_c("Pred_", .)) %>% 
      dplyr::rowwise() %>% 
      mutate(Ensemble = ensemble_fun(c_across(everything())), .before = 1) %>% 
      ungroup() 
    
    ensemble_met <- rmse_vec(
      truth = wrkfls_fit_final[[1]] %>% collect_predictions() %>% pull(y), 
      estimate = pred_res$Ensemble
    ) 
    members_met <- wrkfls_fit_final %>% 
      map(collect_metrics) %>% 
      map2(
        methods, 
        ~ mutate(.x, Member = .y) %>% 
          dplyr::filter(.metric == "rmse") %>% 
          dplyr::select(Member, .metric, .estimate) %>% 
          set_names(c("Member", "Metric", "Estimate"))
      ) %>%
      bind_rows()
    pred_met <- bind_rows(
      tibble("Member" = "ensemble", "Metric" = "rmse", "Estimate" = ensemble_met),
      members_met
    )
    
  } else {
    
    best_models <- model_results %>% map(select_best, metric = "accuracy")
    wrkfls_fit_final <-	map2(workflows, best_models, finalizing_and_fitting)
    
    pred_res <- wrkfls_fit_final %>% 
      map(collect_predictions) %>% 
      map(".pred_class") %>% 
      bind_cols() %>% 
      rename_with(~ str_c("Pred_", .)) %>% 
      dplyr::rowwise() %>% 
      mutate(Ensemble = ensemble_fun(c_across(everything())), .before = 1) %>% 
      ungroup() 
    
    ensemble_met <- accuracy_vec(
      truth = wrkfls_fit_final[[1]] %>% collect_predictions() %>% pull(y), 
      estimate = pred_res$Ensemble
    ) 
    members_met <- wrkfls_fit_final %>% 
      map(collect_metrics) %>% 
      map2(
        methods, 
        ~ mutate(.x, Member = .y) %>% 
          dplyr::filter(.metric == "accuracy") %>% 
          dplyr::select(Member, .metric, .estimate) %>% 
          set_names(c("Member", "Metric", "Estimate"))
      ) %>%
      bind_rows()
    pred_met <- bind_rows(
      tibble("Member" = "ensemble", "Metric" = "rmse", "Estimate" = ensemble_met),
      members_met
    )
    
  }
  
  if (print) {
    print(pred_met)
  }
  
  res <- list(
    "pred_results" = pred_res,
    "pred_metrics" = pred_met
  )
  
  return(invisible(res))
}