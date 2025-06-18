tree_param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 10, 50, 100],
    'min_samples_leaf': [1, 2, 5, 10, 50, 100],
    'sigma': [0.01, 0.05, 0.1, 0.5, 0.75, 1.0],
}

forest_param_grid = {
    **tree_param_grid,
    'n_trees': [50, 100, 150],
    'percent_len_sample_forest': [0.37, 0.5, 0.75],
}

custom_fitting_param_grid = {
    'test_size': [0.1, 0.2, 0.3],
}

boostrap_param_grid = {
    'percent_len_sample': [0.5, 0.75, 1.0],
    'n_samples': [100, 200]
}

gmm_param_grid = {
    'n_components': [1, 2, 5, 10],
}

xgboost_param_grid = {
    'max_depth': [3, 6, 10],
    'sigma': [0.01, 0.05, 0.1],
    'learning_rate': [0.01, 0.05, 0.1],
    'lambda_': [0.01, 0.1, 1],
    'alpha': [0.01, 0.1, 1],
    'num_boost_round': [100, 200, 500],
    'early_stopping_rounds': [10, 20]
}

scikit_param_grid = {
    'n_trees': [10, 20, 50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}


def get_parameter(model, function, is_custom_dist, is_bootstrap):
    if model == "AFTForest":
        param_grid = forest_param_grid
        if is_custom_dist and not is_bootstrap:
            param_grid = {**param_grid, **custom_fitting_param_grid}
        if is_bootstrap:
            param_grid = {**param_grid, **boostrap_param_grid}
        if function == 'gmm' and is_custom_dist:
            param_grid = {**param_grid, **gmm_param_grid}
    elif model == "AFTSurvivalTree":
        param_grid = tree_param_grid
        if is_custom_dist and not is_bootstrap:
            param_grid = {**param_grid, ** custom_fitting_param_grid}
        if is_bootstrap:
            param_grid = {**param_grid, **boostrap_param_grid}
        if function == 'gmm' and is_custom_dist:
            param_grid = {**param_grid, **gmm_param_grid}
    elif model == "XGBoostAFT":
        param_grid = xgboost_param_grid
    elif model == "RandomSurvivalForest":
        param_grid = scikit_param_grid