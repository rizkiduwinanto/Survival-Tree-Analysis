
non_custom_fitting_param_grid = {
    'n_trees': [10, 20, 50, 70, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'sigma': [0.01, 0.05, 0.1, 0.5, 0.75, 1.0],
    'n_samples': [100, 200],
    'percent_len_sample': [0.25, 0.5, 0.75],
    'percent_len_sample_forest': [0.37, 0.5, 0.75],
    'test_size': [0.1, 0.2, 0.3],
    'n_components': [1, 2, 5, 10],
}

custom_fitting_param_grid = {

}

boostrap_param_grid = {
    'is_bootstrap': [True, False],
}

xgboost_param_grid = {
    'max_depth': [3, 6, 9],
    'function': ['norm', 'logistic', 'loglog'],
    'sigma': [0.01, 0.05, 0.1],
    'learning_rate': [0.01, 0.05, 0.1],
    'lambda_': [0.01, 0.1, 1],
    'alpha': [0.01, 0.1, 1],
    'num_boost_round': [100, 200, 500],
    'early_stopping_rounds': [10, 20]
}

random_forest_param_grid = {
    'n_estimators': [10, 20, 50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2'],
    'n_jobs': [-1]
}
