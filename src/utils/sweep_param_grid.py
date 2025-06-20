sweep_config = {
    "method": "bayesian",
    "metric": {"goal": "maximize", "name": "c-index"},
    "parameters": {}
}

def get_sweep_param_grid(model, function, is_custom_dist, is_bootstrap):
    if model == "AFTForest":
        sweep_config["parameters"] = {
            "max_depth": {"values": [5, 10, 15]},
            "min_samples_split": {"values": [2, 10, 50, 100]},
            "min_samples_leaf": {"values": [1, 2, 5, 10, 50, 100]},
            "sigma": {"values": [0.01, 0.05, 0.1, 0.5, 0.75, 1.0]},
            "n_trees": {"values": [50, 75, 100, 125, 150]},
            "percent_len_sample_forest": {"values": [0.37, 0.5, 0.75]},
        }
        if is_custom_dist and not is_bootstrap:
            sweep_config["parameters"].update({"test_size": {"values": [0.1, 0.2, 0.3]}})
        if is_bootstrap:
            sweep_config["parameters"].update({
                "percent_len_sample": {"values": [0.5, 0.75, 1.0]},
                "n_samples": {"values": [100, 200]}
            })
        if function == 'gmm' and is_custom_dist:
            sweep_config["parameters"].update({"n_components": {"values": [1, 2, 5, 10]}})
    elif model == "AFTSurvivalTree":
        sweep_config["parameters"] = {
            "max_depth": {"values": [5, 10, 15]},
            "min_samples_split": {"values": [2, 10, 50, 100]},
            "min_samples_leaf": {"values": [1, 2, 5, 10, 50, 100]},
            "sigma": {"values": [0.01, 0.05, 0.1, 0.5, 0.75, 1.0]},
        }
        if is_custom_dist and not is_bootstrap:
            sweep_config["parameters"].update({"test_size": {"values": [0.1, 0.2, 0.3]}})
        if is_bootstrap:
            sweep_config["parameters"].update({
                "percent_len_sample": {"values": [0.5, 0.75, 1.0]},
                "n_samples": {"values": [100, 200]}
            })
        if function == 'gmm' and is_custom_dist:
            sweep_config["parameters"].update({"n_components": {"values": [1, 2, 5, 10]}})
    elif model == "XGBoostAFT":
        sweep_config["parameters"] = {
            "max_depth": {"values": [3, 6, 10]},
            "sigma": {"values": [0.01, 0.05, 0.1]},
            "learning_rate": {"values": [0.01, 0.05, 0.1]},
            "lambda_": {"values": [0.01, 0.1, 1]},
            "alpha": {"values": [0.01, 0.1, 1]},
            "num_boost_round": {"values": [100, 200, 500]},
            "early_stopping_rounds": {"values": [10, 20]},
        }
    elif model == "RandomSurvivalForest":
        sweep_config["parameters"] = {
            "n_trees": {"values": [10, 20, 50, 100]},
            "max_depth": {"values": [5, 10, 15]},
            "min_samples_split": {"values": [2, 5]},
            "min_samples_leaf": {"values": [1, 2]},
        }

    return sweep_config