from utils.param_grid import get_parameter
import itertools
import numpy as np
import random
import json
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
FIXED_N_TREES = [10, 20, 50, 70, 100] 
default_output_dir = "converted_hyperparam"

def convert_hyperparam(path_to_read, model="AFTForest", is_to_gmm=False, start_idx=0, default_output_dir=default_output_dir):
    if path_to_read is None or not os.path.exists(path_to_read):
        raise ValueError(f"Path to read hyperparameter sets does not exist: {path_to_read}")

    with open(path_to_read, 'r') as file:
        hyperparameter_sets = json.load(file)

    model = hyperparameter_sets['model']
    dataset = hyperparameter_sets['dataset']
    function = hyperparameter_sets['function']
    is_bootstrap = hyperparameter_sets.get('is_bootstrap', False)
    is_custom_dist = hyperparameter_sets.get('is_custom_dist', False)
    split_fitting = hyperparameter_sets.get('split_fitting', False)
    max_depth = hyperparameter_sets.get('max_depth', None)
    min_samples_split = hyperparameter_sets.get('min_samples_split', None)
    min_samples_leaf = hyperparameter_sets.get('min_samples_leaf', None)
    percent_len_samples = hyperparameter_sets.get('percent_len_sample_forest', None)
    n_trees = hyperparameter_sets.get('n_trees', None)

    param_grid_custom_dist = get_parameter(model, 'gmm', False, True)
    param_grid_bootstrap = get_parameter(model, 'gmm', True, True)

    combinations_custom_dist = list(itertools.product(*param_grid_custom_dist.values()))
    random.shuffle(combinations_custom_dist)

    custom_dist_dict = dict(zip(param_grid_custom_dist.keys(), combinations_custom_dist[0]))

    combinations_bootstrap = list(itertools.product(*param_grid_bootstrap.values()))
    random.shuffle(combinations_bootstrap)

    bootstrap_dict = dict(zip(param_grid_bootstrap.keys(), combinations_bootstrap[0]))

    converted_hyperparams_custom_dist = {
        'model': model,
        'dataset': dataset,
        'function': function,
        'is_bootstrap': False,
        'is_custom_dist': True,
        'split_fitting': split_fitting,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'percent_len_sample_forest': percent_len_samples,
        'n_trees': n_trees,
        'test_size': custom_dist_dict.get('test_size', 0.2),
    }

    converted_hyperparams_bootstrap = {
        'model': model,
        'dataset': dataset,
        'function': function,
        'is_bootstrap': True,
        'is_custom_dist': True,
        'split_fitting': split_fitting,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'percent_len_sample_forest': percent_len_samples,
        'n_trees': n_trees,
        'test_size': custom_dist_dict.get('test_size', 0.2),
        'percent_len_sample': custom_dist_dict.get('percent_len_sample', 0.5),
        'n_samples': custom_dist_dict.get('n_samples', 100),
    }

    converted_hyperparams_gmm = {
        'model': model,
        'dataset': dataset,
        'function': 'gmm',
        'is_bootstrap': False,
        'is_custom_dist': True,
        'split_fitting': split_fitting,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'percent_len_sample_forest': percent_len_samples,
        'n_trees': n_trees,
        'test_size': custom_dist_dict.get('test_size', 0.2),
        'n_components': custom_dist_dict.get('n_components', 1),
    }

    converted_hyperparams_gmm_bootstrap = {
        'model': model,
        'dataset': dataset,
        'function': 'gmm',
        'is_bootstrap': True,
        'is_custom_dist': True,
        'split_fitting': split_fitting,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'percent_len_sample_forest': percent_len_samples,
        'n_trees': n_trees,
        'test_size': custom_dist_dict.get('test_size', 0.2),
        'percent_len_sample': bootstrap_dict.get('percent_len_sample', 0.5),
        'n_samples': bootstrap_dict.get('n_samples', 100),
        'n_components': custom_dist_dict.get('n_components', 1),
    }

    if not os.path.exists(default_output_dir):
        os.makedirs(default_output_dir)

    filename = f"{default_output_dir}/params_{start_idx + 1}.json"
    with open(filename, "w") as f:
        json.dump(converted_hyperparams_custom_dist, f, indent=2)

    filename = f"{default_output_dir}/params_{start_idx + 2}.json"
    with open(filename, "w") as f:
        json.dump(converted_hyperparams_bootstrap, f, indent=2)

    if is_to_gmm:
        filename = f"{default_output_dir}/params_{start_idx + 3}.json"
        with open(filename, "w") as f:
            json.dump(converted_hyperparams_gmm, f, indent=2)

        filename = f"{default_output_dir}/params_{start_idx + 4}.json"
        with open(filename, "w") as f:
            json.dump(converted_hyperparams_gmm_bootstrap, f, indent=2)

    return start_idx + 4 if is_to_gmm else start_idx + 2



