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
default_output_dir = "hyperparam"

def generate_hyperparam(model="AFTForest", n_tries=10, dry_run=False, basic=False, default_output_dir=default_output_dir):
    dataset = ['veteran'] if dry_run else ['support', 'nhanes']
    functions = ['extreme', 'normal', 'logistic'] if basic else  ['extreme', 'normal', 'logistic', 'gmm']
    
    idx = 0 
    for dataset_name in dataset:
        for function_name in functions:
            if not basic:
                if function_name == 'gmm':
                    dist_combinations = [
                        (False, True),
                        (True, True)
                    ]
                else:
                    dist_combinations = [
                        (False, False),
                        (False, True),
                        (True, True)
                    ]
                
                for bootstrap, custom_dist in dist_combinations:
                    tries_per_sub_combo = max(1, n_tries // len(dist_combinations))
                    idx = get_hyperparam(
                        tries_per_sub_combo, dataset_name, model, function_name, 
                        bootstrap, custom_dist, idx, default_output_dir
                    )
                    print(f"Generated {tries_per_sub_combo} hyperparams for {dataset_name} with {function_name}, "
                          f"bootstrap={bootstrap}, custom_dist={custom_dist}. Total: {idx}")
            else:
                idx = get_hyperparam(n_tries, dataset_name, model, function_name, 
                                    False, False, idx, default_output_dir)
                print(f"Generated {n_tries} hyperparams for {dataset_name} with {function_name}. Total: {idx}")

def get_hyperparam(n_tries, dataset, model, function, is_bootstrap, is_custom_dist, start_idx, default_output_dir=default_output_dir):
    param_grid = get_parameter(model, function, is_bootstrap, is_custom_dist)
    sampled_n_trees = [random.choice(FIXED_N_TREES) for _ in range(n_tries)]

    combinations = list(itertools.product(*param_grid.values()))
    random.shuffle(combinations)

    combination_indices = np.random.choice(len(combinations), size=n_tries, replace=False)
    combinations = [combinations[i] for i in combination_indices]

    if len(combinations) >= n_tries:
        selected_combinations = random.sample(combinations, n_tries)
    else:
        selected_combinations = [random.choice(combinations) for _ in range(n_tries)]

    for idx, hyperparams in enumerate(combinations):
        hyperparams_dict = dict(zip(param_grid.keys(), hyperparams))

        if model == 'AFTForest':
            hyperparams_dict['n_trees'] = sampled_n_trees[idx]

        params = {
            'model': model,
            'dataset': dataset,
            'function': function,
            'is_bootstrap': is_bootstrap,
            'is_custom_dist': is_custom_dist,
            'split_fitting': True if dataset == 'nhanes' else False,
            **hyperparams_dict
        }

        if is_custom_dist:
            del params['sigma']

        if not os.path.exists(default_output_dir):
            os.makedirs(default_output_dir)

        filename = f"{default_output_dir}/params_{start_idx + idx + 1}.json"
        with open(filename, "w") as f:
            json.dump(params, f, indent=2)

    return start_idx + n_tries


