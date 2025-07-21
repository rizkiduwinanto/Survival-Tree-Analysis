from utils.param_grid import get_all_parameter
import itertools
import numpy as np
import random
import json
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
FIXED_N_TREES = [10, 50, 100, 150] 
default_output_dir = "hyperparam"

def generate_hyperparam(model="AFTForest", n_tries=10, dry_run=False, basic=False, default_output_dir=default_output_dir):
    dataset = ['veteran'] if dry_run else ['support', 'nhanes']
    functions = ['extreme', 'normal', 'logistic'] if basic else  ['extreme', 'normal', 'logistic', 'gmm']

    param_grid = get_all_parameter(model)
    sampled_n_trees = [random.choice(FIXED_N_TREES) for _ in range(n_tries)]

    combinations = list(itertools.product(*param_grid.values()))
    selected_combinations = random.sample(combinations, n_tries)
    
    idx = 0 
    for dataset_name in dataset:
        for function_name in functions:
            if basic:
                dist_combinations = [(False, False)]
            else:
                if function_name == 'gmm':
                    dist_combinations = [(False, True), (True, True)]
                else:
                    dist_combinations = [(False, False), (False, True), (True, True)]

            for is_bootstrap, is_custom_dist in dist_combinations:
                for i, combo in enumerate(selected_combinations):
                    hyperparams = dict(zip(param_grid.keys(), combo))

                    if model == 'AFTForest':
                        hyperparams['n_trees'] = sampled_n_trees[i]

                    params = {
                        'model': model,
                        'dataset': dataset_name,
                        'function': function_name,
                        'is_bootstrap': is_bootstrap,
                        'is_custom_dist': is_custom_dist,
                        'split_fitting': False,
                        **hyperparams
                    }
                    
                    if not os.path.exists(default_output_dir):
                        os.makedirs(default_output_dir) 
                    
                    filename = f"{default_output_dir}/params_{idx + 1}.json"
                    with open(filename, "w") as f:
                        json.dump(params, f, indent=2)

                    idx += 1


