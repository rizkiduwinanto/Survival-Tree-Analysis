from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools
import numpy as np
import random
from forest.forest import AFTForest
from tree.tree import AFTSurvivalTree
from wrapper.xgboost_aft.xgboost_aft import XGBoostAFTWrapper
from wrapper.random_survival_forest_scikit.random_survival_forest import RandomSurvivalForestWrapper
from utils.param_grid import get_parameter
import os
import wandb
from utils.runner import run_n_models, cross_validate
from utils.utils import dump_results_to_csv

MAIN_FOLDER = "models"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

random_seeds = [0, 42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]
FIXED_N_TREES = [10, 20, 50, 70, 100] 

def tune_model_granular(
    x_train, 
    y_train, 
    x_test, 
    y_test, 
    n_models=10, 
    n_splits=5, 
    is_cv=False, 
    path=None,
    path_models=None,
    path_to_image=None,
    index=0, 
    **params  
):
    model = params.get('model', 'AFTForest')
    dataset = params.get('dataset', 'veteran')
    function = params.get('function', 'extreme')
    is_bootstrap = params.get('is_bootstrap', False)
    is_custom_dist = params.get('is_custom_dist', False)

    params['index'] = index + 1

    with wandb.init(
        entity="rizkiduwinanto-university-of-groningen",
        project="random-forest-aft-granular",
        notes="thesis",
        tags=[model, function, "bootstrap" if is_bootstrap else "no_bootstrap", "custom_dist" if is_custom_dist else "no_custom_dist", dataset],
        config=params,
    ) as run:
        params = dict(params)
        params.pop('model', None)

        if is_cv:
            c_indexes, briers, maes, c_index_test, brier_test, mae_test = cross_validate(model, x_train, y_train, x_test, y_test, combinations_index=index, n_splits=n_splits, path=path_models, path_to_image=path_to_image, **params)
        else:
            c_indexes, briers, maes = run_n_models(model, x_train, y_train, x_test, y_test, n_models=n_models, path=path_models, **params)

        run.log({
            'c_index': c_indexes,
            'brier_score': briers,
            'mae': maes,
            'mean_c_index': np.mean(c_indexes),
            'mean_brier_score': np.mean(briers),
            'mean_mae': np.mean(maes),
            'c_index_test': c_index_test if is_cv else None,
            'brier_score_test': brier_test if is_cv else None,
            'mae_test': mae_test if is_cv else None,
        })

        if path:
            os.makedirs(path, exist_ok=True)
            params = dict(params)
            params.pop('dataset', None)
            params.pop('model', None)
            params.pop('index', None)
             
            results = {
                'params': params,
                'c_index': c_indexes,
                'brier_score': briers,
                'mae': maes,
                'mean_c_index': np.mean(c_indexes),
                'mean_brier_score': np.mean(briers),
                'mean_mae': np.mean(maes),
                'c_index_test': c_index_test if is_cv else None,
                'brier_score_test': brier_test if is_cv else None,
                'mae_test': mae_test if is_cv else None,
                'index': index + 1,
            }
            dump_results_to_csv([results], os.path.join(path, f"results_{index + 1}.csv"))




