from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools
import numpy as np
import random
from forest import AFTForest
from tree import AFTSurvivalTree

random_seeds = [0, 42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]

param_grid = {
    'n_trees': [10, 50, 100],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'sigma': [0.1, 0.5, 1.0],
    'n_samples': [100, 200],
    'percent_len_sample': [0.5, 0.75],
    'percent_len_sample_forest': [0.5, 0.75],
    'test_size': [0.2, 0.3],
    'n_components': [1, 2, 5, 10],
}

def run_n_models(model, x_train, y_train, x_test, y_test, prefix=None, n_models=10, **model_params):
    c_indexes = []
    brier_scores = []
    maes = []

    if n_models > len(random_seeds):
        raise ValueError("Number of models exceeds available random seeds.")

    for i in tqdm(range(n_models)):
        if model == "AFTForest":
            one_model = AFTForest(random_state=random_seeds[i], **model_params)
        elif model == "AFTSurvivalTree":
            raise ValueError("AFTSurvivalTree does not support multiple models in this way.")

        one_model.fit(x_train, y_train)

        c_indexes.append(one_model._score(x_test, y_test))
        brier_scores.append(one_model._brier(x_test, y_test))
        maes.append(one_model._mae(x_test, y_test))

        # Save the model if needed
        prefix = prefix if prefix else "model"
        one_model.save(f"{prefix}_model_{i+1}")

    return c_indexes, brier_scores, maes

def cross_validate(model, x, y, n_splits=5, prefix=None, **model_params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seeds[0])
    c_indexes = []
    brier_scores = []
    maes = []

    index = 0

    for train_index, test_index in tqdm(kf.split(x, y)):
        x_train_fold, x_test_fold = x[train_index], x[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        if model == "AFTForest":
            one_model = AFTForest(random_state=42, **model_params)
        elif model == "AFTSurvivalTree":
            params = {
                'function': model_params.get('function', 'norm'),
                'is_bootstrap': model_params.get('is_bootstrap', False),
                'is_custom_dist': model_params.get('is_custom_dist', False),
                'n_components': model_params.get('n_components', 1),
                'max_depth': model_params.get('max_depth', None),
                'min_samples_split': model_params.get('min_samples_split', 2),
                'min_samples_leaf': model_params.get('min_samples_leaf', 1),
                'sigma': model_params.get('sigma', 0.1),
                'n_samples': model_params.get('n_samples', 100),
                'percent_len_sample': model_params.get('percent_len_sample', 0.5),
                'test_size': model_params.get('test_size', 0.2),
            }
            one_model = AFTSurvivalTree(**params)

        one_model.fit(x_train_fold, y_train_fold)

        c_indexes.append(one_model._score(x_test_fold, y_test_fold))
        brier_scores.append(one_model._brier(x_test_fold, y_test_fold))
        maes.append(one_model._mae(x_test_fold, y_test_fold))

        #save the model if needed
        prefix = prefix if prefix else "model"
        one_model.save(f"{prefix}_fold_{index+1}")

        index += 1

    return c_indexes, brier_scores, maes

def tune_model(model, x_train, y_train, x_test, y_test, custom_param_grid=None, n_tries=5, n_models=5, n_splits=5, is_grid=False, is_cv=False, **kwargs):
    results =[]

    if custom_param_grid is None:
        combinations = list(itertools.product(*param_grid.values()))
    else:
        combinations = list(itertools.product(*custom_param_grid.values()))

    if not is_grid:
        random.shuffle(combinations)
        if len(combinations) > n_tries:
            combination_indices = np.random.choice(len(combinations), size=n_tries, replace=False)
            combinations = [combinations[i] for i in combination_indices]
        else:
            raise ValueError("Not enough combinations to sample from.")

    for hyperparams in tqdm(combinations, desc="Tuning Hyperparameters"):
        hyperparam_dict = dict(zip(param_grid.keys(), hyperparams))

        params = {
            **hyperparam_dict,
            'function': kwargs.get('function', 'norm'),
            'is_bootstrap': kwargs.get('is_bootstrap', False),
            'is_custom_dist': kwargs.get('is_custom_dist', False),
        }

        if is_cv:
            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            c_indexes, briers, maes = cross_validate(model, x, y, n_splits=n_splits, **params)
        else:
            c_indexes, briers, maes = run_n_models(model, x_train, y_train, x_test, y_test, n_models=n_models, **params)
        
        results.append({
            'hyperparams': hyperparam_dict,
            'c_index': c_indexes,
            'brier_score': briers,
            'mae': maes,
            'mean_c_index': np.mean(c_indexes),
            'mean_brier_score': np.mean(briers),
            'mean_mae': np.mean(maes)
        })

    return results
    


