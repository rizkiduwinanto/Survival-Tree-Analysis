from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import itertools
import numpy as np
import random
from forest.forest import AFTForest
from tree.tree import AFTSurvivalTree
from wrapper.xgboost_aft.xgboost_aft import XGBoostAFTWrapper
from wrapper.random_survival_forest_scikit.random_survival_forest import RandomSurvivalForestWrapper
from utils.param_grid import get_parameter
from utils.utils import save_model, plot_survival_trees
import os
import wandb

MAIN_FOLDER = "models"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

random_seeds = [0, 42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]
FIXED_N_TREES = [10, 20, 50, 70, 100] 
PARAM_N_TREES = [10, 10, 10, 20, 20, 20, 50, 50, 70, 100, 10, 10, 10, 20, 20, 20, 50, 50, 70, 100]

def run_n_models(model, x_train, y_train, x_test, y_test, path=None, n_models=10, **model_params):
    c_indexes = []
    brier_scores = []
    maes = []

    best_model = None
    best_c_index = 0

    if n_models > len(random_seeds):
        raise ValueError("Number of models exceeds available random seeds.")

    for i in tqdm(range(n_models)):
        if model == "AFTForest":
            one_model = AFTForest(random_state=random_seeds[i], **model_params)
        elif model == "AFTSurvivalTree":
            raise ValueError("AFTSurvivalTree does not support multiple models in this way.")
        elif model == "XGBoostAFT":
            one_model = XGBoostAFTWrapper(random_state=random_seeds[i], **model_params)
        elif model == "RandomSurvivalForest":
            one_model = RandomSurvivalForestWrapper(random_state=random_seeds[i], **model_params)

        one_model.fit(x_train, y_train)

        c_index = one_model._score(x_test, y_test)
        brier_score = one_model._brier(x_test, y_test)
        mae = one_model._mae(x_test, y_test)

        c_indexes.append(c_index)
        brier_scores.append(brier_score)
        maes.append(mae)

        print(f"Model {i+1} - C-Index: {c_index}, Brier Score: {brier_score}, MAE: {mae}")
        save_model(one_model, path, f"model_{i+1}_{model}")

        if c_index > best_c_index:
            best_model = one_model
            best_c_index = c_index
        
    if best_model is not None:
        save_model(best_model, path, f"best_model_{model}")

    return c_indexes, brier_scores, maes

def cross_validate(model, x_train, y_train, x_test, y_test, combinations_index, n_splits=5, path=None, path_to_image=None, **model_params):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_c_indexes = []
    fold_brier_scores = []
    fold_maes = []

    best_model = None
    best_c_index = 0

    c_index_test = 0
    brier_score_test = 0
    mae_test = 0
    
    index = 0

    for train_index, val_index in tqdm(skf.split(x_train, y_train['death']), desc="Cross-Validation Folds"):
        if model == "XGBoostAFT":
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        else:
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        if model == "AFTForest":
            params = {
                'function': model_params.get('function', 'normal'),
                'is_bootstrap': model_params.get('is_bootstrap', False),
                'is_custom_dist': model_params.get('is_custom_dist', False),
                'n_components': model_params.get('n_components', 1),
                'max_depth': model_params.get('max_depth', None),
                'min_samples_split': model_params.get('min_samples_split', 2),
                'min_samples_leaf': model_params.get('min_samples_leaf', 1),
                'sigma': model_params.get('sigma', 0.1),
                'n_samples': model_params.get('n_samples', 100),
                'percent_len_sample_forest': model_params.get('percent_len_sample_forest', 0.5),
                'test_size': model_params.get('test_size', 0.2),
                'aggregator': model_params.get('aggregator', 'mean'),
                'split_fitting': model_params.get('split_fitting', False),
            }
            one_model = AFTForest(random_state=42, **model_params)
        elif model == "AFTSurvivalTree":
            params = {
                'function': model_params.get('function', 'normal'),
                'is_bootstrap': model_params.get('is_bootstrap', False),
                'is_custom_dist': model_params.get('is_custom_dist', False),
                'n_components': model_params.get('n_components', 1),
                'max_depth': model_params.get('max_depth', None),
                'min_samples_split': model_params.get('min_samples_split', 2),
                'min_samples_leaf': model_params.get('min_samples_leaf', 1),
                'sigma': model_params.get('sigma', 0.1),
                'n_samples': model_params.get('n_samples', 100),
                'percent_len_sample': model_params.get('percent_len_sample', 0.5),
                'aggregator': model_params.get('aggregator', 'mean'),
                'test_size': model_params.get('test_size', 0.2),
            }
            one_model = AFTSurvivalTree(**params)
        elif model == "XGBoostAFT":
            params = {
                'max_depth': model_params.get('max_depth', 6),
                'function': model_params.get('function', 'normal'),
                'sigma': model_params.get('sigma', 0.1),
                'learning_rate': model_params.get('learning_rate', 0.05),
                'lambda_': model_params.get('lambda_', 0.01),
                'alpha': model_params.get('alpha', 0.02),
                'num_boost_round': model_params.get('num_boost_round', 1000),
                'early_stopping_rounds': model_params.get('early_stopping_rounds', 10),
            }
            one_model = XGBoostAFTWrapper(**params)
        elif model == "RandomSurvivalForest":
            params = {
                'n_trees': model_params.get('n_trees', 100),
                'max_depth': model_params.get('max_depth', None),
                'min_samples_split': model_params.get('min_samples_split', 2),
                'min_samples_leaf': model_params.get('min_samples_leaf', 1),
                'bootstrap': model_params.get('bootstrap', True),
            }
            one_model = RandomSurvivalForestWrapper(**params)

        one_model.fit(x_train_fold, y_train_fold)
        
        c_index = one_model._score(x_val_fold, y_val_fold)
        brier_score = one_model._brier(x_val_fold, y_val_fold)
        mae = one_model._mae(x_val_fold, y_val_fold)

        fold_c_indexes.append(c_index)
        fold_brier_scores.append(brier_score)
        fold_maes.append(mae)

        print(f"Fold {index+1} - C-Index: {c_index}, Brier Score: {brier_score}, MAE: {mae}")

        if c_index > best_c_index:
            best_c_index = c_index
            best_model = one_model

        #save the model if needed
        save_model(one_model, path, f"model_fold_{index+1}_combi_{combinations_index+1}")

        index += 1

    # Evaluate on the test set with the best model
    if best_model is not None:
        print("Evaluating best model on test set... - best_c_index fold:", best_c_index)
        c_index_test = best_model._score(x_test, y_test)
        brier_score_test = best_model._brier(x_test, y_test)
        mae_test = best_model._mae(x_test, y_test)

        print(f"Best Model with test results - C-Index: {c_index_test}, Brier Score: {brier_score_test}, MAE: {mae_test}")

        if path_to_image is not None:
            print("Plotting survival trees...")
            y_pred = best_model.predict(x_test)
            plot_survival_trees(y_pred, y_test, path=path_to_image, dataset=model_params.get('dataset', 'veteran'), function=model_params.get('function', 'normal'), model=model, save=True, index=combinations_index+1)

    #Save the best model if needed
    if path is not None and best_model is not None:
        save_model(best_model, path, f"best_model_combi_{combinations_index+1}")
        
    return fold_c_indexes, fold_brier_scores, fold_maes, c_index_test, brier_score_test, mae_test

def tune_model(model, dataset, x_train, y_train, x_test, y_test, n_tries=5, n_models=5, n_splits=5, is_grid=False, is_cv=False, path=None, **kwargs):
    results =[]

    function = kwargs.get('function', 'normal')
    is_bootstrap = kwargs.get('is_bootstrap', False)
    is_custom_dist = kwargs.get('is_custom_dist', False)

    param_grid = get_parameter(model, function, is_custom_dist, is_bootstrap)
    
    if model == "AFTForest":
        sampled_n_trees = [PARAM_N_TREES[i] for i in range(n_tries)]

    combinations = list(itertools.product(*param_grid.values()))

    if not is_grid:
        random.shuffle(combinations)
        if len(combinations) > n_tries:
            combination_indices = np.random.choice(len(combinations), size=n_tries, replace=False)
            combinations = [combinations[i] for i in combination_indices]
        else:
            raise ValueError("Not enough combinations to sample from.")
            
    combinations_index = 0

    with wandb.init(
        entity="rizkiduwinanto-university-of-groningen",
        project="random-forest-aft-comparison",
        notes="thesis",
        tags=[model, function, "bootstrap" if is_bootstrap else "no_bootstrap", "custom_dist" if is_custom_dist else "no_custom_dist", dataset]
    ) as run:
        for hyperparams in tqdm(combinations, desc="Tuning Hyperparameters"):
            hyperparam_dict = dict(zip(param_grid.keys(), hyperparams))

            if model == "AFTForest":
                hyperparam_dict['n_trees'] = sampled_n_trees[combinations_index]

            params = {
                **hyperparam_dict,
                'function': kwargs.get('function', 'normal'),
                'is_bootstrap': kwargs.get('is_bootstrap', False),
                'is_custom_dist': kwargs.get('is_custom_dist', False),
                'aggregator': kwargs.get('aggregator', 'mean'),
            }

            if model == "AFTForest":
                params = { 
                    **params,
                    'split_fitting': kwargs.get('is_split_fitting', False),
                }


            print("Hyperparameters:", params)

            run.config.update(params, allow_val_change=True)

        if is_cv:
            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            c_indexes, briers, maes = cross_validate(model, x, y, combinations_index=combinations_index, n_splits=n_splits, path=path, **params)
        else:
            c_indexes, briers, maes = run_n_models(model, x_train, y_train, x_test, y_test, n_models=n_models, path=path, **params)
        
        results.append({
            'hyperparams': hyperparam_dict,
            'c_index': c_indexes,
            'brier_score': briers,
            'mae': maes,
            'mean_c_index': np.mean(c_indexes),
            'mean_brier_score': np.mean(briers),
            'mean_mae': np.mean(maes)
        })

            combinations_index += 1

    return results
    


