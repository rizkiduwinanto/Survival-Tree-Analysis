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

MAIN_FOLDER = "models"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

random_seeds = [0, 42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324]

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

        # Save the model if needed
        if model == "AFTForest":
            path_dir = os.path.join(path, f"model_{i+1}")
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            one_model.save(path_dir)
        elif model == "AFTSurvivalTree":
            path_dir = os.path.join(path, f"model_{i+1}.json")
            one_model.save(path_dir)

        # Check if this model is the best one
        if c_index > best_c_index:
            best_model = one_model
            best_c_index = c_index
        
    # Save the best model if needed
    if best_model is not None:
        if model == "AFTForest":
            best_path_dir = os.path.join(path, "best_model")
            if not os.path.exists(best_path_dir):
                os.makedirs(best_path_dir)
            best_model.save(best_path_dir)
        elif model == "AFTSurvivalTree":
            best_path_dir = os.path.join(path, "best_model.json")
            best_model.save(best_path_dir)

    return c_indexes, brier_scores, maes

def cross_validate(model, x_train, y_train, x_test, y_test, combinations_index, n_splits=5, path=None, **model_params):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seeds[0])
    fold_c_indexes = []
    fold_brier_scores = []
    fold_maes = []

    best_model = None
    best_c_index = 0

    c_index_test = 0
    brier_score_test = 0
    mae_test = 0
    
    index = 0

    for train_index, val_index in tqdm(kf.split(x_train, y_train)):
        if model == "XGBoostAFT":
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        else:
            x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        if model == "AFTForest":
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
                'percent_len_sample_forest': model_params.get('percent_len_sample_forest', 0.5),
                'test_size': model_params.get('test_size', 0.2),
                'aggregator': model_params.get('aggregator', 'mean'),
                'split_fitting': model_params.get('split_fitting', False),
            }
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
                'aggregator': model_params.get('aggregator', 'mean'),
                'test_size': model_params.get('test_size', 0.2),
            }
            one_model = AFTSurvivalTree(**params)
        elif model == "XGBoostAFT":
            params = {
                'max_depth': model_params.get('max_depth', 6),
                'function': model_params.get('function', 'norm'),
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

        #print all averages 
        print(f"Fold {index+1} - C-Index: {c_index}, Brier Score: {brier_score}, MAE: {mae}")

        # Check if this model is the best one
        if c_index > best_c_index:
            best_c_index = c_index
            best_model = one_model

        #save the model if needed
        if model == "AFTForest":
            path_dir = os.path.join(path, f"comb_{combinations_index+1}_fold_{index+1}")
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            one_model.save(path_dir)
        elif model == "AFTSurvivalTree":
            path_dir = os.path.join(path, f"comb_{combinations_index+1}_fold_{index+1}.json")
            one_model.save(path_dir)

        index += 1

    # Evaluate on the test set with the best model
    if best_model is not None:
        print("Evaluating best model on test set... - best_c_index fold:", best_c_index)
        c_index_test = best_model._score(x_test, y_test)
        brier_score_test = best_model._brier(x_test, y_test)
        mae_test = best_model._mae(x_test, y_test)

        print(f"Best Model with test results - C-Index: {c_index_test}, Brier Score: {brier_score_test}, MAE: {mae_test}")

    #Save the best model if needed
    if path is not None and best_model is not None:
        if model == "AFTForest":
            best_path_dir = os.path.join(path, f"best_model_combi_{combinations_index+1}")
            if not os.path.exists(best_path_dir):
                os.makedirs(best_path_dir)
            best_model.save(best_path_dir)
        elif model == "AFTSurvivalTree":
            best_path_dir = os.path.join(path, f"best_model_combi_{combinations_index+1}.json")
            best_model.save(best_path_dir)
        
    return fold_c_indexes, fold_brier_scores, fold_maes, c_index_test, brier_score_test, mae_test

def tune_model(model, dataset, x_train, y_train, x_test, y_test, n_tries=5, n_models=5, n_splits=5, is_grid=False, is_cv=False, path=None, **kwargs):
    results =[]

    function = kwargs.get('function', 'normal')
    is_bootstrap = kwargs.get('is_bootstrap', False)
    is_custom_dist = kwargs.get('is_custom_dist', False)

    param_grid = get_parameter(model, function, is_custom_dist, is_bootstrap)

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
        project="random-forest-aft",
        notes="thesis",
        tags=[model, function, "bootstrap" if is_bootstrap else "no_bootstrap", "custom_dist" if is_custom_dist else "no_custom_dist", dataset]
    ) as run:
        for hyperparams in tqdm(combinations, desc="Tuning Hyperparameters"):
            hyperparam_dict = dict(zip(param_grid.keys(), hyperparams))

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
                c_indexes, briers, maes, c_index_test, brier_test, mae_test = cross_validate(model, x_train, y_train, x_test, y_test, combinations_index=combinations_index, n_splits=n_splits, path=path, **params)
            else:
                c_indexes, briers, maes = run_n_models(model, x_train, y_train, x_test, y_test, n_models=n_models, path=path, **params)

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
            
            results.append({
                'hyperparams': hyperparam_dict,
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

            combinations_index += 1

    return results
    


