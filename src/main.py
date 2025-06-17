import pandas as pd
import numpy as np
from tree.tree import AFTSurvivalTree
from forest.forest import AFTForest
from wrapper.xgboost_aft.xgboost_aft import XGBoostAFTWrapper
from wrapper.random_survival_forest_scikit.random_survival_forest import RandomSurvivalForestWrapper
from dataset.veteran import VeteranLungDataset
from dataset.support import SupportDataset
from dataset.nhanes import NHANESDataset
from sklearn.model_selection import train_test_split
import time
import argparse

def run(args):
    """
    Run the AFT Forest or AFT Tree algorithm based on the provided arguments.
    This function initializes the dataset, splits it into training and testing sets,
    """

    type_algo = args[0]
    dataset = args[1]
    path_to_save = args[2]
    n_trees = args[3]
    function = args[4]
    n_components = args[5]
    max_depth = args[6]
    min_samples_split = args[7]
    min_samples_leaf = args[8]
    sigma = args[9]
    is_bootstrap = args[10]
    is_custom_dist = args[11]
    n_samples = args[12]
    percent_len_sample = args[13]
    percent_len_sample_forest = args[14]
    test_size = args[15]
    is_feature_subsample = args[16]
    percent_feature_sample = args[17]
    is_split_fitting = args[18]
    aggregator = args[19]
    lambda_ = args[20]
    alpha = args[21]
    learning_rate = args[22]
    num_boost_round = args[23]
    early_stopping_rounds = args[24]

    print("Type: ", type_algo)
    print("Dataset: ", dataset)
    print("Path: ", path_to_save)
    print("Number of trees: ", n_trees)
    print("Function: ", function)
    print("Is bootstrap: ", is_bootstrap)
    print("Is custom dist: ", is_custom_dist)
    print("Number of components: ", n_components)
    print("Max depth: ", max_depth)
    print("Min samples split: ", min_samples_split)
    print("Min samples leaf: ", min_samples_leaf)
    print("Number of samples: ", n_samples)
    print("Percent length sample: ", percent_len_sample)    
    print("Percent length sample forest: ", percent_len_sample_forest)
    print("Test size: ", test_size)
    print("Sigma: ", sigma)
    print("Is feature subsample: ", is_feature_subsample)
    print("Percent feature sample: ", percent_feature_sample)
    print("Is split fitting: ", is_split_fitting)
    print("Aggregator: ", aggregator)
    print("Lambda: ", lambda_)
    print("Alpha: ", alpha)
    print("Learning rate: ", learning_rate)
    print("Num boost round: ", num_boost_round)
    print("Early stopping rounds: ", early_stopping_rounds)

    if dataset.lower() == "veteran":
        df = pd.read_csv('data/veterans_lung_cancer.csv')
        data = VeteranLungDataset(df)
    elif dataset.lower() == "support":
        df = pd.read_csv('data/support2.csv')
        data = SupportDataset(df)
    elif dataset.lower() == "nhanes":
        data = NHANESDataset()
    else:
        raise ValueError("Dataset not found")

    if type_algo.lower() == "aftforest":
        if function.lower() == "random":
            kwargs = {
                "is_bootstrap": is_bootstrap, 
                "is_custom_dist": is_custom_dist,
                "n_components": n_components,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split, 
                "min_samples_leaf": min_samples_leaf,
                "sigma": sigma,
                "n_samples": n_samples,
                "percent_len_sample": percent_len_sample,
                "test_size": test_size,
                "aggregator": aggregator
            }
        else:
            kwargs = {
                "function": function.lower(), 
                "is_bootstrap": is_bootstrap, 
                "is_custom_dist": is_custom_dist,
                "n_components": n_components,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split, 
                "min_samples_leaf": min_samples_leaf,
                "sigma": sigma,
                "n_samples": n_samples,
                "percent_len_sample": percent_len_sample,
                "test_size": test_size,
                "aggregator": aggregator
            }

        X_train, X_test, y_train, y_test = data.get_train_test()

        aft_forest = AFTForest(
            n_trees= n_trees, 
            percent_len_sample_forest=percent_len_sample_forest, 
            is_feature_subsample=is_feature_subsample,
            random_state=42,
            split_fitting=is_split_fitting,
            **kwargs
        )

        start = time.time()
        aft_forest.fit(X_train, y_train)
        end = time.time()

        aft_forest.predict(X_test)
        score = aft_forest._score(X_test, y_test)
        print("Score: ", score)

        path = aft_forest.save(path_to_save)
        print("Path: ", path)
    elif type_algo.lower() == "afttree":
        aft_tree = AFTSurvivalTree(
            function=function.lower(), 
            is_bootstrap=is_bootstrap, 
            is_custom_dist=is_custom_dist,
            n_components=n_components,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            sigma=sigma,
            n_samples=n_samples,
            percent_len_sample=percent_len_sample,
            test_size=test_size,
            aggregator=aggregator,
        )

        X_train, X_test, y_train, y_test = data.get_train_test()

        start = time.time()
        aft_tree.fit(X_train, y_train)
        end = time.time()
        score = aft_tree._score(X_test, y_test)
        print("Score: ", score)

        aft_tree.save(path_to_save)
    elif type_algo.lower() == "xgboostaft":
        xgboost_aft = XGBoostAFTWrapper(
            max_depth=max_depth,
            function=function.lower(),
            sigma=sigma,
            learning_rate=0.05,
            lambda_=0.01,   
            alpha=0.02,
            num_boost_round=1000,
            early_stopping_rounds=10
        )
        X_train, X_test, y_train, y_test = data.get_train_test_xgboost()

        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)

        start = time.time()
        xgboost_aft.fit(X_train, y_train)
        end = time.time()
        score = xgboost_aft._score(X_test, y_test)
        print("Score: ", score)
    elif type_algo.lower() == "randomsurvivalforest":
        rsf = RandomSurvivalForestWrapper(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        X_train, X_test, y_train, y_test = data.get_train_test()

        start = time.time()
        rsf.fit(X_train, y_train)
        end = time.time()
        score = rsf._score(X_test, y_test)
        print("Score: ", score)
        score_scikit = rsf.score(X_test, y_test)
        print("Score scikit: ", score_scikit)
    else:
        raise ValueError("Algorithm not found")

    print("Time: ", end - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AFT Forest/Tree')
    parser.add_argument('--parameter', type=str, help='Type of algorithm')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--path', type=str, help='Path to save tree')
    parser.add_argument('--n_trees', type=int, default=10, help='Number of trees')
    parser.add_argument('--function', type=str, help='Function')
    parser.add_argument('--n_components', type=int, default=10, help='Number of components')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth of tree')
    parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples split')
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Min samples leaf')
    parser.add_argument('--sigma', type=float, default=0.5, help='Sigma value')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--percent_len_sample', type=float, default=0.3, help='Percent length sample')
    parser.add_argument('--percent_data_sample_forest', type=float, default=0.37, help='Percent data length sample forest')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--is_bootstrap', action=argparse.BooleanOptionalAction, help='Is bootstrap')
    parser.add_argument('--is_custom_dist', action=argparse.BooleanOptionalAction, help='Is custom distribution')
    parser.add_argument('--is_feature_subsample', action=argparse.BooleanOptionalAction, help='Is feature selection')
    parser.add_argument('--percent_feature_sample', type=float, default=0.33, help='Percent feature length sample')
    parser.add_argument('--is_split_fitting', action=argparse.BooleanOptionalAction, help='Is split fitting')
    parser.add_argument('--aggregator', type=str, default='mean', help='Aggregator function for AFTForest')
    parser.add_argument('--lambdaAFT', type=float, default=0.01, help='Lambda for XGBoost AFT')
    parser.add_argument('--alpha', type=float, default=0.02, help='Alpha for XGBoost AFT')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate for XGBoost AFT')
    parser.add_argument('--num_boost_round', type=int, default=1000, help='Number of boosting rounds for XGBoost AFT')
    parser.add_argument('--early_stopping_rounds', type=int, default=10, help='Early stopping rounds for XGBoost AFT')
    
    args = parser.parse_args()

    run([
        args.parameter, 
        args.dataset, 
        args.path, 
        args.n_trees, 
        args.function, 
        args.n_components, 
        args.max_depth, 
        args.min_samples_split, 
        args.min_samples_leaf, 
        args.sigma,
        args.is_bootstrap,
        args.is_custom_dist,
        args.n_samples,
        args.percent_len_sample,
        args.percent_data_sample_forest,
        args.test_size,
        args.is_feature_subsample,
        args.percent_feature_sample,
        args.is_split_fitting,
        args.aggregator,
        args.lambdaAFT,
        args.alpha,
        args.learning_rate,
        args.num_boost_round,
        args.early_stopping_rounds
    ])