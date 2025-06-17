import pandas as pd
from dataset.veteran import VeteranLungDataset
from dataset.support import SupportDataset
from dataset.nhanes import NHANESDataset
from utils.runner import tune_model
from utils.utils import dump_results_to_csv
import argparse

def run(args):
    """
    Run the AFT Forest/Tree experiment based on the provided arguments.
    """

    type_algo = args[0]
    dataset = args[1]
    path_to_save = args[2] 
    path_res = args[3]
    n_tries = args[4]
    n_models = args[5]
    n_splits = args[6]
    is_grid = args[7]
    is_cv = args[8]
    function = args[9]
    is_bootstrap = args[10]
    is_custom_dist = args[11]
    aggregator = args[12]

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

    if type_algo.lower() == "xgboostaft":
        X_train, X_test, y_train, y_test = data.get_train_test_xgboost()
    else:
        X_train, X_test, y_train, y_test = data.get_train_test()
        
    fixed_params = {
        'function': function,
        'is_bootstrap': False if not is_bootstrap else is_bootstrap,
        'is_custom_dist': False if not is_custom_dist else is_custom_dist,
        'aggregator': aggregator,
    }

    if type_algo.lower() == "aftforest": 
        model = "AFTForest"
    elif type_algo.lower() == "afttree":
        model = "AFTSurvivalTree"
    elif type_algo.lower() == "xgboostaft":
        model = "XGBoostAFT"
    elif type_algo.lower() == "randomsurvivalforest":
        model = "RandomSurvivalForest"
    else:
        raise ValueError("Algorithm not found")

    res = tune_model(
        model=model,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        custom_param_grid=None,
        n_tries=n_tries,
        n_models=n_models,
        n_splits=n_splits,
        is_grid=is_grid,
        is_cv=is_cv,
        path=path_to_save,
        **fixed_params
    )

    print("Results: ", res)
    dump_results_to_csv(res, path_res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AFT Forest/Tree Experiment')
    parser.add_argument('--parameter', type=str, help='Type of algorithm')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--path', type=str, help='Path to save tree')  
    parser.add_argument('--path-results', type=str, help='Path to save results')
    parser.add_argument('--n_tries', type=int, default=5, help='Number of tries')
    parser.add_argument('--n_models', type=int, default=5, help='Number of models')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for cross-validation')
    parser.add_argument('--is_grid', action=argparse.BooleanOptionalAction, help='Is grid search')
    parser.add_argument('--is_cv', action=argparse.BooleanOptionalAction, help='Is cross-validation')
    parser.add_argument('--function', type=str, default='lognormal', help='Distribution function to use')
    parser.add_argument('--is_bootstrap', action=argparse.BooleanOptionalAction, help='Is bootstrap')
    parser.add_argument('--is_custom_dist', action=argparse.BooleanOptionalAction, help='Is custom distribution')
    parser.add_argument('--aggregator', type=str, default='mean', help='Aggregator function for AFTForest')
    parser.add_argument('--is_split_fitting', action=argparse.BooleanOptionalAction, help='Is split fitting')

    args = parser.parse_args()

    run([
        args.parameter,
        args.dataset,
        args.path,
        args.path_results,
        args.n_tries,
        args.n_models,  
        args.n_splits,
        args.is_grid,
        args.is_cv,
        args.function,
        args.is_bootstrap,
        args.is_custom_dist,
        args.aggregator
    ])


