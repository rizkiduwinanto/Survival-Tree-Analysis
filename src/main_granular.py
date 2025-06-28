import pandas as pd
from dataset.veteran import VeteranLungDataset
from dataset.support import SupportDataset
from dataset.nhanes import NHANESDataset
from utils.runner_granular import tune_model_granular
from utils.utils import dump_results_to_csv
import argparse
import json
import os

def run(args):
    """
    Run the AFT Forest/Tree experiment based on the provided arguments.
    """
    index = args[0]
    path_to_read = args[1]
    path_to_save = args[2]
    path_models = args[3]
    n_models = args[4]
    n_splits = args[5]
    path_to_image = args[6]

    # Read the hyperparameter set from the json file
    file_name = f"params_{index}.json"

    path_to_read = os.path.join(path_to_read, file_name)

    with open(path_to_read, 'r') as file:
        hyperparameter_sets = json.load(file)

    model = hyperparameter_sets['model']
    dataset = hyperparameter_sets['dataset']

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

    if model.lower() == "xgboostaft":
        X_train, X_test, y_train, y_test = data.get_train_test_xgboost()
    else:
        X_train, X_test, y_train, y_test = data.get_train_test()

    print(f"Running hyperparameter set {index} for model {model} on dataset {dataset}")
    print(f"Hyperparameters: {hyperparameter_sets}")

    #Run CV with hyperparameter set
    tune_model_granular(
        n_models=n_models,
        n_splits=n_splits,
        is_cv=True,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        path=path_to_save,
        path_models=path_models,
        path_to_image=path_to_image,
        index=index-1,
        **hyperparameter_sets
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AFT Forest/Tree Experiment granularly')
    parser.add_argument('--index', type=int, help='Index of the hyperparameter set to run')
    parser.add_argument('--path_to_read', type=str, help='Path to the CSV file containing hyperparameter sets')
    parser.add_argument('--path_to_save', type=str, help='Path to save the results')
    parser.add_argument('--path_models', type=str, help='Path to save the models')
    parser.add_argument('--n_models', type=int, default=10, help='Number of models to train')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for cross-validation')
    parser.add_argument('--path_to_image', type=str, default='images', help='Path to save the images')

    args = parser.parse_args()

    run([
        args.index,
        args.path_to_read,
        args.path_to_save,
        args.path_models,
        args.n_models,
        args.n_splits,
        args.path_to_image
    ])


