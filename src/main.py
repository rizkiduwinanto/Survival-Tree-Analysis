import pandas as pd
import numpy as np
from tree import AFTSurvivalTree
from forest import AFTForest
from dataset import SupportDataset, SyntheticDataset, VeteranLungDataset
from sklearn.model_selection import train_test_split
import time
import argparse

def run(args):
    type_algo = args[0]
    dataset = args[1]
    path_to_save = args[2]
    n_trees = args[3]
    function = args[4]
    is_bootstrap = args[5]
    is_custom_dist = args[6]

    print("Type: ", type_algo)
    print("Dataset: ", dataset)
    print("Path: ", path_to_save)
    print("Number of trees: ", n_trees)
    print("Function: ", function)
    print("Is bootstrap: ", is_bootstrap)
    print("Is custom dist: ", is_custom_dist)

    if dataset.lower() == "veteran":
        df = pd.read_csv('data/veterans_lung_cancer.csv')
        data = VeteranLungDataset(df)
    elif dataset.lower() == "support":
        df = pd.read_csv('data/support2.csv')
        data = SupportDataset(df)
    else:
        raise ValueError("Dataset not found")
    
    X_train, X_test, y_train, y_test = data.get_train_test()

    if type_algo.lower() == "aftforest":
        if function.lower() == "random":
            kwargs = {
                "is_bootstrap": is_bootstrap, 
                "is_custom_dist": is_custom_dist,
                "n_components": 2
            }
        else:
            kwargs = {
                "function": function.lower(), 
                "is_bootstrap": is_bootstrap, 
                "is_custom_dist": is_custom_dist,
                "n_components": 2
            }

        aft_forest = AFTForest(n_trees= n_trees, random_params=False, **kwargs)

        start = time.time()
        aft_forest.fit(X_train, y_train)
        end = time.time()

        aft_forest.predict(X_test)
        score = aft_forest._score(X_test, y_test)
        print("Score: ", score)

        path = aft_forest.save(path_to_save)
        print("Path: ", path)
    elif type_algo.lower() == "aftsurvivaltree":
        aft_tree = AFTSurvivalTree(
            function=function.lower(), 
            is_bootstrap=is_bootstrap, 
            is_custom_dist=is_custom_dist,
            n_components=2
        )
        start = time.time()
        aft_tree.fit(X_train, y_train)
        end = time.time()
        score = aft_tree._score(X_test, y_test)
        print("Score: ", score)

        aft_tree.save(path_to_save)

    print("Time: ", end - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AFT Forest/Tree')
    parser.add_argument('--parameter', type=str, help='Type of algorithm')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--path', type=str, help='Path to save tree')
    parser.add_argument('--n_trees', type=int, help='Number of trees')
    parser.add_argument('--function', type=str, help='Function')
    parser.add_argument('--is_bootstrap', type=bool, help='Is bootstrap')
    parser.add_argument('--is_custom_dist', type=bool, help='Is custom distribution')

    args = parser.parse_args()

    run([args.parameter, args.dataset, args.path, args.n_trees, args.function, args.is_bootstrap, args.is_custom_dist])