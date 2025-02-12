import pandas as pd
import numpy as np
from tree import AFTSurvivalTree
from forest import AFTForest
from dataset import SupportDataset, SyntheticDataset
from sklearn.model_selection import train_test_split
import time

if __name__ == "__main__":
    df = pd.read_csv('data/support2.csv')
    data = SupportDataset(df)
    X_train, X_test, y_train, y_test = data.get_train_test()

    # synthetic_data = SyntheticDataset(n_feature=3, n_censored=100, n_uncensored=200)
    # X_train, y_train = synthetic_data.create_data()

    # X_predict, y_predict = synthetic_data.create_one_data()

    # synthetic_data_test = SyntheticDataset(n_feature=3, n_censored=10, n_uncensored=20)
    # X_test, y_test = synthetic_data_test.create_data()
    
    # aft_surv_tree = AFTSurvivalTree(function="norm", is_parallel=False)

    aft_forest = AFTForest(n_trees=3, random_params=True)
    
    start = time.time()
    # aft_surv_tree.fit(X_train, y_train)
    aft_forest.fit(X_train, y_train)

    end = time.time()

    print("Time: ", end - start, "s")

    print("Class: ", aft_forest.predict(X_test[:1]))
    print("Score: ", aft_forest._score(X_test, y_test))

    # print("Class: ", aft_surv_tree.predict(X_test[:1]))
    # aft_surv_tree._visualize()
    # print("Score: ", aft_surv_tree._score(X_test, y_test))

    # print(aft_surv_tree.tree)