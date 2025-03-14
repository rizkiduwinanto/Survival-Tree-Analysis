import pandas as pd
import numpy as np
from tree import AFTSurvivalTree
from forest import AFTForest
from dataset import SupportDataset, SyntheticDataset, VeteranLungDataset
from sklearn.model_selection import train_test_split
import time

if __name__ == "__main__":
    df = pd.read_csv('data/support2.csv')
    data = SupportDataset(df)
    X_train, X_test, y_train, y_test = data.get_train_test()
    
    kwargs = {
        "function": "gmm", 
        "is_bootstrap": True, 
        "is_custom_dist": True,
        "n_components": 5
    }
    aft_forest = AFTForest(n_trees=18, random_params=False, **kwargs)

    start = time.time()
    aft_forest.fit(X_train, y_train)

    end = time.time()

    print("Time: ", end - start, "s")

    print("Score: ", aft_forest._score(X_test, y_test))