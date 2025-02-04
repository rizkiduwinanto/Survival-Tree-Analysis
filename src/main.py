import pandas as pd
import numpy as np
from tree import AFTSurvivalTree
from dataset import Dataset, SyntheticDataset
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # df = pd.read_csv('data/support2.csv')
    # data = Dataset(df)
    # trunc_data = data.get_data()
    # data_label = data.get_label()
    # X_train, X_test, y_train, y_test = data.get_train_test()

    synthetic_data = SyntheticDataset(n_feature=3, n_censored=5, n_uncensored=10)
    X, y = synthetic_data.create_data()

    X_predict, y_predict = synthetic_data.create_one_data()
    
    aft_surv_tree = AFTSurvivalTree()
    aft_surv_tree.fit(X, y)
    aft_surv_tree.print()
    # print("Class: ", aft_surv_tree.predict(X_predict))

    # print(aft_surv_tree.tree)