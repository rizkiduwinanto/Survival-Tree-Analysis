import numpy as np

from sklearn.model_selection import train_test_split
from .dataset import Dataset

class SyntheticDataset(Dataset):
    """
    Synthetic dataset for survival analysis.
    This dataset is generated with a specified number of censored and uncensored samples.
    It includes features and survival outcomes, which can be used for testing and benchmarking survival analysis algorithms.
    """
    def __init__(self, n_censored, n_uncensored, n_feature=2):
        self.n_feature = n_feature
        self.n_samples = n_censored + n_uncensored
        self.n_censored = n_censored
        self.n_uncensored = n_uncensored

        self.X, self.y = self.create_data() 

        self.label = self.create_label()
        self.xgboost_label = self.create_xgboost_label()
    
    def create_data(self):
        X = np.random.rand(self.n_samples, self.n_feature)
        y_censored = [(False, np.random.randint(1, 100)) for _ in range(self.n_censored)]
        y_uncensored = [(True, np.random.randint(1, 100)) for _ in range(self.n_uncensored)]

        y = np.concatenate([y_censored, y_uncensored])
        return X, y

    def create_one_data(self):
        X = np.random.rand(1, self.n_feature)
        y = [(True, np.random.randint(1, 100))]
        return X, y
        
    def create_label(self):
        label = np.array([1 if y[0] else 0 for y in self.y])
        d_time = np.array([y[1] for y in self.y])
        structured_arr = np.stack((label, d_time), axis=1)
        return structured_arr

    def create_xgboost_label(self):
        lower_bound = np.array([y[1] for y in self.y])
        upper_bound = np.array([y[1] if y[0] else np.inf for y in self.y])
        return np.column_stack((lower_bound, upper_bound))

    def preprocess(self, *args, **kwargs):
        # No preprocessing needed for synthetic data
        pass

    def get_train_test(self, test_size=0.2, random_state=42):
        pass

    def get_train_test_xgboost(self, test_size=0.2, random_state=42):
        pass

    def get_data(self):
        return self.X

    def get_label(self):
        return self.label

    def get_xgboost_label(self):
        return self.xgboost_label
