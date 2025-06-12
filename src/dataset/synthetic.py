import numpy as np

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