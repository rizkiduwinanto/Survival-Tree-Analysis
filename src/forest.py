import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tree import AFTSurvivalTree
from lifelines.utils import concordance_index

class AFTForest():
    """
        Survival Regression Forest for AFTLoss
    """
    def __init__(self, n_trees=10, random_params=True, percent_len_sample=0.8, is_parallel=True, **kwargs):
        self.random_params = random_params
        self.percent_len_sample = percent_len_sample
        self.is_parallel = is_parallel

        if random_params:
            self.trees = [
                AFTSurvivalTree(**self._get_params()) for _ in range(n_trees)
            ]
        else:
            self.trees = [
                AFTSurvivalTree(**kwargs) for _ in range(n_trees)
            ]

        self.n_trees = n_trees

    def _get_params(self):
        return {
            "max_depth": np.random.randint(1, 10),
            "min_samples_split": np.random.randint(1, 10),
            "min_samples_leaf": np.random.randint(1, 10),
            "sigma": np.random.uniform(0, 1),
            "function": np.random.choice(['norm', 'logistic', 'extreme']),
            "is_parallel": False
        }

    def fit(self, X, y):
        if self.is_parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(tree.fit)(X, y) for tree in self.trees)
        else:
            for tree in self.trees:
                len_sample = int(np.round(len(X) * self.percent_len_sample))
                X_sample, y_sample = self.sample(X, y, len_sample)
                tree.fit(X, y)

    def sample(self, X, y, len_sample):
        indices = np.random.choice(len(X), len_sample, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        predictions = [self.get_prediction(x) for x in X]
        return predictions

    def get_prediction(self, X): 
        preds = []
        for tree in self.trees:
            preds.append(tree.predict(X))
        np_preds = np.array(preds)
        np_preds = np_preds.flatten()
        mean = np.mean(np_preds, axis=0)
        return mean

    def _score(self, X, y_true):
        times_pred = self.predict(X)
        event_true = [1 if not censored else 0 for censored, _ in y_true]
        times_true = [time for _, time in y_true]
        print(times_true, times_pred, event_true)

        c_index = concordance_index(times_true, times_pred, event_true)
        return c_index