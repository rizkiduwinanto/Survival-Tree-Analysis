import numpy as np
from joblib import Parallel, delayed
import json
import time
import multiprocessing
from tree import AFTSurvivalTree
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score

np.random.seed(42)

class AFTForest():
    """
        Survival Regression Forest for AFTLoss
    """
    def __init__(self, n_trees=10, percent_len_sample=0.8, **kwargs):
        self.percent_len_sample = percent_len_sample

        self.trees = [
            AFTSurvivalTree(**self._get_params(**kwargs)) for _ in range(n_trees)
        ]

        self.n_trees = n_trees

    # Tuning search for hyperparameters 
    def _get_params(self, **kwargs):
        args = {
            "max_depth": np.random.randint(1, 10) if "max_depth" not in kwargs else kwargs["max_depth"],
            "min_samples_split": np.random.randint(1, 10) if "min_samples_split" not in kwargs else kwargs["min_samples_split"],
            "min_samples_leaf": np.random.randint(1, 10) if "min_samples_leaf" not in kwargs else kwargs["min_samples_leaf"],
            "sigma": np.random.uniform(0, 1) if "sigma" not in kwargs else kwargs["sigma"],
            "function": np.random.choice(['norm', 'logistic', 'extreme']) if "function" not in kwargs else kwargs["function"],
            "is_custom_dist": False if "is_custom_dist" not in kwargs else kwargs["is_custom_dist"],
            "is_bootstrap": False if "is_bootstrap" not in kwargs else kwargs["is_bootstrap"],
            "n_components": 10 if "n_components" not in kwargs else kwargs["n_components"],
        }

        time_int = int(time.time())
        json.dump(args, open("params-{}.json".format(time_int), "w"))        

        return args

    def fit(self, X, y):
        self.trees = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self._fit_tree)(tree, X, y) for tree in self.trees
        )

    def _fit_tree(self, tree, X, y):
        len_sample = int(np.round(len(X) * self.percent_len_sample))
        X_sample, y_sample = self.sample(X, y, len_sample)
        tree.fit(X_sample, y_sample)
        return tree

    def fit_non_parallel(self, X, y):
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

        c_index = concordance_index(times_true, times_pred, event_true)
        return c_index

    def _brier(self, X, y_true):
        times_pred = self.predict(X)
        event_true = [1 if not censored else 0 for censored, _ in y_true]
        times_true = [time for _, time in y_true]

        ibs = integrated_brier_score(times_true, times_pred, event_true)
        return ibs

    def save(self, path):
        forest_state = {
            'n_trees': self.n_trees,
            'percent_len_sample': self.percent_len_sample
        }
        
        with open(path + "_metadata.json", 'w') as f:
            json.dump(forest_state, f, indent=4)
        
        for i, tree in enumerate(self.trees):
            tree.save(path + "_tree{}.json".format(i))

    @classmethod
    def load(cls, path):
        with open(path + "_metadata.json", 'r') as f:
            forest_state = json.load(f)
        
        forest = cls(
            n_trees=forest_state['n_trees'],
            percent_len_sample=forest_state['percent_len_sample']
        )
        
        forest.trees = []
        for i in range(forest_state['n_trees']):
            tree = AFTSurvivalTree.load(path + "_tree{}.json".format(i))
            forest.trees.append(tree)
        
        return forest