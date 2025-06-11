import numpy as np
from joblib import Parallel, delayed
import json
import time
import multiprocessing
from tree import AFTSurvivalTree
from lifelines.utils import concordance_index
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import random
import os
import cupy as cp
from cupy.cuda import Stream

MAIN_FOLDER = "models/forest"
MAX_GPU = 8  # Maximum number of GPU streams to use for parallel fitting

class AFTBoosting():
    """
        AFTForestBoosting is a class that implements a forest of Accelerated Failure Time (AFT) trees.
        It allows for parallel fitting of multiple trees and supports GPU acceleration.
    """
    def __init__(
        self, 
        n_trees=10, 
        learning_rate=0.1,
        **kwargs
    ):
        self.default_params = {
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 5,
            "sigma": 0.5,
            "function": 'norm',
            "is_custom_dist": False,
            "is_bootstrap": False,
            "n_components": 10,
            "n_samples": 1000,
            "percent_len_sample": 0.8,
            "test_size": 0.2,
        }
        self.trees = []
        self.n_trees = n_trees
        self.learning_rate = learning_rate

        self.init_prediction = 0.0

    # Tuning search for hyperparameters 
    def _get_params(self, **kwargs):
        """f
            Get the parameters for the tree
        """
        args = {
            "max_depth": self.default_params['max_depth'] if "max_depth" not in kwargs else kwargs["max_depth"],
            "min_samples_split": self.default_params['min_samples_split'] if "min_samples_split" not in kwargs else kwargs["min_samples_split"],
            "min_samples_leaf": self.default_params['min_samples_split'] if "min_samples_leaf" not in kwargs else kwargs["min_samples_leaf"],
            "sigma": self.default_params['sigma'] if "sigma" not in kwargs else kwargs["sigma"],
            "function": np.random.choice(['norm', 'logistic', 'extreme']) if "function" not in kwargs else kwargs["function"],
            "is_custom_dist": False if "is_custom_dist" not in kwargs else kwargs["is_custom_dist"],
            "is_bootstrap": False if "is_bootstrap" not in kwargs else kwargs["is_bootstrap"],
            "n_components": self.default_params['n_components'] if "n_components" not in kwargs else kwargs["n_components"],
            "n_samples": self.default_params['n_samples'] if "n_samples" not in kwargs else kwargs["n_samples"],
            "percent_len_sample": self.default_params['percent_len_sample'] if "percent_len_sample" not in kwargs else kwargs["percent_len_sample"],
            "test_size": self.default_params['test_size'] if "test_size" not in kwargs else kwargs["test_size"],
            "aggregator": kwargs.get("aggregator", "mean")
        }
        return args

    def fit(self, X, y):
        predictions = np.full(len(y), self.init_prediction, dtype=np.float64)
        for _ in range(self.n_trees):
            residual = self.compute_loss(y, predictions)
            tree = AFTSurvivalTree(**self._get_params())
            tree.fit(X, {'death': [1 if not censored else 0 for censored, _ in y], 'time': [time for _, time in y]}, residual)
            self.trees.append(tree)

            predictions += self.learning_rate * tree.predict(X)

    def compute_aft_loss(y, predictions):
        """
            Compute the loss between the true values and the predictions.
            This is used to compute the residuals for the next tree.
        """

        #y_uncensored
        y_uncesored = [time for censored, time in y if not censored]
        #y_censored
        y_censored = [time for censored, time in y if censored]
        if len(y_uncesored) == 0:
            residual = np.array([0.0] * len(predictions))
        else:
            residual = np.array([np.log(time / pred) if time > pred else 0.0 for time, pred in zip(y_uncesored, predictions)])
        return residual

    def fit_gpu(self):
        """
            Fit the trees using GPU
        """
        n_streams = min(self.n_trees, MAX_GPU)
        streams = [cp.cuda.Stream() for _ in range(n_streams)]

        for stream_idx in range(n_streams):
            with streams[stream_idx]:
                for tree_idx in range(stream_idx, self.n_trees, n_streams):
                    tree = self.trees[tree_idx]
                    tree.rf_fit()

        cp.cuda.Device().synchronize()

    def predict(self, X):
        predictions = [self.get_prediction(x) for x in X]
        return predictions

    def get_prediction(self, X): 
        preds = []
        for tree in self.trees:
            pred = self.learning * tree.predict(X)
            preds.append(pred)
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

    def _brier(self, X, y):
        """
            Compute the Integrated Brier Score (IBS).
        """
        pred_times = self.predict(X)

        y_structured = np.array([(bool(not censor), float(time)) for censor, time in y], dtype=[('event', bool), ('time', float)])

        times_true = [time for _, time in y]
        min_time = min(times_true) 
        max_time = max(times_true)
        time_points = np.linspace(min_time,  max_time * 0.999, 100)

        survival_probs = np.array([[1.0 if t < pred_time else 0.0 for t in time_points] 
                              for pred_time in pred_times])

        ibs = integrated_brier_score(y_structured, y_structured, survival_probs, time_points)
        return ibs

    def _auc(self, X, y):
        """
            Compute the Area Under the Curve (AUC).
        """
        pred_times = self.predict(X)

        y_structured = np.array([(bool(not censor), float(time)) for censor, time in y], dtype=[('event', bool), ('time', float)])

        times_true = [time for _, time in y]
        min_time = min(times_true) 
        max_time = max(times_true)
        time_points = np.linspace(min_time, max_time*0.998, 100)

        survival_probs = np.array([[1.0 if t < pred_time else 0.0 for t in time_points] 
                              for pred_time in pred_times])

        auc, mean_auc = cumulative_dynamic_auc(y_structured, y_structured, survival_probs, time_points)
        return auc, mean_auc

    def _mae(self, X, y):
        """
            Compute the Mean Absolute Error (MAE).
        """
        pred_times = self.predict(X)

        event_true = [1 if not censored else 0 for censored, _ in y]
        times_true = [time for _, time in y]

        mae = mean_absolute_error(times_true, pred_times)
        return mae

    def save(self, path):
        if not os.path.exists(MAIN_FOLDER):
            os.makedirs(MAIN_FOLDER)

        new_path = os.path.join(MAIN_FOLDER, path)
        os.makedirs(new_path, exist_ok=True)

        forest_state = {
            'n_trees': self.n_trees,
            'percent_len_sample': self.percent_len_sample_forest
        }

        metadata_path = os.path.join(new_path, "_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(forest_state, f, indent=4)
        
        for i, tree in enumerate(self.trees):
            tree_path = os.path.join(new_path, "_tree{}.json".format(i))
            tree.save(tree_path)

        return new_path
            
    @classmethod
    def load(cls, path):
        """
            Load a forest from a path
        """

        metadata_path = os.path.join(path, "_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("The metadata file does not exist in the specified path.")

        with open(metadata_path, 'r') as f:
            forest_state = json.load(f)
        
        forest = cls(
            n_trees=forest_state['n_trees'],
            percent_len_sample=forest_state['percent_len_sample']
        )
        
        forest.trees = []
        for i in range(forest_state['n_trees']):
            tree_path = os.path.join(path, "_tree{}.json".format(i))
            
            tree = AFTSurvivalTree.load(tree_path)
            forest.trees.append(tree)
        
        return forest