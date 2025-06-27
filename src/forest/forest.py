import numpy as np
from joblib import Parallel, delayed
import json
import multiprocessing
from tree.tree import AFTSurvivalTree
from tqdm import tqdm
import random
import os
import cupy as cp
from cupy.cuda import Stream
from utils.metrics.metrics import c_index, brier, auc, mae
from concurrent.futures import ThreadPoolExecutor

MAIN_FOLDER = "models/forest"
MAX_GPU = 8  # Maximum number of GPU streams to use for parallel fitting

class AFTForest():
    """
        Survival Regression Forest for AFTLoss

        Parameters:
        ---------
        n_trees: int, default=10
            Number of trees in the forest.
        percent_len_sample_forest: float, default=0.37
            Percentage of the length of the sample to use for each tree in the forest.
        is_feature_subsample: bool, default=False
            Whether to subsample features for each tree.
        random_state: int, default=42
            Random state for reproducibility.
        split_fitting: bool, default=False
            Whether to use split fitting for the trees.
        **kwargs: dict
    """
    def __init__(
        self, 
        n_trees=10, 
        percent_len_sample_forest=0.37,
        is_feature_subsample=False,
        random_state=42,
        split_fitting=False,
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

        self.random_state = random_state
        self.split_fitting = split_fitting

        self.percent_len_sample_forest = percent_len_sample_forest
        self.is_feature_subsample = is_feature_subsample

        self.trees = [
            AFTSurvivalTree(**self._get_params(**kwargs)) for _ in range(n_trees)
        ]

        self.n_trees = n_trees

    # Tuning search for hyperparameters 
    def _get_params(self, **kwargs):
        """
            Get the parameters for the tree
            params: keyword arguments for the tree
            Returns: dict of parameters for the tree
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
        """
            Fit the forest to the data
            X: np.ndarray, shape (n_samples, n_features)
                The input data.
            y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
        """
        # if split fitting is enabled, we will use the GPU to fit the trees, but first we will prefit the trees on the CPU in case of custom distributions parallely
        if self.split_fitting:
            self.prefitting(X, y)
            self.fit_gpu()
        else:
            # else just fit the trees in parallel using cpu
            self.trees = list(tqdm(Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(self._fit_tree)(self.trees[idx], X, y, self.random_state + idx) for idx in range(len(self.trees))),
                total=self.n_trees
            ))

    def _fit_tree(self, tree, X, y, random_state=None):
        """
            Fit a single tree to the data
            tree: AFTSurvivalTree
                The tree to fit.
            X: np.ndarray, shape (n_samples, n_features)
                The input data.
            y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            random_state: int, optional
                Random state for reproducibility.
        """

        len_sample = int(np.round(len(X) * self.percent_len_sample_forest))
        X_sample, y_sample = self.data_sample(X, y, len_sample, random_state=random_state)

        if self.is_feature_subsample:
            len_feature_sample = int(np.round(X.shape[1] * self.percent_len_sample_forest))
            X_sample = self.feature_subsample(X_sample, len_feature_sample, random_state=random_state)

        tree.fit(X_sample, y_sample)

        return tree

    def prefitting(self, X, y):
        """
            Prefit the trees on the CPU before fitting on the GPU
            X: np.ndarray, shape (n_samples, n_features)
                The input data.
            y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
        """
        self.trees = list(tqdm(Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self._prefit)(self.trees[idx], X, y, self.random_state + idx) for idx in range(len(self.trees))),
            total=self.n_trees
        ))

    def _prefit(self, tree, X, y, random_state=None):
        """
            Prefit a single tree to the data
            tree: AFTSurvivalTree
                The tree to prefit.
            X: np.ndarray, shape (n_samples, n_features)
                The input data.
            y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            random_state: int, optional
                Random state for reproducibility.
        """
        len_sample = int(np.round(len(X) * self.percent_len_sample_forest))
        X_sample, y_sample = self.data_sample(X, y, len_sample, random_state=random_state)

        if self.is_feature_subsample:
            len_feature_sample = int(np.round(X.shape[1] * self.percent_len_sample_forest))
            X_sample = self.feature_subsample(X_sample, len_feature_sample, random_state=random_state)

        tree.prefit(X_sample, y_sample)

        return tree

    def fit_gpu(self):
        """
            Fit the trees using GPU, using multiple streams for parallel fitting
            This method uses cupy to handle GPU operations and parallelizes the fitting of trees across multiple GPU streams.
            preconditions:
            - The trees must be pre-fitted on the CPU if `split_fitting` is enabled.
        """
        n_streams = min(self.n_trees, MAX_GPU)
        streams = [cp.cuda.Stream() for _ in range(n_streams)]

        def worker(stream_idx):
            for stream_idx in range(n_streams):
                with streams[stream_idx]:
                    for tree_idx in range(stream_idx, self.n_trees, n_streams):
                        tree = self.trees[tree_idx]
                        tree.special_fit()

        with ThreadPoolExecutor(max_workers=n_streams) as executor:
            executor.map(worker, range(n_streams))

        cp.cuda.Device().synchronize()

    def data_sample(self, X, y, len_sample, random_state=None):
        """
            Subsample the data for each tree
            param X: np.ndarray, shape (n_samples, n_features)
                The input data.
            param y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            param len_sample: int
                The number of samples to sample.
            param random_state: int, optional
                Random state for reproducibility.
            returns: tuple (X_sample, y_sample)
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state)

        indices = rng.choice(len(X), size=len_sample, replace=True)
        return X[indices], y[indices]

    def feature_subsample(self, X, len_sample, random_state=None):
        """
            Subsample the features for each tree
            param X: np.ndarray, shape (n_samples, n_features)
                The input data.
            param len_sample: int
                The number of features to sample.
            param random_state: int, optional
                Random state for reproducibility.
            returns: np.ndarray, shape (n_samples, len_sample)
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state)

        select_feature = rng.choice(X.shape[1], size=len_sample, replace=False)
        X_sample = X[:, select_feature]
        return X_sample

    def predict(self, X):
        """
            Predict the time for each sample in X
            param X: np.ndarray, shape (n_samples, n_features)
            The input data.
            Returns: list of float
                The predicted times for each sample.
        """
        predictions = [self.get_prediction(x) for x in X]
        return predictions

    def get_prediction(self, X): 
        """
            Get the prediction for a single sample
            param X: np.ndarray, shape (n_features,)
                The input data for a single sample.
            Returns: float
                The predicted time for the sample.
        """
        preds = []
        for tree in self.trees:
            preds.append(tree.predict(X))
        np_preds = np.array(preds)
        np_preds = np_preds.flatten()
        agg = np.median(np_preds)
        return agg

    def _score(self, X, y):
        """
            Compute the concordance index.
            param X: np.ndarray, shape (n_samples, n_features)
                The input data.
            param y_true: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            Returns: float
                The concordance index between the predicted times and the true times.
        """

        pred_times = self.predict(X)
        return c_index(pred_times, y)

    def _brier(self, X, y):
        """
            Compute the Integrated Brier Score (IBS).
            param X: np.ndarray, shape (n_samples, n_features)
                The input data.
            param y: list of tuples (censored, time)    
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            Returns: float
                The Integrated Brier Score between the predicted times and the true times.
        """
        pred_times = self.predict(X)
        return brier(pred_times, y)

    def _auc(self, X, y):
        """
            Compute the Area Under the Curve (AUC).
            param X: np.ndarray, shape (n_samples, n_features)
                The input data.
            param y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            Returns: tuple (auc, mean_auc)
        """
        pred_times = self.predict(X)
        return auc(pred_times, y)

    def _mae(self, X, y):
        """
            Compute the Mean Absolute Error (MAE).
            param X: np.ndarray, shape (n_samples, n_features)
                The input data.
            param y: list of tuples (censored, time)
                The target data, where each tuple contains a boolean indicating if the event is censored and the time of the event.
            Returns: float
                The Mean Absolute Error between the predicted times and the true times.
        """
        pred_times = self.predict(X)
        return mae(pred_times, y)

    def save(self, path):
        """
            Save the forest to a path
            param: path: str
                The path to save the forest directory containing the metadata and tree files.
            Returns: str
                The path to the saved forest directory.
            Raises: OSError if the directory cannot be created.
        """
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except OSError as e:
                path = MAIN_FOLDER
                os.makedirs(path, exist_ok=True)

        forest_state = {
            'n_trees': self.n_trees,
            'percent_len_sample': self.percent_len_sample_forest
        }

        metadata_path = os.path.join(path, "_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(forest_state, f, indent=4)
        
        for i, tree in enumerate(self.trees):
            tree_path = os.path.join(path, "_tree{}.json".format(i))
            tree.save(tree_path)

        return path
            
    @classmethod
    def load(cls, path):
        """
            Load a forest from a path
            param: path: str
                The path to the forest directory containing the metadata and tree files.
            Returns: AFTForest instance
            Raises: FileNotFoundError if the metadata file does not exist.  
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