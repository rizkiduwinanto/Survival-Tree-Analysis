import numpy as np
import cupy as cp
from .node import TreeNode
from distribution.weibull import Weibull
from distribution.norm import LogNormal
from distribution.logistic import LogLogistic
from distribution.extreme import LogExtremeNew
from distribution.GMM import GMM_New
from utils.math.math_utils_gpu import norm_pdf, norm_cdf, logistic_pdf, logistic_cdf, extreme_pdf, extreme_cdf
from utils.utils import stratified_gpu_train_test_split
import graphviz
import uuid
import json
from sklearn.model_selection import train_test_split
from utils.metrics.metrics import c_index, brier, auc, mae

from collections import deque
from sklearn.model_selection import train_test_split

class AFTSurvivalTree():
    """
        Regression tree that implements AFTLoss

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree. Default is 5.
        min_samples_split : int, optional
            Minimum number of samples required to split an internal node. Default is 5.
        min_samples_leaf : int, optional
            Minimum number of samples required to be at a leaf node. Default is 5.
        sigma : float, optional
            Standard deviation for the loss function. Default is 0.5.
        function : str, optional
            Distribution function to use for the loss calculation. Options are "normal", "logistic", "extreme", "weibull", "gmm", "gmm_new". Default is "normal".
        is_custom_dist : bool, optional
            Whether to use a custom distribution for the loss calculation. Default is False.
        is_bootstrap : bool, optional
            Whether to use bootstrap sampling for fitting the custom distribution. Default is False.
        n_components : int, optional
            Number of components for Gaussian Mixture Model (GMM) if using GMM distribution. Default is 10.
        n_samples : int, optional
            Number of samples to use for bootstrap sampling if `is_bootstrap` is True. Default is 1000.
        percent_len_sample : float, optional
            Percentage of the length of the sample to use for bootstrap sampling if `is_bootstrap` is True. Default is 0.8.
        test_size : float, optional
            Proportion of the dataset to include in the test split when using `train_test_split`. Default is 0.2.
        mode : str, optional
            Mode for building the tree. Options are "recursive", "bfs" (breadth-first search), or "dfs" (depth-first search). Default is "bfs".
    """
    def __init__(
        self, 
        max_depth=5, 
        min_samples_split=5, 
        min_samples_leaf=5,
        sigma=0.5, 
        function="normal", 
        is_custom_dist=False,
        is_bootstrap=False,
        n_components=10,
        n_samples=1000,
        percent_len_sample=0.8,
        test_size=0.2,
        mode="bfs",
        aggregator="mean"
    ):
        self.tree = None
        self.max_depth = (2**31) - 1 if max_depth is None else max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sigma = sigma 
        self.epsilon = 10e-12
        self.function = function.lower()
        self.custom_dist = None
        self.is_bootstrap = is_bootstrap
        self.is_custom_dist = is_custom_dist
        self.n_samples = n_samples
        self.percent_len_sample = percent_len_sample
        self.test_size = test_size
        self.mode = mode
        self.aggregator = aggregator

        self.X_train = None
        self.y_death_train = None
        self.y_time_train = None

        if is_custom_dist:
            if function == "weibull":
                self.custom_dist = Weibull()
            elif function == "logistic":
                self.custom_dist = LogLogistic()
            elif function == "normal":
                self.custom_dist = LogNormal()
            elif function == "extreme":
                self.custom_dist = LogExtremeNew()
            elif function == "gmm":
                self.custom_dist = GMM_New(n_components=n_components)
            else:
                raise ValueError("Custom distribution not supported")

    def fit(self, X, y, random_state=42):
        """
        Fit the AFTSurvivalTree model to the training data.
        :param X: array-like, shape (n_samples, n_features)
            Training data features.
        :param y: structured array, shape (n_samples,)
            Training data labels with fields 'death' and 'd.time'.
        :param random_state: int, optional
            Random seed for reproducibility. Default is 42.
        :return: None
        """

        if self.custom_dist is not None:
            if self.is_bootstrap:
                ## If bootstrap sampling is enabled, fit the whole dataset to the custom distribution
                self.custom_dist.fit_bootstrap(y, n_samples=self.n_samples, percentage=self.percent_len_sample)

                X_gpu = cp.asarray(X)
                y_death_gpu = cp.asarray(y['death'])
                y_time_gpu = cp.asarray(y['d.time'])

            else:
                X_gpu = cp.asarray(X)
                y_death_gpu = cp.asarray(y['death'])
                y_time_gpu = cp.asarray(y['d.time'])

                ## If bootstrap sampling is not enabled, split the dataset into training and distribution sets
                X_train, y_death_train, y_time_train, X_dist, y_death_dist, y_time_dist = stratified_gpu_train_test_split(X_gpu, y_death_gpu, y_time_gpu, test_size=self.test_size, random_state=random_state)

                y_death_dist_cpu = cp.asnumpy(y_death_dist)
                y_time_dist_cpu = cp.asnumpy(y_time_dist)

                y_dist = np.rec.fromarrays([y_death_dist_cpu, y_time_dist_cpu], names=['death', 'd.time'])

                self.custom_dist.fit(y_dist)

                X_gpu = X_train
                y_death_gpu = y_death_train
                y_time_gpu = y_time_train
        else:
            ## If no custom distribution is used, use the whole dataset
            X_gpu = cp.asarray(X)
            y_death_gpu = cp.asarray(y['death'])
            y_time_gpu = cp.asarray(y['d.time'])

        if self.mode == "recursive":
            self.build_tree(X_gpu, y_death_gpu, y_time_gpu)
        elif self.mode == "bfs":
            self.build_tree_bfs(X_gpu, y_death_gpu, y_time_gpu)
        elif self.mode == "dfs":
            self.build_tree_dfs(X_gpu, y_death_gpu, y_time_gpu)
        return

    def prefit(self, X, y, random_state=None):
        """
        Pre-fit the AFTSurvivalTree model to the training data.
        This method is used to set the training data and labels before building the tree.
        Mainly used for Forest GPU Parallelization.
        :param X: array-like, shape (n_samples, n_features)
        :param y: structured array, shape (n_samples,)
            Training data labels with fields 'death' and 'd.time'.
        :param random_state: int, optional
            Random seed for reproducibility. Default is None.
        :return: None
        """
        if self.custom_dist is None:
            self.X_train = X
            self.y_death_train = y['death']
            self.y_time_train = y['d.time']
        else:
            if self.is_bootstrap:
                self.custom_dist.fit_bootstrap(y, n_samples=self.n_samples, percentage=self.percent_len_sample)
                self.X_train = X
                self.y_death_train = y['death']
                self.y_time_train = y['d.time']

            else:
                X_train, _, y_train, y_dist = train_test_split(X, y, stratify=y['death'], test_size=self.test_size, random_state=random_state)
                self.custom_dist.fit(y_dist)

                self.X_train = X_train
                self.y_death_train = y_train['death']
                self.y_time_train = y_train['d.time']
        return

    def special_fit(self):
        """
        Special fit method to build the tree only after pre-fitting the model with prefit.
        This method is used to build the tree after the training data and labels have been set.

        """
        if self.X_train is None or self.y_death_train is None or self.y_time_train is None:
            raise ValueError("Model must be pre-fitted with prefit method before calling special_fit.")

        X_gpu = cp.asarray(self.X_train)
        y_death_gpu = cp.asarray(self.y_death_train)
        y_time_gpu = cp.asarray(self.y_time_train)

        if self.mode == "recursive":
            self.build_tree(X_gpu, y_death_gpu, y_time_gpu)
        elif self.mode == "bfs":
            self.build_tree_bfs(X_gpu, y_death_gpu, y_time_gpu)
        elif self.mode == "dfs":
            self.build_tree_dfs(X_gpu, y_death_gpu, y_time_gpu)
        return

    def build_tree(self, X, y_death, y_time, depth=0):   
        """
        Build the AFTSurvivalTree recursively.
        :param X: array-like, shape (n_samples, n_features)
        :param y_death: array-like, shape (n_samples,)
            Binary array indicating whether the event (death) occurred.
        :param y_time: array-like, shape (n_samples,)
            Array of survival times.
        :param depth: int, optional
            Current depth of the tree. Default is 0.
        :return: TreeNode
        """

        value = self.set_leaf_value(y_time)

        if depth > self.max_depth or len(y_time) < self.min_samples_split:
            node = TreeNode(None, None, value, None, None, num_sample=len(y_time))
            if depth == 0: 
                self.tree = node
            return node
            
        split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized(X, y_death, y_time)

        if split is None and feature is None:
            node = TreeNode(None, None, value, None, None, num_sample=len(y_time))
            if depth == 0: 
                self.tree = node
            return node
        
        X_left = X[left_indices]
        y_death_left = y_death[left_indices]
        y_time_left = y_time[left_indices]
        X_right = X[right_indices]
        y_death_right = y_death[right_indices]
        y_time_right = y_time[right_indices]
        

        if len( y_time_left) == 0 or len(y_time_right) == 0:
            node = TreeNode(None, None, value, None, None, num_sample=len(y_time))
            if depth == 0: 
                self.tree = node
            return node

        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            node = TreeNode(feature, None, value, None, None, num_sample=len(y_time))
            if depth == 0:
                self.tree = node
            return node

        left = self.build_tree(X_left, y_death_left, y_time_left, depth+1)
        right = self.build_tree(X_right, y_death_right, y_time_right, depth+1)
    
        node = TreeNode(feature, split, None, left, right, loss=loss, num_sample=len(y_time))
        if depth == 0:
            self.tree = node
        return node

    def build_tree_bfs(self, X, y_death, y_time, depth=0): 
        """
        Build the AFTSurvivalTree using breadth-first search (BFS).
        :param X: array-like, shape (n_samples, n_features)
        :param y_death: array-like, shape (n_samples,)
            Binary array indicating whether the event (death) occurred. 
        :param y_time: array-like, shape (n_samples,)
            Array of survival times.
        :param depth: int, optional 
            Current depth of the tree. Default is 0.
        :return: TreeNode
        """

        queue = deque()
        root = TreeNode(None, None, None, None, None, num_sample=len(y_time))
        self.tree = root

        queue.append({
            'parent_node':root,
            'X': X,
            'y_death': y_death,
            'y_time': y_time,
            'depth': depth
        })

        while queue:
            current_node = queue.popleft()
            X_current = current_node['X']
            y_death_current = current_node['y_death']
            y_time_current = current_node['y_time']
            depth = current_node['depth']
            parent_node = current_node['parent_node']

            value = self.set_leaf_value(y_time_current)

            if depth > self.max_depth or len(y_time_current) < self.min_samples_split:
                parent_node.set_value(value)
                continue
            
            split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized(X_current, y_death_current, y_time_current)

            if split is None and feature is None:
                parent_node.set_value(value)
                continue

            X_left = X[left_indices]
            y_death_left = y_death[left_indices]
            y_time_left = y_time[left_indices]
            X_right = X[right_indices]
            y_death_right = y_death[right_indices]
            y_time_right = y_time[right_indices]

            if len(X_left) == 0 or len(X_right) == 0:
                parent_node.set_value(value)
                continue

            if (len(y_time_left) < self.min_samples_leaf) or (len(y_time_right) < self.min_samples_leaf):
                parent_node.set_value(value)
            else:
                parent_node.set_feature_index(feature)
                parent_node.set_threshold(split)
                parent_node.set_loss(loss)
                parent_node.set_num_sample(len(y_time_current))

                parent_node.set_left(TreeNode(None, None, None, None, None, num_sample=len(y_time_left)))
                parent_node.set_right(TreeNode(None, None, None, None, None, num_sample=len(y_time_right)))

                queue.append({
                    'parent_node':parent_node.left,
                    'X': X_left,
                    'y_death': y_death_left,
                    'y_time': y_time_left,
                    'depth': depth + 1
                })

                queue.append({
                    'parent_node':parent_node.right,
                    'X': X_right,
                    'y_death': y_death_right,
                    'y_time': y_time_right,
                    'depth': depth + 1
                })

        return self.tree  

    def build_tree_dfs(self, X, y_death, y_time, depth=0):
        """
        Build the AFTSurvivalTree using depth-first search (DFS).
        :param X: array-like, shape (n_samples, n_features)
        :param y_death: array-like, shape (n_samples,)
            Binary array indicating whether the event (death) occurred.
        :param y_time: array-like, shape (n_samples,)   
            Array of survival times.
        :param depth: int, optional
            Current depth of the tree. Default is 0.  
        :return: TreeNode
        """     
        stack = []
        root = TreeNode(None, None, None, None, None, num_sample=len(y_time))
        self.tree = root

        stack.append({
            'parent_node':root,
            'X': X,
            'y_death': y_death,
            'y_time': y_time,
            'depth': depth
        })

        while stack:
            current_node = stack.pop()
            X_current = current_node['X']
            y_death_current = current_node['y_death']
            y_time_current = current_node['y_time']
            depth = current_node['depth']
            parent_node = current_node['parent_node']

            value = self.set_leaf_value(y_time_current)

            if depth > self.max_depth or len(y_time_current) < self.min_samples_split:
                parent_node.set_value(value)
                continue
            
            split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized(X_current, y_death_current, y_time_current)

            if split is None and feature is None:
                parent_node.set_value(value)
                continue

            X_left = X[left_indices]
            y_death_left = y_death[left_indices]
            y_time_left = y_time[left_indices]
            X_right = X[right_indices]
            y_death_right = y_death[right_indices]
            y_time_right = y_time[right_indices]

            if (len(y_time_left) < self.min_samples_leaf) or (len(y_time_right) < self.min_samples_leaf):
                parent_node.set_value(value)
            else:
                parent_node.set_feature_index(feature)
                parent_node.set_threshold(split)
                parent_node.set_loss(loss)
                parent_node.set_num_sample(len(y_time_current))

                parent_node.left = TreeNode(None, None, None, None, None, num_sample=len(y_time_left))
                parent_node.right = TreeNode(None, None, None, None, None, num_sample=len(y_time_right))

                stack.append({
                    'parent_node':parent_node.left,
                    'X': X_left,
                    'y_death': y_death_left,
                    'y_time': y_time_left,
                    'depth': depth + 1
                })

                stack.append({
                    'parent_node':parent_node.right,
                    'X': X_right,
                    'y_death': y_death_right,
                    'y_time': y_time_right,
                    'depth': depth + 1
                })

        return self.tree

    def set_leaf_value(self, y_time_current):
        """
        Set the value of the node based on the survival times.
        :param y_time_current: array-like, shape (n_samples,)
            Array of survival times.
        :return: None
        """
        if self.aggregator == "mean":
            self.value = cp.exp(self.mean_y(cp.log(y_time_current)))
        elif self.aggregator == "median":
            self.value = cp.exp(self.median_y(cp.log(y_time_current)))
        else:
            raise ValueError("Aggregator not supported. Use 'mean' or 'median'.")
        
        return self.value

    def get_best_split_vectorized(self, X, y_death, y_time):
        """
        Get the best split for the AFTSurvivalTree using vectorized operations.
        :param X: array-like, shape (n_samples, n_features)
            Training data features.
        :param y_death: array-like, shape (n_samples,)
            Binary array indicating whether the event (death) occurred.
        :param y_time: array-like, shape (n_samples,)
            Array of survival times.
        :return: tuple
            (best_split, best_feature, best_left_indices, best_right_indices, best_loss)
            - best_split: float or None
                The value at which to split the feature.
            - best_feature: int or None
                The index of the feature to split on.
            - best_left_indices: array-like
                Indices of samples in the left child.
            - best_right_indices: array-like
                Indices of samples in the right child.
            - best_loss: float
                The loss value for the split.
        """
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = cp.inf

        n_samples = len(X)
        n_features = len(X[0])

        for feature in range(n_features):
            feature_values = X[:, feature]
            unique_values = cp.unique(feature_values)
            
            if len(unique_values) <= 1:
                continue 

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            left_mask = feature_values[:, cp.newaxis] <= thresholds
            right_mask = ~left_mask

            left_counts = cp.sum(left_mask, axis=0)
            right_counts = cp.sum(right_mask, axis=0)

            valid_splits = (left_counts >= self.min_samples_split) & (right_counts >= self.min_samples_split)

            if not cp.any(valid_splits):
                continue

            thresholds = thresholds[valid_splits]
            left_mask = left_mask[:, valid_splits]
            right_mask = right_mask[:, valid_splits]
            left_counts = left_counts[valid_splits]
            right_counts = right_counts[valid_splits]

            if self.aggregator == "mean":
                pred_left = cp.array([self.mean_y(cp.log(y_time[left_mask[:, i]])) for i in range(len(thresholds))])
                pred_right = cp.array([self.mean_y(cp.log(y_time[right_mask[:, i]])) for i in range(len(thresholds))])
            elif self.aggregator == "median":
                pred_left = cp.array([self.median_y(cp.log(y_time[left_mask[:, i]])) for i in range(len(thresholds))])
                pred_right = cp.array([self.median_y(cp.log(y_time[right_mask[:, i]])) for i in range(len(thresholds))])
            else:
                raise ValueError("Aggregator not supported. Use 'mean' or 'median'.")

            if len(thresholds) == 0:
                continue

            left_loss = cp.array([self.calculate_loss_vectorized(y_death[left_mask[:, i]], y_time[left_mask[:, i]], pred_left[i]) for i in range(len(thresholds))])
            right_loss = cp.array([self.calculate_loss_vectorized(y_death[right_mask[:, i]], y_time[right_mask[:, i]], pred_right[i]) for i in range(len(thresholds))])

            total_loss = (left_loss * left_counts + right_loss * right_counts) / n_samples

            best_idx = cp.argmin(total_loss).item()
            current_min_loss = total_loss[best_idx].item()

            if current_min_loss < best_loss:
                best_loss = current_min_loss
                best_split = thresholds[best_idx].item()
                best_feature = feature
                best_left_indices = cp.where(left_mask[:, best_idx])[0]
                best_right_indices = cp.where(right_mask[:, best_idx])[0]

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def calculate_loss_vectorized(self, y_death, y_time, pred=None):
        """
            Calculate the loss for the given survival data.
            :param y_death: array-like, shape (n_samples,)
            Binary array indicating whether the event (death) occurred.
            :param y_time: array-like, shape (n_samples,)
                Array of survival times.
            :param pred: array-like, shape (n_samples,), optional   
                Predicted values. If None, the mean of y_time is used.
            :return: float, the total loss for the given survival data
        """
        is_uncensored = y_death.astype(bool)

        uncensored_loss = self.get_uncensored_value(y_time, pred)
        censored_loss = self.get_censored_value(y_time, cp.inf, pred)

        loss = cp.where(is_uncensored, uncensored_loss, censored_loss)

        return cp.sum(loss)
    
    def get_uncensored_value(self, y, pred):
        """
            Calculate the loss for uncensored data.
            :param y: float, the uncensored survival time
            :param pred: float, the predicted value
            :return: float, the loss for uncensored data
        """
        if self.custom_dist is not None:
            link_function = cp.log(y) - pred
            pdf = self.custom_dist.pdf_gpu(link_function)
        else:
            if self.function == "normal":
                pdf = norm_pdf(y, pred, self.sigma)
            elif self.function == "logistic":
                pdf = logistic_pdf(y, pred, self.sigma)
            elif self.function == "extreme":
                pdf = extreme_pdf(y, pred, self.sigma)
            else:
                raise ValueError("Distribution not supported")

        pdf = cp.maximum(pdf, self.epsilon)
        return -cp.log(pdf)

    def get_censored_value(self, y_lower, y_upper, pred):
        """
            Calculate the loss for censored data.
            :param y_lower: float, lower bound of the censored interval
            :param y_upper: float, upper bound of the censored interval
            :param pred: float, predicted value
            :return: float, the loss for censored data
        """

        if self.custom_dist is not None:
            link_function_lower = cp.log(y_lower) - pred
            link_function_upper = cp.log(y_upper) - pred
            cdf_diff = self.custom_dist.cdf_gpu(link_function_upper) - self.custom_dist.cdf_gpu(link_function_lower)
        else:
            if self.function == "normal":
                cdf_diff = norm_cdf(y_upper, pred, self.sigma) - norm_cdf(y_lower, pred, self.sigma)
            elif self.function == "logistic":
                cdf_diff = logistic_cdf(y_upper, pred, self.sigma) - logistic_cdf(y_lower, pred, self.sigma)
            elif self.function == "extreme":
                cdf_diff = extreme_cdf(y_upper, pred, self.sigma) - extreme_cdf(y_lower, pred, self.sigma)
            else:
                raise ValueError("Distribution not supported")

        cdf_diff = cp.maximum(cdf_diff, self.epsilon)
        return -cp.log(cdf_diff)

    def mean_y(self, y):
        """
            Calculate the mean of y.
            param: y: np.ndarray, shape (n_samples,)
            return: float, the mean of y
        """
        return cp.mean(y)

    def median_y(self, y):
        """
            Calculate the median of y.
            param: y: np.ndarray, shape (n_samples,)
        """
        return cp.median(y)

    def predict(self, X):
        """
            Predict the survival time for the input samples X.
            param: X: np.ndarray, shape (n_samples, n_features)
        """
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
        predictions = [self.get_prediction(x, self.tree) for x in X]
        return predictions

    def get_prediction(self, X, tree):
        """
            Get the prediction for a single sample X using the tree.
            param: X: np.ndarray, shape (n_features,)
            Param: tree: TreeNode, the current node in the tree
            return: float, the predicted time  
        """
        try:
            if tree.value is not None:
                if isinstance(tree.value, cp.ndarray):
                    return tree.value.get()
                else:
                    return tree.value
            else:
                feature_value = X[tree.feature_index]
                if feature_value <= tree.threshold:
                    return self.get_prediction(X, tree.left)
                else:
                    return self.get_prediction(X, tree.right)
        except:
            raise ValueError("Error in get_prediction")

    def _print(self):
        """
            Print the tree structure.   
        """
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        else:
            self.print_tree(self.tree)

    def print_tree(self, tree, indent=" "):
        """
            Print the tree structure.
            param: tree: TreeNode
            param: indent: str, indentation for printing
        """
        if tree is None:
            print(f"{indent}None")
            return

        if tree.value is not None:
            print(f"{indent}value: {tree.value}")
        else:
            print(f"{indent}X_{tree.feature_index} <= {tree.threshold}")
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + "  ")
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + "  ")

    def _score(self, X, y):
        """
            Compute the concordance index.
            param: X: np.ndarray, shape (n_samples, n_features)
            param: y_true: list of tuples, where each tuple is (censored, time)
            return: float, the concordance index    
        """
        pred_times = self.predict(X)
        return c_index(pred_times, y)

    def _brier(self, X, y):
        """
            Compute the Integrated Brier Score (IBS).
            param: X: np.ndarray, shape (n_samples, n_features)
            param: y: list of tuples, where each tuple is (censored, time)
            return: float, the integrated brier score
        """
        pred_times = self.predict(X)
        return brier(pred_times, y)


    def _auc(self, X, y):
        """
            Compute the Area Under the Curve (AUC).
            param: X: np.ndarray, shape (n_samples, n_features)
            param: y: list of tuples, where each tuple is (censored, time)
            return: float, the area under the curve
        """
        pred_times = self.predict(X)
        return auc(pred_times, y)

    def _mae(self, X, y):
        """
            Compute the Mean Absolute Error (MAE).
            param: X: np.ndarray, shape (n_samples, n_features)
            param: y: list of tuples, where each tuple is (censored, time)
            return: float, the mean absolute error
        """
        pred_times = self.predict(X)
        return mae(pred_times, y)

    def _visualize(self, path='doctest-output/decision_tree'):
        '''
            Visualize the tree using graphviz.
        '''

        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        else:
            dot = graphviz.Digraph(comment='AFT Survival Tree')
            dot = self.visualize(dot, self.tree)
            dot.render(path).replace('\\', '/')

    def visualize(self, dot, tree, node_id=None):
        '''
            Visualize the tree using graphviz.
            param: dot: graphviz.Digraph
            param: tree: TreeNode
            param: node_id: str, optional
            return: graphviz.Digraph
        '''

        if node_id is None:
            node_id = str(uuid.uuid4())

        if tree.value is not None:
            dot.node(node_id, f"value: {np.round(tree.value, 3)} \n num_sample = {tree.num_sample}", shape='rectangle')
        else:
            dot.node(node_id, f"X_{tree.feature_index} <= {np.round(tree.threshold, 2)} \n loss = {np.round(tree.loss, 2)} \n num_sample = {tree.num_sample}", shape='rectangle')
            if tree.left is not None:
                node_left = str(uuid.uuid4())
                dot = self.visualize(dot, tree.left, node_left)
                dot.edge(node_id, node_left)
            if tree.right is not None:
                node_right = str(uuid.uuid4())
                dot = self.visualize(dot, tree.right, node_right)
                dot.edge(node_id, node_right)
            
        return dot

    def save(self, path):
        '''
            Save the model to a JSON file.
            param: path: str
        '''

        model_state = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'sigma': float(self.sigma),
            'function': self.function,
            'epsilon': float(self.epsilon),
            'is_custom_dist': self.is_custom_dist,
            'is_bootstrap': self.is_bootstrap,
            'tree': self.tree.to_dict() if self.tree is not None else None
        }
        
        if self.custom_dist is not None:
            model_state['custom_dist_type'] = self.custom_dist.__class__.__name__
            model_state['custom_dist_params'] = self.custom_dist.get_params()
        
        with open(path, 'w') as f:
            json.dump(model_state, f, indent=4)

    @classmethod
    def load(cls, path):
        '''
            Load a model from a JSON file.
            param: path: str
            return: AFTSurvivalTree instance
        '''

        with open(path, 'r') as f:
            model_state = json.load(f)
        
        model = cls(
            max_depth=model_state['max_depth'],
            min_samples_split=model_state['min_samples_split'],
            min_samples_leaf=model_state['min_samples_leaf'],
            sigma=model_state['sigma'],
            function=model_state['function'],
            is_custom_dist=model_state['is_custom_dist'],
            is_bootstrap=model_state['is_bootstrap']
        )
        
        model.epsilon = model_state['epsilon']
        
        if 'custom_dist_type' in model_state:
            dist_type = model_state['custom_dist_type']
            dist_params = model_state['custom_dist_params']
            
            if dist_type == 'Weibull':
                model.custom_dist = Weibull()
            elif dist_type == 'LogLogistic':
                model.custom_dist = LogLogistic()
            elif dist_type == 'LogNormal':
                model.custom_dist = LogNormal()
            elif dist_type == 'LogExtreme':
                model.custom_dist = LogExtreme()
            elif dist_type == 'GMM':
                model.custom_dist = GMM()

            model.custom_dist.set_params(dist_params)
        
        if model_state['tree'] is not None:
            model.tree = TreeNode.from_dict(model_state['tree'])
        
        return model


