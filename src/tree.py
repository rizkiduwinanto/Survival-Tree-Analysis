import numpy as np
import math
from joblib import Parallel, delayed
import multiprocessing
from node import TreeNode
from distribution import Weibull, LogLogistic, LogNormal, LogExtreme, GMM
from math_utils import norm_pdf, norm_cdf, logistic_pdf, logistic_cdf, extreme_pdf, extreme_cdf
from lifelines.utils import concordance_index
import graphviz
import uuid
from concurrent.futures import ThreadPoolExecutor
import numba as nb
import json
from sklearn.model_selection import train_test_split
from sksurv.metrics import integrated_brier_score
import time

class AFTSurvivalTree():
    """
        Regression tree that implements AFTLoss
    """
    def __init__(
        self, 
        max_depth=5, 
        min_samples_split=5, 
        min_samples_leaf=5,
        sigma=0.5, 
        function="norm", 
        is_custom_dist=False,
        is_bootstrap=False,
        n_components=10
    ):
        self.tree = None
        self.max_depth = (2**31) - 1 if max_depth is None else max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sigma = sigma 
        self.epsilon = 10e-12
        self.function = function
        self.custom_dist = None
        self.is_bootstrap = is_bootstrap
        self.is_custom_dist = is_custom_dist

        if is_custom_dist:
            if function == "weibull":
                self.custom_dist = Weibull()
            elif function == "logistic":
                self.custom_dist = LogLogistic()
            elif function == "norm":
                self.custom_dist = LogNormal()
            elif function == "extreme":
                self.custom_dist = LogExtreme()
            elif function == "gmm":
                self.custom_dist = GMM(n_components=n_components)
            else:
                raise ValueError("Custom distribution not supported")

    def fit(self, X, y, split=None, test_size=0.2):
        if self.custom_dist is not None:
            if self.is_bootstrap:
                self.custom_dist.fit_bootstrap(y)
                self.build_tree(X, y)
                return
            else:
                x_train, x_dist, y_train, y_dist = train_test_split(X, y, test_size=test_size, random_state=42)
                self.custom_dist.fit(y_dist)
                self.build_tree(x_train, y_train)
                return
        else:
            self.build_tree(X, y)
            return

    def build_tree(self, X, y, depth=0):   
        if depth > self.max_depth or len(y) < self.min_samples_split:
            node = TreeNode(None, None, self.mean_y(y), None, None, num_sample=len(y))
            if depth == 0: 
                self.tree = node
            return node
            
        split, feature, left_indices, right_indices, loss = self.get_best_split_vectorized(X, y)
        
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        if len(y_left) == 0 or len(y_right) == 0:
            node = TreeNode(None, None, self.mean_y(y), None, None, num_sample=len(y))
            if depth == 0: 
                self.tree = node
            return node

        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            node = TreeNode(feature, None, self.mean_y(y), None, None, num_sample=len(y))
            if depth == 0:
                self.tree = node
            return node

        left = self.build_tree(X_left, y_left, depth+1)
        right = self.build_tree(X_right, y_right, depth+1)
    
        node = TreeNode(feature, split, None, left, right, loss=loss, num_sample=len(y))
        if depth == 0:
            self.tree = node
        return node

    def get_best_split(self, X, y, feature):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        n_samples = len(X)

        sorted_indices = np.argsort(X[:, feature])
        sorted_X = X[sorted_indices, feature]
        sorted_y = y[sorted_indices]

        for i in range(n_samples - 1):
            if sorted_X[i] != sorted_X[i+1]:
                split = (sorted_X[i] + sorted_X[i+1]) / 2

                left_y = sorted_y[:i]
                right_y = sorted_y[i:]

                mean_y = self.mean_y(y)

                left_loss = self.calculate_loss(left_y, mean_y)
                right_loss = self.calculate_loss(right_y, mean_y)

                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = split
                    best_feature = feature
                    best_left_indices = sorted_indices[:i]
                    best_right_indices = sorted_indices[i:]

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def get_best_feature(self, X, y):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        n_features = len(X[0])

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_best_split, X, y, feature) for feature in range(n_features)]
            
            for future in futures:
                split, feature, left_indices, right_indices, loss = future.result()

                if loss < best_loss:
                    best_loss = loss
                    best_split = split
                    best_feature = feature
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss

    def get_best_split_vectorized(self, X, y):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        n_samples = len(X)
        n_features = len(X[0])

        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])
            thresholds = unique_values[:, np.newaxis]

            left_mask = X[:, feature][np.newaxis, :] < thresholds
            right_mask = X[:, feature][np.newaxis, :] >= thresholds

            mean_y = self.mean_y(y)

            valid_splits = np.logical_and(np.any(left_mask, axis=1), np.any(right_mask, axis=1))
            thresholds = thresholds[valid_splits]
            left_mask = left_mask[valid_splits]
            right_mask = right_mask[valid_splits]

            if len(thresholds) == 0:
                continue

            left_loss = np.array([self.calculate_loss(y[mask], mean_y) for mask in left_mask])
            right_loss = np.array([self.calculate_loss(y[mask], mean_y) for mask in right_mask])
    
            left_counts = np.sum(left_mask, axis=1)
            right_counts = np.sum(right_mask, axis=1)

            total_loss = left_loss + right_loss

            best_idx = np.argmin(total_loss)
            current_min_loss = total_loss[best_idx].item()

            if current_min_loss < best_loss:
                best_loss = current_min_loss
                best_split = thresholds[best_idx].item()
                best_feature = feature
                best_left_indices = np.where(left_mask[best_idx])[0]
                best_right_indices = np.where(right_mask[best_idx])[0]

        return best_split, best_feature, best_left_indices, best_right_indices, best_loss
            
    def calculate_loss(self, y, pred=None):
        if pred is None:
            pred = self.mean_y(y)

        loss = 0
        for i in range(len(y)):
            censor, value = y[i]
            if censor:
                loss += self.get_censored_value(value, np.inf, pred)
            else:
                loss += self.get_uncensored_value(value, pred)
        return loss
    
    def get_uncensored_value(self, y, pred):
        if self.custom_dist is not None:
            link_function = np.log(y) - pred
            pdf = self.custom_dist.pdf(link_function)
        else:
            if self.function == "norm":
                pdf = norm_pdf(y, pred, self.sigma)
            elif self.function == "logistic":
                pdf = logistic_pdf(y, pred, self.sigma)
            elif self.function == "extreme":
                pdf = extreme_pdf(y, pred, self.sigma)
            else:
                raise ValueError("Distribution not supported")

        if pdf <= 0:
            pdf = self.epsilon
        return -np.log(pdf/(self.sigma*y))

    def get_censored_value(self, y_lower, y_upper, pred):
        if self.custom_dist is not None:
            link_function_lower = np.log(y_lower) - pred
            link_function_upper = np.log(y_upper) - pred
            cdf_diff = self.custom_dist.cdf(link_function_upper) - self.custom_dist.cdf(link_function_lower)
        else:
            if self.function == "norm":
                cdf_diff = norm_cdf(y_upper, pred, self.sigma) - norm_cdf(y_lower, pred, self.sigma)
            elif self.function == "logistic":
                cdf_diff = logistic_cdf(y_upper, pred, self.sigma) - logistic_cdf(y_lower, pred, self.sigma)
            elif self.function == "extreme":
                cdf_diff = extreme_cdf(y_upper, pred, self.sigma) - extreme_cdf(y_lower, pred, self.sigma)
            else:
                raise ValueError("Distribution not supported")

        if cdf_diff <= 0:
            cdf_diff = self.epsilon
        return -np.log(cdf_diff)

    def mean_y(self, y):
        return np.mean([value for _, value in y])

    def predict(self, X):
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            X = X.reshape(1, -1)
        predictions = [self.get_prediction(x, self.tree) for x in X]
        return predictions

    def get_prediction(self, X, tree):
        if tree.value is not None:
            return tree.value
        else:
            feature_value = X[tree.feature_index]
            if feature_value <= tree.threshold:
                return self.get_prediction(X, tree.left)
            else:
                return self.get_prediction(X, tree.right)

    def _print(self):
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        else:
            self.print_tree(self.tree)

    def print_tree(self, tree, indent=" "):
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

    def _score(self, X, y_true):
        """
            Implement C-Index
        """
        times_pred = self.predict(X)
        event_true = [1 if not censored else 0 for censored, _ in y_true]
        times_true = [time for _, time in y_true]

        c_index = concordance_index(times_true, times_pred, event_true)
        return c_index

    def _brier(self, X, y_train, y_test):
        """
            Implement Integrated Brier Score
        """
        times_pred = self.predict(X)
        event_true = [1 if not censored else 0 for censored, _ in y_true]
        times_true = [time for _, time in y_true]

        y_true_structured = np.array([(not censored, time) for censored, time in y_true], dtype=[('event', '?'), ('time', '<f8')])

        min_time = min(times_true)
        max_time = max(times_true)
        time_points = np.linspace(min_time, max_time, 100)

        ibs = integrated_brier_score(y_train, y_test, times_pred, time_points)
        return ibs

    def _visualize(self):
        if self.tree is None:
            raise ValueError("Tree has not been built. Call `fit` first.")
        else:
            dot = graphviz.Digraph(comment='AFT Survival Tree')
            dot = self.visualize(dot, self.tree)
            dot.render('doctest-output/decision_tree').replace('\\', '/')

    def visualize(self, dot, tree, node_id=None):
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


