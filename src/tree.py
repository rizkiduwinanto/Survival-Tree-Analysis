import numpy as np
import math
from joblib import Parallel, delayed
import multiprocessing
from node import TreeNode
from utils import norm_pdf, norm_cdf, logistic_pdf, logistic_cdf, extreme_pdf, extreme_cdf
from lifelines.utils import concordance_index
import graphviz
import uuid

class AFTSurvivalTree():
    """
        Regression tree that implements AFTLoss
    """
    def __init__(self, max_depth=5, min_samples_split=5, min_samples_leaf=5, sigma=0.5, function="norm"):
        self.tree = None
        self.max_depth = (2**31) - 1 if max_depth is None else max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sigma = sigma 
        self.epsilon = 10e-12
        self.function = function

    def fit(self, X, y):
        self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):    
        if depth > self.max_depth or len(y) < self.min_samples_split:
            node = TreeNode(None, None, self.mean_y(y), None, None, num_sample=len(y))
            if depth == 0: 
                self.tree = node
            return node
            
        split, feature, left_indices, right_indices, loss = self.get_best_split(X, y)
        
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

    def get_best_split(self, X, y):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        n_samples, n_features = len(X), len(X[0])

        for feature in range(n_features):
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
        if self.function == "norm":
            pdf = norm_pdf(y, pred, self.sigma)
        elif self.function == "logistic":
            pdf = logistic_pdf(y, pred, self.sigma)
        else:
            pdf = extreme_pdf(y, pred, self.sigma)

        if pdf <= 0:
            pdf = self.epsilon
        return -np.log(pdf/(self.sigma*y))

    def get_censored_value(self, y_lower, y_upper, pred):
        if self.function == "norm":
            cdf_diff = norm_cdf(y_upper, pred, self.sigma) - norm_cdf(y_lower, pred, self.sigma)
        elif self.function == "logistic":
            cdf_diff = logistic_cdf(y_upper, pred, self.sigma) - logistic_cdf(y_lower, pred, self.sigma)
        else:
            cdf_diff = extreme_cdf(y_upper, pred, self.sigma) - extreme_cdf(y_lower, pred, self.sigma)

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
        pass


