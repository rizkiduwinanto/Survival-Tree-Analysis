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
from numba import njit, jit
import numpy as np

@jit(forceobj=True)
def _get_best_split(X, y, is_custom_dist, custom_dist, function, sigma, epsilon):
    best_split = None
    best_feature = None
    best_left_indices = None
    best_right_indices = None
    best_loss = np.inf

    n_samples = len(X)
    n_features = len(X[0])

    for feature in range(n_features):
        sorted_indices = np.argsort(X[:, feature])
        sorted_X = X[sorted_indices, feature]
        sorted_y = y[sorted_indices]

        best_split, best_feature, best_left_indices, best_right_indices, best_loss  = _get_best_split_feature(
            X,
            y, 
            feature, 
            sorted_indices, 
            sorted_X, 
            sorted_y, 
            n_samples, 
            is_custom_dist, 
            custom_dist, 
            function, 
            sigma, 
            epsilon
        )

    return best_split, best_feature, best_left_indices, best_right_indices, best_loss

@jit(forceobj=True)
def _get_best_split_feature(X, y, feature, sorted_indices, sorted_X, sorted_y, n_samples, is_custom_dist, custom_dist, function, sigma, epsilon):
    best_split = None
    best_feature = None
    best_left_indices = None
    best_right_indices = None
    best_loss = np.inf

    for i in range(n_samples - 1):
        if sorted_X[i] != sorted_X[i+1]:
            split = (sorted_X[i] + sorted_X[i+1]) / 2

            left_y = sorted_y[:i]
            right_y = sorted_y[i:]

            pred = mean_y(y)

            if is_custom_dist:
                left_loss  = _calculate_loss(left_y, pred, is_custom_dist, custom_dist.pdf, custom_dist.cdf, function, sigma, epsilon)
                right_loss = _calculate_loss(right_y, pred, is_custom_dist, custom_dist.pdf, custom_dist.cdf, function, sigma, epsilon)
            else:
                left_loss  = _calculate_loss(left_y, pred, is_custom_dist, None, None, function, sigma, epsilon)
                right_loss = _calculate_loss(right_y, pred, is_custom_dist, None, None, function, sigma, epsilon)

            total_loss = left_loss + right_loss

            if total_loss < best_loss:
                best_loss = total_loss
                best_split = split
                best_feature = feature
                best_left_indices = sorted_indices[:i]
                best_right_indices = sorted_indices[i:]

    return best_split, best_feature, best_left_indices, best_right_indices, best_loss

@jit(forceobj=True)
def _get_uncensored_value(y, pred, is_custom, custom_pdf, function, sigma, epsilon):
    if is_custom:
        link_function = np.log(y) - pred
        pdf = custom_pdf(link_function)
    else:
        if function == "norm":
            pdf = norm_pdf(y, pred, sigma)
        elif function == "logistic":
            pdf = logistic_pdf(y, pred, sigma)
        elif function == "extreme":
            pdf = extreme_pdf(y, pred, sigma)
        else:
            raise ValueError("Distribution not supported")

    if pdf <= 0:
        pdf = epsilon
    return -np.log(pdf/(sigma*y))

@jit(forceobj=True)
def _get_censored_value(y_lower, y_upper, pred, is_custom, custom_cdf, function, sigma, epsilon):
    if is_custom:
        link_function_lower = np.log(y_lower) - pred
        link_function_upper = np.log(y_upper) - pred
        cdf_diff = custom_cdf(link_function_upper) - custom_cdf(link_function_lower)
    else:
        if function == "norm":
            cdf_diff = norm_cdf(y_upper, pred, sigma) - norm_cdf(y_lower, pred, sigma)
        elif function == "logistic":
            cdf_diff = logistic_cdf(y_upper, pred, sigma) - logistic_cdf(y_lower, pred, sigma)
        elif function == "extreme":
            cdf_diff = extreme_cdf(y_upper, pred, sigma) - extreme_cdf(y_lower, pred, sigma)
        else:
            raise ValueError("Distribution not supported")

    if cdf_diff <= 0:
        cdf_diff = epsilon
    return -np.log(cdf_diff)

@jit(nopython=True)
def mean_y(y):
    acc = 0
    for i in range(len(y)):
        value = y[i][1]
        acc += value
    return acc / len(y)

@jit(forceobj=True)
def _calculate_loss(y, pred, is_custom, custom_pdf, custom_cdf, function, sigma, epsilon):
    if pred is None:
        pred = mean_y(y)

    loss = 0
    for i in range(len(y)):
        censor = y[i][0]
        value = y[i][1]

        if censor:
            loss += _get_censored_value(value, np.inf, pred, is_custom, custom_pdf, function, sigma, epsilon)
        else:
            loss += _get_uncensored_value(value, pred, is_custom, custom_cdf, function, sigma, epsilon)
    return loss

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
        is_bootstrap=False
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
                self.custom_dist = GMM()
            else:
                raise ValueError("Custom distribution not supported")

    def fit(self, X, y):
        #Split the data into training and testing sets
        if self.custom_dist is not None:
            if self.is_bootstrap:
                self.custom_dist.fit_bootstrap(y)
            else:
                #KL Divergence on split data, KS Test on split data
                self.custom_dist.fit(y)
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
        return _get_best_split(X, y, self.is_custom_dist, self.custom_dist, self.function, self.sigma, self.epsilon)

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


