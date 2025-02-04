import numpy as np
import math
from node import TreeNode
from utils import norm_pdf, norm_cdf, logistic_pdf, logistic_cdf, extreme_pdf, extreme_cdf


## To Do:
## Check XGboost examples
## Fix Print the tree
## implement mean as the threshold
## fix the prediction
## implement C-index
## UI/UX bootstartp

## Implement the predict method
class AFTSurvivalTree():
    """
        Regression tree that implements AFTLoss
    """
    def __init__(self, max_depth=10, min_samples_split=1, min_samples_leaf=1, sigma=0.5):
        self.tree = None
        self.max_depth = (2**31) - 1 if max_depth is None else max_depth

        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.sigma = sigma 
        self.epsilon = 10e-12

    def fit(self, X, y):
        self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):    

        if depth > self.max_depth:
            # print('A')
            return TreeNode(None, None, np.mean(y), None, None)

        if len(y) < self.min_samples_split:
    
            # print('B')
            return TreeNode(None, None, np.mean(y), None, None)
            
        ## Choose best feature to split on
        split, feature, left_indices, right_indices = self.get_best_split(X, y)

        if left_indices is None or right_indices is None:
            
            # print('C')
            return TreeNode(feature, None, np.mean(y), None, None)

        if len(left_indices[0]) < self.min_samples_leaf or len(right_indices[0]) < self.min_samples_leaf:
            
            # print('D')
            return TreeNode(feature, None, np.mean(y), None, None)

        # print("f: y:", y[left_indices], y[right_indices])

        left = self.build_tree(X[left_indices], y[left_indices], depth+1)
        right = self.build_tree(X[right_indices], y[right_indices], depth+1)

        # print('E')
    
        self.tree = TreeNode(feature, split, None, left, right)

    def get_best_split(self, X, y):
        best_split = None
        best_feature = None
        best_left_indices = None
        best_right_indices = None
        best_loss = np.inf

        for feature in range(X.shape[1]):
            for split in np.unique(X[:, feature]):
                left_indices = np.where(X[:, feature] <= split)
                right_indices = np.where(X[:, feature] > split)

                left_y = y[left_indices]
                right_y = y[right_indices]

                init_pred = self.calculate_loss(y, pred=0)

                left_loss = self.calculate_loss(left_y, init_pred)
                right_loss = self.calculate_loss(right_y, init_pred)

                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_split = split
                    best_feature = feature
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_split, best_feature, best_left_indices, best_right_indices

    def calculate_loss(self, y, pred=0):
        loss = 0
        for i in range(len(y)):
            censor, value = y[i]
            if censor:
                loss += self.get_censored_value(value, np.inf, pred)
            else:
                loss += self.get_uncensored_value(value, pred)
        return loss
    
    def get_uncensored_value(self, y, pred):
        pdf = norm_pdf(y, pred, self.sigma)
        if pdf <= 0:
            pdf = self.epsilon
        return -np.log(pdf/(self.sigma*y))

    def get_censored_value(self, y_lower, y_upper, pred):
        cdf_diff = norm_cdf(y_upper, pred, self.sigma) - norm_cdf(y_lower, pred, self.sigma)
        if cdf_diff <= 0:
            cdf_diff = self.epsilon
        return -np.log(cdf_diff)

    def predict(self, X):
        if self.tree is None:
            return None
            
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

    def print(self):
        if self.tree is None:
            print("Tree is empty")
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

    def _score(self, X, y):
        pass