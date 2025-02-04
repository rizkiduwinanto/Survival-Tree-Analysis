import numpy as np

class TreeNode():
    def __init__(self, feature_idx, threshold, value, left=None, right=None, max_depth=None):
        self.feature_index = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.max_depth = max_depth

    def info(self):
        return self.feature_index, self.value