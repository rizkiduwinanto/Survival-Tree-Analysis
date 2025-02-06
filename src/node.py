import numpy as np

class TreeNode():
    def __init__(self, feature_idx, threshold, value, left=None, right=None, max_depth=None, loss=None, num_sample=None):
        self.feature_index = feature_idx
        self.threshold = threshold
        self.loss = loss
        self.num_sample = num_sample
        self.value = value
        self.left = left
        self.right = right
        self.max_depth = max_depth