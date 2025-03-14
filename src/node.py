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

    def to_dict(self):
        node_dict = {
            'feature_index': self.feature_index,
            'threshold': float(self.threshold) if self.threshold is not None else None,
            'loss': float(self.loss) if self.loss is not None else None,
            'num_sample': self.num_sample,
            'value': float(self.value) if self.value is not None else None,
            'max_depth': self.max_depth,
            'left': self.left.to_dict() if self.left is not None else None,
            'right': self.right.to_dict() if self.right is not None else None
        }
        return node_dict

    @classmethod
    def from_dict(cls, node_dict):
        if node_dict is None:
            return None
            
        left = cls.from_dict(node_dict['left'])
        right = cls.from_dict(node_dict['right'])
        
        return cls(
            feature_idx=node_dict['feature_index'],
            threshold=node_dict['threshold'],
            value=node_dict['value'],
            left=left,
            right=right,
            max_depth=node_dict['max_depth'],
            loss=node_dict['loss'],
            num_sample=node_dict['num_sample']
        )