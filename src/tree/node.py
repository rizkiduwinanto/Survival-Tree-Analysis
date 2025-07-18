import numpy as np
import cupy as cp

class TreeNode():
    '''
    A class representing a node in a decision tree.
    Each node contains information about the feature index, threshold,
    value, left and right children, maximum depth, loss, and number of samples.
    '''

    def __init__(self, feature_idx, threshold, value=None, left=None, right=None, max_depth=None, loss=None, num_sample=None, y_time=None, y_death=None):
        self.feature_index = feature_idx
        self.threshold = threshold
        self.loss = loss
        self.num_sample = num_sample
        self.value = value
        self.left = left
        self.right = right
        self.max_depth = max_depth
        self.y_time = y_time
        self.y_death = y_death

    def set_feature_index(self, feature_index):
        '''
        Set the feature index for the node.
        :param feature_index: Index of the feature used for splitting.
        '''
        self.feature_index = feature_index 

    def set_threshold(self, threshold):
        '''
        Set the threshold value for the node.
        :param threshold: Threshold value for the feature at this node.
        '''
        self.threshold = threshold

    def set_loss(self, loss):
        '''
        Set the loss value for the node.
        :param loss: Loss value associated with the node.
        '''
        self.loss = loss
    
    def set_num_sample(self, num_sample):
        '''
        Set the number of samples at this node.
        :param num_sample: Number of samples that reach this node.
        '''
        self.num_sample = num_sample

    def set_value(self, value):
        '''
        Set the value for the node.
        :param value: Value associated with the node, typically the prediction.
        '''
        self.value = value

    def set_left(self, left):
        '''
        Set the left child of the node.
        :param left: Left child TreeNode.
        '''
        self.left = left
    
    def set_right(self, right):
        '''
        Set the right child of the node.
        :param right: Right child TreeNode.
        '''
        self.right = right
    
    def set_max_depth(self, max_depth):
        '''
        Set the maximum depth of the node.
        :param max_depth: Maximum depth of the tree.
        '''
        self.max_depth = max_depth

    def set_y_time(self, y_time):
        '''
        Set the time values associated with the node.
        :param y_time: Time values for the samples at this node.
        '''
        self.y_time = y_time

    def set_y_death(self, y_death):
        '''
        Set the death values associated with the node.
        :param y_death: Death values for the samples at this node.
        '''
        self.y_death = y_death

    def set_node(self, y_time=None, y_death=None):
        '''
        Set the node's value and optionally its time and death values.
        :param value: Value to set for the node.
        :param y_time: Optional time values for the samples at this node.
        :param y_death: Optional death values for the samples at this node.
        '''
        if y_time is not None:
            self.set_y_time(y_time)

        if y_death is not None:
            self.set_y_death(y_death)

    def is_leaf(self):
        '''
        Check if the node is a leaf node.
        A leaf node is defined as one that has no left or right children.
        :return: True if the node is a leaf, False otherwise.
        '''
        return self.left is None and self.right is None

    def get_value(self):
        '''
        Get the value of the node.
        :return: The value associated with the node.
        '''
        return self.value

    def set_leaf_value_median(self):
        '''
        Set the value of the node to the median of the y_time values.
        This is typically used for leaf nodes in regression trees.
        '''
        if self.y_time is not None and len(self.y_time) > 0:
            events = self.y_time[self.y_death == 1] if self.y_death is not None else self.y_time
            self.value = float(np.median(events))
        else:
            self.value = None

    def set_leaf_value_geometric(self, is_extreme=False, sigma=1):
        '''
        Set the value of the node to the geometric mean of the y_time values.
        This is typically used for leaf nodes in survival analysis trees.
        '''
        if self.y_time is not None and len(self.y_time) > 0:
            events = self.y_time[self.y_death == 1] if self.y_death is not None else self.y_time
            mean_value = np.mean(np.log(events)) if len(events) > 0 else 0
            self.value = float(np.exp(mean_value)) if not is_extreme else float(np.exp(mean_value) + sigma * np.log(np.log(2)))
        else:
            self.value = None

    @staticmethod
    def to_serializable(val):
        """Convert common non-serializable types to serializable formats."""
        if val is None:
            return None
        try:
            if hasattr(val, 'tolist'):
                return val.tolist()
            elif hasattr(val, 'item'):
                return val.item()
            return val
        except Exception as e:
            raise ValueError(f"Could not serialize value {val}: {str(e)}")

    def to_dict(self):
        """Convert the TreeNode instance to a dictionary representation."""
        node_dict = {
            'feature_index': self.to_serializable(self.feature_index),
            'threshold': self.to_serializable(self.threshold),
            'loss': self.to_serializable(self.loss),
            'num_sample': self.to_serializable(self.num_sample),
            'value': self.to_serializable(self.value),
            'max_depth': self.to_serializable(self.max_depth),
            'left': self.left.to_dict() if self.left is not None else None,
            'right': self.right.to_dict() if self.right is not None else None,
            'y_time': self.to_serializable(self.y_time),
            'y_death': self.to_serializable(self.y_death)
        }
        return node_dict

    @classmethod
    def from_dict(cls, node_dict):
        """Create a TreeNode instance from a dictionary representation."""
        if node_dict is None:
            return None
            
        return cls(
            feature_idx=node_dict.get('feature_index'),
            threshold=node_dict.get('threshold'),
            value=node_dict.get('value'),
            left=cls.from_dict(node_dict.get('left')),
            right=cls.from_dict(node_dict.get('right')),
            max_depth=node_dict.get('max_depth'),
            loss=node_dict.get('loss'),
            num_sample=node_dict.get('num_sample'),
            y_time=node_dict.get('y_time'),
            y_death=node_dict.get('y_death')
        )