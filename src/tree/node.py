
class TreeNode():
    '''
    A class representing a node in a decision tree.
    Each node contains information about the feature index, threshold,
    value, left and right children, maximum depth, loss, and number of samples.
    '''

    def __init__(self, feature_idx, threshold, value, left=None, right=None, max_depth=None, loss=None, num_sample=None):
        self.feature_index = feature_idx
        self.threshold = threshold
        self.loss = loss
        self.num_sample = num_sample
        self.value = value
        self.left = left
        self.right = right
        self.max_depth = max_depth

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

    def to_dict(self):
        '''
        Convert the TreeNode instance to a dictionary representation.
        This is useful for serialization or saving the tree structure.
        '''

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
        '''
        Create a TreeNode instance from a dictionary representation.
        This is useful for deserialization or loading the tree structure.
        '''

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