import numpy as np

from DecisionTree import DecisionTree


class DecisionTreeRegressor(DecisionTree):
    class LeafNode:
        def __init__(self, y):
            self.full_info = y.mean()

        def get_prediction(self):
            return self.full_info

        def get_full_info(self):
            return self.full_info



    def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='variance', debug=False):
        super().__init__(max_depth, min_samples_split, criterion, debug)



    def _create_leaf_node(self, y):
        return DecisionTreeRegressor.LeafNode(y)
