from collections import Counter

import numpy as np

from DecisionTree import DecisionTree


class DecisionTreeClassifier(DecisionTree):
    class LeafNode:
        def __init__(self, y, number_of_classes):
            counter = Counter(y)
            n = sum(counter.values())
            self.class_name = np.array([counter.most_common(1)[0][0]])
            self.prob = np.array([counter[i] / n
                                  for i in range(number_of_classes)])

        def get_prediction(self):
            return self.class_name

        def get_prob(self):
            return self.prob



    def __init__(self, max_depth=np.inf, min_samples_split=2, criterion='gini', debug=False):
        super().__init__(max_depth, min_samples_split, criterion, debug)



    def _create_leaf_node(self, y):
        return DecisionTreeClassifier.LeafNode(y, self.number_of_classes)



    def predict_proba(self, X):
        return np.array([node.get_prob()
                         for node in self._predict(X, self._tree_root)])
