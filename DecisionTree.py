import abc
from sklearn.base import BaseEstimator
from criterions import *


class DecisionTree(BaseEstimator):
    class Node:
        def __init__(self, predicate, node_left, node_right):
            self.predicate = predicate
            self.node_left = node_left
            self.node_right = node_right



    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion='gini', debug=False):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.debug = debug
        self._tree_root = None
        self.number_of_classes = 0



    @staticmethod
    def _get_predicate(feature, value):
        def predicate(X):
            return X[:, feature] < value



        return predicate



    def _get_best_predicate(self, X: np.ndarray, y: np.ndarray):
        best_score = -np.inf
        best_feature = 0
        best_value = 0
        for feature in range(X.shape[1]):
            sorted_X_ind = np.argsort(X[:, feature])
            sorted_y = y[sorted_X_ind]
            sorted_X = X[sorted_X_ind, feature]
            i = 0
            for score_l, score_r in criterion_generators[self.criterion](sorted_y):
                i += 1
                cur_score = -(score_l * i + score_r * (len(y) - i))
                if sorted_X[i] != sorted_X[i - 1] and best_score < cur_score:
                    best_score = cur_score
                    best_feature = feature
                    best_value = (sorted_X[i] + sorted_X[i - 1]) / 2
        predicate = self._get_predicate(best_feature, best_value)

        if self.debug:
            print("predicate found: feature =", best_feature,
                  "value =", best_value,
                  "score =", best_score)

        return predicate



    def _build_tree(self, X, y, depth=0):
        if len(X) < self.min_samples_split:
            if self.debug:
                print("created leaf node(min_samples_split): depth =", depth)
            return self._create_leaf_node(y)

        if depth >= self.max_depth:
            if self.debug:
                print("created leaf node(max_depth): depth =", depth)
            return self._create_leaf_node(y)

        if y.max() == y.min():
            if self.debug:
                print("created leaf node(all y are equals): depth =", depth)
            return self._create_leaf_node(y)

        predicate = self._get_best_predicate(X, y)

        mask = predicate(X)
        X_l, X_r = X[mask], X[~mask]
        y_l, y_r = y[mask], y[~mask]

        if len(X_l) == 0:
            if self.debug:
                print("created leaf node (bad split): depth =", depth)
            return self._create_leaf_node(y_r)

        elif len(X_r) == 0:
            if self.debug:
                print("created leaf node (bad split): depth =", depth)
            return self._create_leaf_node(y_l)

        return DecisionTree.Node(
            predicate,
            self._build_tree(X_l, y_l, depth=depth + 1),
            self._build_tree(X_r, y_r, depth=depth + 1))



    def _predict(self, X, node):
        if not isinstance(node, DecisionTree.Node):
            return np.array([node] * len(X))

        mask = node.predicate(X)
        y_l = iter(self._predict(X[mask], node.node_left))
        y_r = iter(self._predict(X[~mask], node.node_right))
        return np.array([next(y_l) if isLeft else next(y_r) for isLeft in mask])



    @abc.abstractmethod
    def _create_leaf_node(self, y):
        return



    def fit(self, X, y):
        self.number_of_classes = max(y) + 1
        self._tree_root = self._build_tree(X, y)

        if self.debug:
            print("fitted", self.get_params())



    def predict(self, X):
        return np.array([node.get_prediction() for node in self._predict(X, self._tree_root)])
