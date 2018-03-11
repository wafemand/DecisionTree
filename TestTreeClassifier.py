import unittest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from DecisionTreeClassifier import *


class TestTreeClassifier(unittest.TestCase):
    def test_digits(self):
        digits = load_digits()
        X = digits.images
        X = np.array(list(map(np.ravel, X)))
        y = digits.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=17)

        tree = self.get_fitted_tree(X_train, y_train, debug=False, max_depth=5, criterion='entropy')

        y_predicted = tree.predict(X_test)
        score = accuracy_score(y_test, y_predicted)
        self.assertGreaterEqual(score, 0.5)



    def test_simple(self):
        X = np.zeros((100, 1))
        X[:, 0] = np.linspace(-30, 30, 100)
        y = np.int8(X < 0).T[0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

        tree = self.get_fitted_tree(X_train, y_train, debug=False, max_depth=2, criterion='entropy')

        y_predicted = tree.predict(X_test)
        score = accuracy_score(y_test, y_predicted)
        self.assertGreaterEqual(score, 0.9999)



    def test_proba(self):
        X = np.zeros((100, 1))
        X[:, 0] = np.linspace(-30, 30, 100)
        y = np.int8(X < 0).T[0]

        tree = self.get_fitted_tree(X, y, debug=False, max_depth=2, criterion='entropy')

        X_test = np.array([[-1000, -1, 1, 1000]])
        y_proba = tree.predict_proba(X_test)
        for i in range(y_proba.shape[0]):
            for j in range(y_proba.shape[1]):
                if X_test[i][0] < 0 and j == 1 or X_test[i][0] >= 0 and j == 0:
                    self.assertGreaterEqual(y_proba[i][j], 0.9999)
                else:
                    self.assertLessEqual(y_proba[i][j], 0.0001)



    @staticmethod
    def get_fitted_tree(X, y, **kwargs):
        tree = DecisionTreeClassifier(**kwargs)
        tree.fit(X, y)
        return tree


if __name__ == '__main__':
    unittest.main()
