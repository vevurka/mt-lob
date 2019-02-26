import unittest

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from calculation.calculation import SVMCalculationPerformer


class TestCalculation(unittest.TestCase):

    def test_svm_alg_data_is_prepared(self):
        iris = load_iris()
        classification = SVC(kernel='linear', gamma=3, C=1, coef0=34)
        classification.fit(iris.data, iris.target)
        cp = SVMCalculationPerformer(classification)
        data = cp.prepare_data(3)
        self.assertEqual(data['algorithm']['svm'], {
            'kernel': 'linear',
            'gamma': 3,
            'c': 1,
            'coef0': 34})

    def test_svm_alg_data_is_executed(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])

        classification = SVC(kernel='linear', gamma=3, C=1, coef0=34)
        classification.fit(X, y)
        cp = SVMCalculationPerformer(classification)
        data = cp.execute([[-0.8, -1], [-1, -1]], [1, 0], 'random stuff')
        self.assertEqual(data, 'sdf')