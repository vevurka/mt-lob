import unittest
import numpy as np
from sklearn import metrics


class TestValidateModel(unittest.TestCase):
    class MockClf(object):
        def __init__(self, to_return=None):
            self.to_return = to_return

        def fit(self, X, y):
            return self

        def predict(self, y):
            if self.to_return is None:
                return y
            else:
                return self.to_return[0:len(y)]

    def test_divide_fold_first_fold(self):
        x_train, y_train, x_test, y_test = _divide_folds(np.ones(10), np.ones(10), 0, 2)
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_train), 3)
        self.assertEqual(len(x_test), 3)

    def test_divide_fold_second_fold(self):
        x_train, y_train, x_test, y_test = _divide_folds(np.ones(10), np.ones(10), 1, 2)
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_train), 6)
        self.assertEqual(len(x_test), 3)

    def test_divide_fold_last_fold(self):
        x_train, y_train, x_test, y_test = _divide_folds(np.ones(10), np.ones(10), 2, 2)
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_train), 9)
        self.assertEqual(len(x_test), 1)

    def test_validate_false_model(self):
        res = validate_model(self.MockClf(to_return=np.zeros(10)), np.ones(10), np.ones(10), folds=2)
        self.assertEqual(res['precision'], 0.0)
        self.assertEqual(res['f1'], 0.0)
        self.assertEqual(res['recall'],  0.0)
        np.testing.assert_array_equal(res['roc_auc'], np.nan)

    def test_validate_true_model(self):
        x = np.zeros(5).tolist() + np.ones(5).tolist()
        res = validate_model(self.MockClf(), x, x, folds=2)
        self.assertEqual(res['precision'], 1.0)
        self.assertEqual(res['f1'], 1.0)
        self.assertEqual(res['recall'],  1.0)
        np.testing.assert_array_equal(res['roc_auc'], np.nan)


def _divide_folds(x, y, i, folds=10):
    fold_size = len(x) // (folds + 1)
    x_fold_train = x[0: (i + 1) * fold_size]
    y_fold_train = y[0: (i + 1) * fold_size]
    x_fold_test = x[(i + 1) * fold_size: (i + 2) * fold_size]
    y_fold_test = y[(i + 1) * fold_size: (i + 2) * fold_size]
    return x_fold_train, y_fold_train, x_fold_test, y_fold_test


def test_model(clf, test_data, labels):
    prediction = clf.predict(test_data)
    f1_score = metrics.f1_score(labels, prediction)
    recall_score = metrics.recall_score(labels, prediction)
    precision_score = metrics.precision_score(labels, prediction)
    kappa_score = metrics.cohen_kappa_score(labels, prediction)
    matthews_score = metrics.matthews_corrcoef(labels, prediction)
    try:
        roc_auc_score = metrics.roc_auc_score(labels, prediction)
    except ValueError as e:
        roc_auc_score = np.nan
    return {
        'test_precision': precision_score,
        'test_f1': f1_score,
        'test_recall': recall_score,
        'test_roc_auc': roc_auc_score,
        'test_kappa': kappa_score,
        'test_matthews': matthews_score,
    }


def validate_model(clf, train_data, labels, folds=10):
    f1_scores = []
    recall_scores = []
    precision_scores = []
    roc_auc_scores = []
    kappa_scores = []
    matthews_scores = []
    train_f1_scores = []
    train_recall_scores = []
    train_precision_scores = []
    train_roc_auc_scores = []
    train_kappa_scores = []
    train_matthews = []

    for i in range(folds - 1):
        x_fold_train, y_fold_train, x_fold_test, y_fold_test = _divide_folds(train_data, labels, i, folds=folds)

        clf.fit(x_fold_train, y_fold_train)
        prediction = clf.predict(x_fold_test)
        f1_scores.append(metrics.f1_score(y_fold_test, prediction))
        recall_scores.append(metrics.recall_score(y_fold_test, prediction))
        precision_scores.append(metrics.precision_score(y_fold_test, prediction))
        kappa_scores.append(metrics.cohen_kappa_score(y_fold_test, prediction))
        matthews_scores.append(metrics.matthews_corrcoef(y_fold_test, prediction))

        train_prediction = clf.predict(x_fold_train)
        train_f1_scores.append(metrics.f1_score(y_fold_train, train_prediction))
        train_recall_scores.append(metrics.recall_score(y_fold_train, train_prediction))
        train_precision_scores.append(metrics.precision_score(y_fold_train, train_prediction))
        train_kappa_scores.append(metrics.cohen_kappa_score(y_fold_train, train_prediction))
        train_matthews.append(metrics.matthews_corrcoef(y_fold_train, train_prediction))
        try:
            roc_auc_scores.append(metrics.roc_auc_score(y_fold_test, prediction))
            train_roc_auc_scores.append(metrics.roc_auc_score(y_fold_train, train_prediction))
        except ValueError as e:
            roc_auc_scores.append(np.nan)
            train_roc_auc_scores.append(metrics.roc_auc_score(y_fold_train, train_prediction))
    return {
        'precision': precision_scores,
        'f1': f1_scores,
        'recall': recall_scores,
        'roc_auc': roc_auc_scores,
        'kappa': kappa_scores,
        'matthews': matthews_scores,
        'train_precision': train_precision_scores,
        'train_f1': train_f1_scores,
        'train_recall': train_recall_scores,
        'train_roc_auc': train_roc_auc_scores,
        'train_kappa': train_kappa_scores,
        'train_matthews': train_matthews
    }
