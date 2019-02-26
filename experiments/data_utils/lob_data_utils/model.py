import unittest

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
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
        self.assertEqual(res['recall'], 0.0)
        np.testing.assert_array_equal(res['roc_auc'], np.nan)

    def test_validate_true_model(self):
        x = np.zeros(5).tolist() + np.ones(5).tolist()
        res = validate_model(self.MockClf(), x, x, folds=2)
        self.assertEqual(res['precision'], 1.0)
        self.assertEqual(res['f1'], 1.0)
        self.assertEqual(res['recall'], 1.0)
        np.testing.assert_array_equal(res['roc_auc'], np.nan)


def _divide_folds(x, y, i, folds=10, step_size=3, print_debug=False):
    fold_size = len(x) // (folds + 1)
    if print_debug:
        print('train: ', (i - step_size) * fold_size, i * fold_size)
        print('val:', (i + 0) * fold_size, (i + 2) * fold_size)
    x_fold_train = x[(i - step_size) * fold_size: i * fold_size]
    y_fold_train = y[(i - step_size) * fold_size: i * fold_size]
    x_fold_test = x[(i + 0) * fold_size: (i + 2) * fold_size]
    y_fold_test = y[(i + 0) * fold_size: (i + 2) * fold_size]
    return x_fold_train, y_fold_train, x_fold_test, y_fold_test


def calculate_scores(labels, prediction, prefix=None):
    f1_score = metrics.f1_score(labels, prediction)
    recall_score = metrics.recall_score(labels, prediction)
    precision_score = metrics.precision_score(labels, prediction)
    kappa_score = metrics.cohen_kappa_score(labels, prediction)
    matthews_score = metrics.matthews_corrcoef(labels, prediction)
    try:
        roc_auc_score = metrics.roc_auc_score(labels, prediction)
    except ValueError as e:
        roc_auc_score = np.nan
    if not prefix:
        return {
            'precision'.format(prefix): precision_score,
            'f1'.format(prefix): f1_score,
            'recall'.format(prefix): recall_score,
            'roc_auc'.format(prefix): roc_auc_score,
            'kappa'.format(prefix): kappa_score,
            'matthews'.format(prefix): matthews_score,
        }
    else:
        return {
            '{}_precision'.format(prefix): precision_score,
            '{}_f1'.format(prefix): f1_score,
            '{}_recall'.format(prefix): recall_score,
            '{}_roc_auc'.format(prefix): roc_auc_score,
            '{}_kappa'.format(prefix): kappa_score,
            '{}_matthews'.format(prefix): matthews_score,
        }


def test_model(clf, test_data, labels, prefix=None, is_lstm=False):
    if not prefix:
        prefix = 'test'
    if is_lstm:
        prediction = clf.predict_classes(test_data)
    else:
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
        '{}_precision'.format(prefix): precision_score,
        '{}_f1'.format(prefix): f1_score,
        '{}_recall'.format(prefix): recall_score,
        '{}_roc_auc'.format(prefix): roc_auc_score,
        '{}_kappa'.format(prefix): kappa_score,
        '{}_matthews'.format(prefix): matthews_score,
    }


def train_model(clf, train_data, labels, prefix=None, fit_kwargs=None, compile_kwargs=None, is_lstm=False,
                class_weight=None, validation_data=None):
    if not prefix:
        prefix = 'train'
    if is_lstm:
        if not fit_kwargs or not compile_kwargs:
            raise Exception('You need to set fit and compile kwargs for LSTM')
    if fit_kwargs and is_lstm and compile_kwargs:
        clf.compile(**compile_kwargs)
        if validation_data:
            clf.fit(train_data, labels, **fit_kwargs, class_weight=class_weight,
                    validation_data=validation_data,
                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                             verbose=1, mode='auto')])
        else:
            clf.fit(train_data, labels, **fit_kwargs, class_weight=class_weight)
    else:
        if class_weight:
            clf.fit(train_data, labels, class_weight=class_weight)
        else:
            clf.fit(train_data, labels)
    return test_model(clf, train_data, labels, prefix=prefix, is_lstm=is_lstm)


def validate_model(clf, train_data, labels, folds=10, print_debug=False, fit_kwargs=None, compile_kwargs=None,
                   class_weight=None):
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
    step_size = 5
    for i in range(step_size, folds, 1):
        if print_debug:
            print('Training fold ', i, len(train_data))
        x_fold_train, y_fold_train, x_fold_test, y_fold_test = _divide_folds(
            train_data, labels, i, folds=folds, step_size=step_size, print_debug=print_debug)
        validation_data = (x_fold_test, y_fold_test)
        train_model(clf, x_fold_train, y_fold_train, fit_kwargs=fit_kwargs, compile_kwargs=compile_kwargs,
                    is_lstm=False, class_weight=class_weight, validation_data=(x_fold_test, y_fold_test))

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
    train_scores = train_model(clf, train_data, labels, fit_kwargs=fit_kwargs, compile_kwargs=compile_kwargs,
                               is_lstm=False, class_weight=class_weight)
    return {
        'precision': precision_scores,
        'f1': f1_scores,
        'recall': recall_scores,
        'roc_auc': roc_auc_scores,
        'kappa': kappa_scores,
        'matthews': matthews_scores,
        'train_val_precision': train_precision_scores,
        'train_val_f1': train_f1_scores,
        'train_val_recall': train_recall_scores,
        'train_val_roc_auc': train_roc_auc_scores,
        'train_val_kappa': train_kappa_scores,
        'train_val_matthews': train_matthews,
        **train_scores
    }


def validate_model_lstm(get_model, train_data, labels, folds=10, print_debug=False,
                        fit_kwargs=None, compile_kwargs=None, plot_name=None, class_weight=None):
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
    step_size = 5
    for i in range(step_size, folds, 1):
        if print_debug:
            print('Training fold ', i, len(train_data))
        x_fold_train, y_fold_train, x_fold_test, y_fold_test = _divide_folds(
            train_data, labels, i, folds=folds, step_size=step_size, print_debug=print_debug)
        validation_data = (x_fold_test, y_fold_test)
        m = get_model()
        train_model(m, x_fold_train, y_fold_train, fit_kwargs=fit_kwargs,
                    compile_kwargs=compile_kwargs, is_lstm=True,
                    class_weight=class_weight, validation_data=(x_fold_test, y_fold_test))

        prediction = m.predict_classes(x_fold_test)
        train_prediction = m.predict_classes(x_fold_train)

        f1_scores.append(metrics.f1_score(y_fold_test, prediction))
        recall_scores.append(metrics.recall_score(y_fold_test, prediction))
        precision_scores.append(metrics.precision_score(y_fold_test, prediction))
        kappa_scores.append(metrics.cohen_kappa_score(y_fold_test, prediction))
        matthews_scores.append(metrics.matthews_corrcoef(y_fold_test, prediction))
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
    if plot_name:
        m = get_model()
        history = m.fit(train_data, labels, **fit_kwargs, class_weight=class_weight,
                        validation_data=validation_data)
        for k in history.history.keys():
            if validation_data and 'val' in k:
                continue
            plt.figure()
            plt.title(k)
            plt.plot(history.history[k], label='Train ' + k)
            plt.plot(history.history['val_' + k], label='Validation ' + k)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{plot_name}_{k}.png')
            plt.close('all')

    m = get_model()
    train_scores = train_model(m, train_data, labels, fit_kwargs=fit_kwargs, compile_kwargs=compile_kwargs,
                               class_weight=class_weight, is_lstm=True)
    return {
               'precision': precision_scores,
               'f1': f1_scores,
               'recall': recall_scores,
               'roc_auc': roc_auc_scores,
               'kappa': kappa_scores,
               'matthews': matthews_scores,
               'train_val_precision': train_precision_scores,
               'train_val_f1': train_f1_scores,
               'train_val_recall': train_recall_scores,
               'train_val_roc_auc': train_roc_auc_scores,
               'train_val_kappa': train_kappa_scores,
               'train_val_matthews': train_matthews,
               **train_scores
           }, m
