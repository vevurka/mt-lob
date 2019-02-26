import logging

import numpy as np
import pandas as pd
from sklearn import utils
from lob_data_utils import lob, model

from sklearn.svm import SVC


logger = logging.getLogger(__name__)


class LobClassifySimple(object):

    def __init__(self, stock: str, data_length: int=10000, data_dir: str='../data/prepared'):
        self.stock = stock
        self.data_length = data_length
        self.data_dir = data_dir
        self.df, self.df_test = self._read_stock()

    feature_columns_dict = {
        'que': ['queue_imbalance'],
        'que_prev': ['queue_imbalance', 'prev_queue_imbalance'],
    }

    def get_score_for_clf(self, clf, df_test: pd.DataFrame, feature_name: str) -> dict:
        x_test = df_test[self.feature_columns_dict[feature_name]]
        y_test = df_test['mid_price_indicator'].values
        return model.test_model(clf, x_test, y_test)

    @staticmethod
    def get_mean_scores(scores: dict) -> dict:
        mean_scores = {}
        for k, v in scores.items():
            mean_scores[k] = np.mean(v)
        return mean_scores

    def get_classes_weights(self):
        y_train = self.df['mid_price_indicator'].values
        classes = np.unique(y_train)
        class_weight_list = utils.class_weight.compute_class_weight('balanced', classes, y_train)
        class_weights = {classes[0]: class_weight_list[0], classes[1]: class_weight_list[1]}
        return class_weights

    def train_clf(self, clf, feature_name='', should_validate=True, method=None, class_weight=None):
        logger.info('Training %s: clf=%s', self.stock, clf)
        train_x = self.df[self.feature_columns_dict[feature_name]]
        if should_validate:
            scores_arrays = model.validate_model(
                clf, train_x, self.df['mid_price_indicator'], class_weight=class_weight)
            scores = self.get_mean_scores(scores_arrays)
        else:
            scores = model.train_model(clf, train_x, self.df['mid_price_indicator'],
                                       class_weight=class_weight)
        if not method:
            method = 'logistic'
        res = {
            **scores,
            'stock': self.stock,
            'kernel': method,
            'features': feature_name,
        }
        test_scores = self.get_score_for_clf(clf, self.df_test, feature_name=feature_name)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        return {**res, **test_scores}

    def train_svm(self, C=np.nan, gamma=np.nan, feature_name='', kernel='rbf', coef0=np.nan,
                  should_validate=True, class_weight=None):
        logger.info('Training %s: kernel=%s C=%s gamma=%s coef0=%s',
                    self.stock, kernel, C, gamma, coef0)
        if C and gamma and coef0:
            clf = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0, class_weight=class_weight)
        elif C and gamma:
            clf = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight)
        else:
            clf = SVC(kernel=kernel, class_weight=class_weight)
        train_x = self.df[self.feature_columns_dict[feature_name]]
        if should_validate:
            scores_arrays = model.validate_model(clf, train_x, self.df['mid_price_indicator'])
            scores = self.get_mean_scores(scores_arrays)
        else:
            scores = model.train_model(clf, train_x, self.df['mid_price_indicator'])

        res = {
            **scores,
            'stock': self.stock,
            'C': C,
            'gamma': gamma,
            'coef0': coef0,
            'kernel': kernel,
            'features': feature_name,
        }
        test_scores = self.get_score_for_clf(clf, self.df_test, feature_name=feature_name)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        return {**res, **test_scores}

    def _read_stock(self):
        reg_filename = '{}'.format(self.stock)
        logger.debug('Will read %s', reg_filename)
        df, df_test = lob.load_prepared_data(reg_filename, data_dir=self.data_dir, length=self.data_length)

        df['datetime'] = df['Unnamed: 0']
        df['prev_queue_imbalance'] = df['queue_imbalance'].shift()
        df.index = pd.to_datetime(df['datetime'])
        df.dropna(inplace=True)
        df_test['datetime'] = df_test['Unnamed: 0']
        df_test['prev_queue_imbalance'] = df_test['queue_imbalance'].shift()
        df_test.index = pd.to_datetime(df_test['datetime'])
        df_test.dropna(inplace=True)
        return df, df_test
