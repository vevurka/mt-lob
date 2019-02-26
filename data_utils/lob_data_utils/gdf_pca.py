import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import utils
from lob_data_utils import lob, model
from sklearn.decomposition import PCA
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class SvmGdfResults(object):

    def __init__(self, stock, r=1.0, s=1.0, data_length=10000, gdf_filename_pattern='',
                 data_dir='../data/data_gdf', reg_data_dir='../data/prepared'):
        self.stock = stock
        self.r = r
        self.s = s
        self.data_length = data_length
        self.gdf_filename_pattern = gdf_filename_pattern
        self.data_dir = data_dir
        self.reg_data_dir = reg_data_dir
        self.df, self.df_test = self._read_stock()

    all_gdf = ['gdf_{}'.format(i) for i in range(0, 50)]
    all_gdf_que = ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance']
    all_gdf_que_prev = ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance', 'prev_queue_imbalance']

    feature_columns_dict = {
        'que': ['queue_imbalance'],
        'que_prev': ['queue_imbalance', 'prev_queue_imbalance'],
        'gdf_24_26': ['gdf_24', 'gdf_25'],
        'gdf_24-26_que': ['gdf_24', 'gdf_25', 'queue_imbalance'],
        'gdf_24-26_que_prev': ['gdf_24', 'gdf_25', 'queue_imbalance', 'prev_queue_imbalance'],
        'gdf_23-27': ['gdf_23', 'gdf_24', 'gdf_25', 'gdf_26'],
        'gdf_23-27_que': ['gdf_23', 'gdf_24', 'gdf_25', 'gdf_26', 'queue_imbalance'],
        'gdf_23-27_que_prev': ['gdf_23', 'gdf_24', 'gdf_25', 'gdf_26', 'queue_imbalance', 'prev_queue_imbalance'],
        'gdf_20_30': ['gdf_{}'.format(i) for i in range(20, 30)],
        'gdf_20_30_que': ['gdf_{}'.format(i) for i in range(20, 30)] + ['queue_imbalance'],
        'gdf_20_30_que_prev': ['gdf_{}'.format(i) for i in range(20, 30)] + ['queue_imbalance', 'prev_queue_imbalance'],
        'gdf_0_50': all_gdf,
        'gdf_0-50_que': all_gdf_que,
        'gdf_0-50_que_prev': all_gdf_que_prev,
        'pca_gdf1': all_gdf,
        'pca_gdf2': all_gdf,
        'pca_gdf3': all_gdf,
        'pca_gdf4': all_gdf,
        'pca_gdf5': all_gdf,
        'pca_gdf6': all_gdf,
        'pca_gdf7': all_gdf,
        'pca_gdf8': all_gdf,
        'pca_gdf9': all_gdf,
        'pca_gdf10': all_gdf,
        'pca_n_gdf': all_gdf,
        'pca_gdf_que1': all_gdf_que,
        'pca_gdf_que2': all_gdf_que,
        'pca_gdf_que3': all_gdf_que,
        'pca_gdf_que4': all_gdf_que,
        'pca_gdf_que5': all_gdf_que,
        'pca_gdf_que6': all_gdf_que,
        'pca_gdf_que7': all_gdf_que,
        'pca_gdf_que8': all_gdf_que,
        'pca_gdf_que9': all_gdf_que,
        'pca_gdf_que10': all_gdf_que,
        'pca_n_gdf_que': all_gdf_que,
        'pca_gdf_que_prev1': all_gdf_que_prev,
        'pca_gdf_que_prev2': all_gdf_que_prev,
        'pca_gdf_que_prev3': all_gdf_que_prev,
        'pca_gdf_que_prev4': all_gdf_que_prev,
        'pca_gdf_que_prev5': all_gdf_que_prev,
        'pca_gdf_que_prev6': all_gdf_que_prev,
        'pca_gdf_que_prev7': all_gdf_que_prev,
        'pca_gdf_que_prev8': all_gdf_que_prev,
        'pca_gdf_que_prev9': all_gdf_que_prev,
        'pca_gdf_que_prev10': all_gdf_que_prev,
        'pca_n_gdf_que_prev': all_gdf_que_prev,
        'pca_gdf_que_prev_split10': all_gdf_que_prev
    }

    def get_score_for_clf(self, clf, df_test, feature_name, pca=None):
        x_test = df_test[self.feature_columns_dict[feature_name]]
        if pca:
            x_test = pca.transform(x_test)
        y_test = df_test['mid_price_indicator'].values
        return model.test_model(clf, x_test, y_test)

    @staticmethod
    def get_number_of_pca_components(feature_name: str) -> Optional[int]:
        if 'pca_gdf_que_prev_split' in feature_name:
            return int(feature_name.replace('pca_gdf_que_prev_split', ''))
        if 'pca_gdf_que_prev' in feature_name:
            return int(feature_name.replace('pca_gdf_que_prev', ''))
        if 'pca_gdf_que' in feature_name:
            return int(feature_name.replace('pca_gdf_que', ''))
        if 'pca_gdf' in feature_name:
            return int(feature_name.replace('pca_gdf', ''))
        return None

    @classmethod
    def split_sequences(cls, sequences, labels, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            end_ix = i + n_steps
            if end_ix > len(sequences):
                break
            seq_x = sequences[i:end_ix]
            lab = labels[end_ix - 1]
            X.append(seq_x)
            y.append(lab)
        return np.array(X), np.array(y)

    @staticmethod
    def get_mean_scores(scores: dict) -> dict:
        mean_scores = {}
        for k, v in scores.items():
            mean_scores[k] = np.mean(v)
        return mean_scores

    def get_score_for_clf_split_pca(self, clf, df_test, feature_name, pca=None) -> dict:
        x_test = df_test[self.feature_columns_dict[feature_name]]
        x_test_pca = x_test[[col for col in x_test.columns if 'gdf' in col]]
        x_test = x_test[[col for col in x_test.columns if 'gdf' not in col]]
        if pca:
            x_test_pca = pca.transform(x_test_pca)
        for n in range(pca.n_components):
            x_test['pca_{}'.format(n)] = x_test_pca[:, n]
        y_test = df_test['mid_price_indicator'].values
        return model.test_model(clf, x_test, y_test)

    def get_pca(self, feature_name) -> Optional[PCA]:
        train_x = self.df[self.feature_columns_dict[feature_name]].values
        if feature_name in ['pca_n_gdf_que', 'pca_n_gdf_que_prev', 'pca_n_gdf']:
            n_components = self.calculate_number_of_components(train_x, threshold=0.99)
        else:
            n_components = self.get_number_of_pca_components(feature_name)
        if n_components:
            pca = PCA(n_components=n_components)
            pca.fit(train_x)
            return pca
        return None

    @classmethod
    def calculate_number_of_components(cls, train_x, threshold=0.99) -> int:
        pca = PCA(n_components=10)
        pca.fit(train_x)
        for i in range(1, len(pca.explained_variance_ratio_)):
            sum_of_ratio = np.sum(pca.explained_variance_ratio_[0:i])
            if sum_of_ratio > threshold:
                return i
        return 10

    def get_classes_weights(self):
        y_train = self.df['mid_price_indicator'].values
        classes = np.unique(y_train)
        class_weight_list = utils.class_weight.compute_class_weight('balanced', classes, y_train)
        class_weights = {classes[0]: class_weight_list[0], classes[1]: class_weight_list[1]}
        return class_weights

    def train_clf_with_split_pca(self, clf, feature_name, method=None):
        """
        Deprecated
        """
        logger.info('Training %s r=%s s=%s:',
                    self.stock, self.r, self.s)
        train_x = self.df[self.feature_columns_dict[feature_name]]
        train_pca = train_x[[col for col in train_x.columns if 'gdf' in col]]
        train_x = train_x[[col for col in train_x.columns if 'gdf' not in col]]
        n_components = self.get_number_of_pca_components(feature_name)
        pca = None
        if n_components:
            pca = PCA(n_components=n_components)
            pca.fit(train_pca)
            train_pca = pca.transform(train_pca)
        for n in range(n_components):
            train_x['pca_{}'.format(n)] = train_pca[:, n]
        scores = model.validate_model(clf, train_x, self.df['mid_price_indicator'])
        res = {
            **self.get_mean_scores(scores),
            'stock': self.stock,
            'method': method,
            'features': feature_name
        }
        test_scores = self.get_score_for_clf_split_pca(clf, self.df_test, feature_name=feature_name, pca=pca)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        return {**res, **test_scores}

    def get_train_set(self, feature_name='', n_steps=None):
        train_x = self.df[self.feature_columns_dict[feature_name]].values
        train_y = self.df['mid_price_indicator'].values

        pca = self.get_pca(feature_name)
        if pca:
            train_x = pca.transform(train_x)
        if n_steps:
            train_x, train_y = self.split_sequences(train_x, train_y, n_steps=n_steps)
        else:

            train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        return train_x, train_y

    def get_test_set(self, feature_name='', n_steps=None):
        test_x = self.df_test[self.feature_columns_dict[feature_name]].values
        test_y = self.df_test['mid_price_indicator'].values

        pca = self.get_pca(feature_name)
        if pca:
            test_x = pca.transform(test_x)
        if n_steps:
            test_x, test_y = self.split_sequences(test_x, test_y, n_steps=n_steps)
        return test_x, test_y

    def train_mlp(self, clf, feature_name='', should_validate=True, method=None,
                  fit_kwargs=None, compile_kwargs=None, plot_name=None, class_weight=None,
                  should_return_model=False):
        logger.info('Training %s r=%s s=%s: clf=%s', self.stock, self.r, self.s, clf)

        train_x = self.df[self.feature_columns_dict[feature_name]].values
        test_x = self.df_test[self.feature_columns_dict[feature_name]].values
        train_y = self.df['mid_price_indicator'].values
        test_y = self.df_test['mid_price_indicator'].values

        pca = self.get_pca(feature_name)
        if pca:
            train_x = pca.transform(train_x)
            test_x = pca.transform(test_x)

        if should_validate:
            scores_arrays, m = model.validate_model_lstm(
                clf, train_x, train_y, fit_kwargs=fit_kwargs, compile_kwargs=compile_kwargs,
                plot_name=plot_name, class_weight=class_weight, print_debug=False)
            scores = self.get_mean_scores(scores_arrays)
        else:
            m = clf()
            scores = model.train_model(
                m, train_x, train_y, compile_kwargs=compile_kwargs, fit_kwargs=fit_kwargs, is_lstm=True,
                class_weight=class_weight)
        if not method:
            method = 'mlp'
        components_num = None
        if pca:
            components_num = pca.n_components_
        res = {
            **scores,
            'stock': self.stock,
            'kernel': method,
            'features': feature_name,
            'pca_components': components_num
        }
        m = clf()
        model.train_model(
            m, train_x, train_y, compile_kwargs=compile_kwargs, fit_kwargs=fit_kwargs, is_lstm=True,
            class_weight=class_weight)  # to have a clean fitted model
        test_scores = model.test_model(m, test_x, test_y, is_lstm=True)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        if should_return_model:
            return {**res, **test_scores, 'arch': m.to_json()}, m
        else:
            return {**res, **test_scores, 'arch': m.to_json()}

    def train_lstm(self, clf, feature_name='', should_validate=True, method=None,
                   fit_kwargs=None, compile_kwargs=None, n_steps=None,
                   plot_name=None, class_weight=None, should_return_model=False):
        logger.info('Training %s r=%s s=%s: clf=%s', self.stock, self.r, self.s, clf)

        train_x = self.df[self.feature_columns_dict[feature_name]].values
        test_x = self.df_test[self.feature_columns_dict[feature_name]].values
        train_y = self.df['mid_price_indicator'].values
        test_y = self.df_test['mid_price_indicator'].values

        pca = self.get_pca(feature_name)
        if pca:
            train_x = pca.transform(train_x)
            test_x = pca.transform(test_x)

        if n_steps:
            train_x, train_y = self.split_sequences(train_x, train_y, n_steps=n_steps)
            test_x, test_y = self.split_sequences(test_x, test_y, n_steps=n_steps)
        else:

            train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
            test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        if should_validate:
            scores_arrays, m = model.validate_model_lstm(
                clf, train_x, train_y, fit_kwargs=fit_kwargs, compile_kwargs=compile_kwargs,
                plot_name=plot_name, class_weight=class_weight, print_debug=False)
            scores = self.get_mean_scores(scores_arrays)
        else:
            m = clf()
            scores = model.train_model(
                m, train_x, train_y, compile_kwargs=compile_kwargs, fit_kwargs=fit_kwargs, is_lstm=True,
                class_weight=class_weight)
        if not method:
            method = 'lstm'
        components_num = None
        if pca:
            components_num = pca.n_components_
        res = {
            **scores,
            'stock': self.stock,
            'kernel': method,
            'features': feature_name,
            'pca_components': components_num
        }
        # m = clf()
        # model.train_model(
        #     m, train_x, train_y, compile_kwargs=compile_kwargs, fit_kwargs=fit_kwargs, is_lstm=True,
        #     class_weight=class_weight)
        test_scores = model.test_model(m, test_x, test_y, is_lstm=True)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        if should_return_model:
            return {**res, **test_scores, 'arch': m.to_json()}, m
        else:
            return {**res, **test_scores, 'arch': m.to_json()}

    def train_clf(self, clf, feature_name='', should_validate=True, method=None, class_weight=None):
        logger.info('Training %s r=%s s=%s: clf=%s',
                    self.stock, self.r, self.s, clf)
        train_x = self.df[self.feature_columns_dict[feature_name]]
        pca = self.get_pca(feature_name)
        if pca:
            train_x = pca.transform(train_x)
        if should_validate:
            scores_arrays = model.validate_model(clf, train_x, self.df['mid_price_indicator'],
                                                 class_weight=class_weight)
            scores = self.get_mean_scores(scores_arrays)
        else:
            scores = model.train_model(clf, train_x, self.df['mid_price_indicator'], class_weight=class_weight)
        if not method:
            method = 'logistic'
        components_num = None
        if pca:
            components_num = pca.n_components_
        res = {
            **scores,
            'stock': self.stock,
            'kernel': method,
            'features': feature_name,
            'pca_components': components_num
        }
        test_scores = self.get_score_for_clf(clf, self.df_test, feature_name=feature_name, pca=pca)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        return {**res, **test_scores}

    def train_svm(self, C=np.nan, gamma=np.nan, feature_name='', kernel='rbf', coef0=np.nan, should_validate=True,
                  class_weight=None):
        logger.info('Training %s r=%s s=%s: kernel=%s C=%s gamma=%s coef0=%s',
                    self.stock, self.r, self.s, kernel, C, gamma, coef0)
        if C and gamma and coef0:
            clf = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
        elif C and gamma:
            clf = SVC(kernel=kernel, C=C, gamma=gamma)
        else:
            clf = SVC(kernel=kernel)
        train_x = self.df[self.feature_columns_dict[feature_name]]
        pca = self.get_pca(feature_name)
        if pca:
            train_x = pca.transform(train_x)
        if should_validate:
            scores_arrays = model.validate_model(clf, train_x, self.df['mid_price_indicator'],
                                                 class_weight=class_weight)
            scores = self.get_mean_scores(scores_arrays)
        else:
            scores = model.train_model(clf, train_x, self.df['mid_price_indicator'], class_weight=class_weight)
        components_num = None
        if pca:
            components_num = pca.n_components_
        res = {
            **scores,
            'stock': self.stock,
            'C': C,
            'gamma': gamma,
            'coef0': coef0,
            'kernel': kernel,
            'features': feature_name,
            'pca_components': components_num
        }
        test_scores = self.get_score_for_clf(clf, self.df_test, feature_name=feature_name, pca=pca)
        logger.info('Finished training %s %s', self.stock, {**res, **test_scores})
        return {**res, **test_scores}

    def _read_stock(self):
        gdf_filename = self.gdf_filename_pattern.format(self.stock, self.r, self.s)
        reg_filename = '{}'.format(self.stock)
        logger.debug('Will read %s and %s', gdf_filename, reg_filename)
        d = lob.load_prepared_data(
            gdf_filename, data_dir=self.data_dir, length=self.data_length)
        if len(d) == 2:
            df, df_test = d
        else:
            return pd.DataFrame(), pd.DataFrame()
        df_reg, df_reg_test = lob.load_prepared_data(
            reg_filename, data_dir=self.reg_data_dir, length=self.data_length)
        df['datetime'] = df_reg['Unnamed: 0']
        df['bid_price'] = df_reg['bid_price']
        df['ask_price'] = df_reg['ask_price']
        df['queue_imbalance'] = df_reg['queue_imbalance']
        df['prev_queue_imbalance'] = df['queue_imbalance'].shift()
        df.index = pd.to_datetime(df['datetime'])
        df.dropna(inplace=True)
        df_test['datetime'] = df_reg_test['Unnamed: 0']
        df_test['bid_price'] = df_reg_test['bid_price']
        df_test['ask_price'] = df_reg_test['ask_price']
        df_test['queue_imbalance'] = df_reg_test['queue_imbalance']
        df_test['prev_queue_imbalance'] = df_test['queue_imbalance'].shift()
        df_test.index = pd.to_datetime(df_test['datetime'])
        df_test.dropna(inplace=True)
        return df, df_test

    def features(self, clf) -> pd.DataFrame:
        res = []

        for feature_name, features in self.feature_columns_dict.items():
            logger.debug('Checking feature %s for %s r=%s s=%s', feature_name, self.stock, self.r, self.s)
            train_x = self.df[features]
            n_components = self.get_number_of_pca_components(feature_name)
            if n_components:
                pca = PCA(n_components=n_components)
                pca.fit(train_x)
                train_x = pca.transform(train_x)
            scores = model.validate_model(clf, train_x, self.df['mid_price_indicator'])
            df_score = pd.DataFrame(scores)
            scores = {
                'matthews': np.mean(df_score['matthews'].values),
                'f1': np.mean(df_score['f1'].values),
                'roc_auc': np.mean(df_score['roc_auc'].values),
                'stock': self.stock,
                'features': feature_name
            }
            logger.info('Scores %s r=%s s=%s - %s', self.stock, self.r, self.s, scores)
            res.append(scores)
        return pd.DataFrame(res)
