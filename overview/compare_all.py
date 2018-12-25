import unittest
from typing import Optional

from lob_data_utils import lob, model

# the goal is to compare all algorithms on test set now.
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd

import numpy as np


class Test(unittest.TestCase):
    def test_get_components(self):
        n = SvmGdfResults.get_number_of_pca_components('no_pca')
        self.assertIsNone(n)
        n = SvmGdfResults.get_number_of_pca_components('pca_gdf_que3')
        self.assertEqual(3, n)


class SvmGdfResults(object):

    def __init__(self, stocks, r=1.0, s=1.0, data_length=10000, gdf_filename_pattern='',
                 data_dir='../gaussian_filter/data_gdf'):
        self.stocks = stocks
        self.r = r
        self.s = s
        self.data_length = data_length
        self.gdf_filename_pattern = gdf_filename_pattern
        self.data_dir = data_dir
        self.dfs, self.dfs_test= self._read_stocks()

    feature_columns_dict = {
        'gdf_24-26_que_prev': ['gdf_24', 'gdf_25', 'queue_imbalance', 'prev_queue_imbalance'],
        'que': ['queue_imbalance'],
        'que_prev': ['queue_imbalance', 'prev_queue_imbalance'],
        'gdf_24-26_que': ['gdf_24', 'gdf_25', 'queue_imbalance'],
        'gdf_23-27_que': ['gdf_23', 'gdf_24', 'gdf_25', 'gdf_26', 'queue_imbalance'],
        'gdf_23-27': ['gdf_23', 'gdf_24', 'gdf_25', 'gdf_26'],
        'gdf_24_26': ['gdf_24', 'gdf_25'],
        'pca_gdf': ['gdf_{}'.format(i) for i in range(0, 50)],
        'pca_gdf_que': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],
        'pca_gdf_que2': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],  # TODO: hm
        'pca_gdf_que3': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'], # TODO: hm
        'pca_gdf_que4': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],  # TODO: hm
        'pca_gdf_que5': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],  # TODO: hm
        'pca_gdf_que6': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],  # TODO: hm
        'pca_gdf_que7': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],  # TODO: hm
        'pca_gdf_que8': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance'],  # TODO: hm
        'pca_gdf_que_prev': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance', 'prev_queue_imbalance'],
        'pca_gdf_que_prev3': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance', 'prev_queue_imbalance'],
        'pca_gdf_que_prev4': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance', 'prev_queue_imbalance'],
        'pca_gdf_que_prev5': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance', 'prev_queue_imbalance'],
        'pca_gdf_que_prev7': ['gdf_{}'.format(i) for i in range(0, 50)] + ['queue_imbalance', 'prev_queue_imbalance'],

        'gdf_20_30': ['gdf_{}'.format(i) for i in range(20, 30)],
        'gdf_20_30_que': ['gdf_{}'.format(i) for i in range(20, 30)] + ['queue_imbalance'],
        'gdf_20_30_que_prev': ['gdf_{}'.format(i) for i in range(20, 30)] + ['queue_imbalance', 'prev_queue_imbalance']
    }

    def get_score_for_clf(self, clf, df_test, feature_name, pca=None):
        x_test = df_test[self.feature_columns_dict[feature_name]]
        if pca:
            x_test = pca.transform(x_test)
        y_test = df_test['mid_price_indicator'].values
        return model.test_model(clf, x_test, y_test)

    @staticmethod
    def get_number_of_pca_components(feature_name: str) -> Optional[int]:
        if 'pca_gdf_que_prev' in feature_name:
            return int(feature_name.replace('pca_gdf_que_prev', ''))
        if 'pca_gdf_que' in feature_name:
            return int(feature_name.replace('pca_gdf_que', ''))
        return None

    @staticmethod
    def get_mean_scores(scores: dict) -> dict:
        mean_scores = {}
        for k, v in scores.items():
            mean_scores[k] = np.mean(v)
        return  mean_scores

    def train_svm(self, stock, C, gamma, feature_name, kernel='rbf', coef0=np.nan):
        if coef0:
            clf = SVC(kernel=kernel, C=C, gamma=gamma, coef0=coef0)
        else:
            clf = SVC(kernel=kernel, C=C, gamma=gamma)
        train_x = self.dfs[stock][self.feature_columns_dict[feature_name]]
        n_components = self.get_number_of_pca_components(feature_name)
        pca = None
        if n_components:
            pca = PCA(n_components=n_components)
            pca.fit(train_x)
            train_x = pca.transform(train_x)

        scores = model.validate_model(clf, train_x, self.dfs[stock]['mid_price_indicator'])
        res = {
            **self.get_mean_scores(scores),
            'stock': stock,
            'C': C,
            'gamma': gamma,
            'coef0': coef0,
            'kernel': kernel,
            'features': feature_name
        }
        test_scores = self.get_score_for_clf(clf, self.dfs_test[stock], feature_name=feature_name, pca=pca)
        return {**res, **test_scores}

    def _read_stocks(self):
        dfs = {}
        dfs_test = {}
        dfs_reg = {}
        dfs_reg_test = {}

        for stock in self.stocks:
            gdf_filename = self.gdf_filename_pattern.format(stock, self.r, self.s)
            reg_filename = '{}'.format(stock)
            dfs[stock], dfs_test[stock] = lob.load_prepared_data(
                gdf_filename, data_dir=self.data_dir, cv=False, length=self.data_length)
            dfs_reg[stock], dfs_reg_test[stock] = lob.load_prepared_data(
                reg_filename, data_dir='../gaussian_filter/data', cv=False, length=self.data_length)
        for stock in self.stocks:
            dfs[stock]['queue_imbalance'] = dfs_reg[stock]['queue_imbalance']
            dfs[stock]['prev_queue_imbalance'] = dfs[stock]['queue_imbalance'].shift()
            dfs[stock].dropna(inplace=True)
            dfs_test[stock]['queue_imbalance'] = dfs_reg_test[stock]['queue_imbalance']
            dfs_test[stock]['prev_queue_imbalance'] = dfs_test[stock]['queue_imbalance'].shift()
            dfs_test[stock].dropna(inplace=True)
        return dfs, dfs_test


class GdfRegularResults(object):
    @staticmethod
    def get_regular_results(data_length):
        r = 1.0
        s = 1.0
        gdf_res = SvmGdfResults(
            ['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
            gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(stock='9265', C=1000.0, gamma=0.1, feature_name='pca_gdf_que3')
        res2 = gdf_res.train_svm(stock='9061', C=1.0, gamma=0.1, feature_name='gdf_24-26_que_prev')
        res3 = gdf_res.train_svm(stock='9064', C=1.0, gamma=10.0, feature_name='pca_gdf_que7')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results

    @staticmethod
    def get_small_r_results(data_length):
        r = 0.1
        s = 1.0
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(stock='9265', C=100.0, gamma=0.1, feature_name='pca_gdf_que3')
        res2 = gdf_res.train_svm(stock='9061', C=0.1, gamma=1.0, feature_name='gdf_24-26_que_prev')
        res3 = gdf_res.train_svm(stock='9064', C=100.0, gamma=1.0, feature_name='pca_gdf_que7')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results

    @staticmethod
    def get_small_s_results(data_length):
        r = 1.0
        s = 0.1
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(stock='9265', C=100.0, gamma=1.0, feature_name='pca_gdf_que8')
        res2 = gdf_res.train_svm(stock='9061', C=1.0, gamma=0.1, feature_name='gdf_24-26_que_prev')
        res3 = gdf_res.train_svm(stock='9064', C=10.0, gamma=0.1, feature_name='pca_gdf_que7')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results

    @staticmethod
    def get_small_rs_results(data_length):
        r = 0.1
        s = 0.1
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(stock='9265', C=1.0, gamma=1.0, feature_name='pca_gdf_que7')
        res2 = gdf_res.train_svm(stock='9061', C=1.0, gamma=0.1, feature_name='pca_gdf_que_prev7')
        res3 = gdf_res.train_svm(stock='9064', C=1000.0, gamma=0.1, feature_name='pca_gdf_que_prev7')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results


class GdfScaledResults(object):
    @staticmethod
    def get_regular_results(data_length):
        r = 1.0
        s = 1.0

        data_dir = '../gaussian_filter/data_gdf_feature_scaling'
        gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50_feature_scaling'
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                data_dir=data_dir, gdf_filename_pattern=gdf_filename_pattern)

        res1 = gdf_res.train_svm(stock='9265', C=100.0, gamma=0.1, feature_name='gdf_24-26_que')
        res2 = gdf_res.train_svm(stock='9061', C=0.1, gamma=1.0, feature_name='pca_gdf_que_prev3')
        res3 = gdf_res.train_svm(stock='9064', C=1.0, gamma=1.0, feature_name='pca_gdf_que4')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'scaled'
        return df_results

    @staticmethod
    def get_small_s_results(data_length):
        r = 0.1
        s = 1.0
        data_dir = '../gaussian_filter/data_gdf_feature_scaling'
        gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50_feature_scaling'
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                data_dir=data_dir, gdf_filename_pattern=gdf_filename_pattern)

        res1 = gdf_res.train_svm(stock='9265', C=1000.0, gamma=0.1, feature_name='pca_gdf_que_prev4')
        res2 = gdf_res.train_svm(stock='9061', C=10.0, gamma=0.1, feature_name='gdf_24-26_que')
        res3 = gdf_res.train_svm(stock='9064', C=0.1, gamma=0.1, feature_name='pca_gdf_que3')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'scaled'
        return df_results

    @staticmethod
    def get_small_r_results(data_length):
        r = 1.0
        s = 0.1
        data_dir = '../gaussian_filter/data_gdf_feature_scaling'
        gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50_feature_scaling'
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                data_dir=data_dir, gdf_filename_pattern=gdf_filename_pattern)

        res1 = gdf_res.train_svm(stock='9265', C=1.0, gamma=1.0, feature_name='pca_gdf_que2')
        res2 = gdf_res.train_svm(stock='9061', C=10.0, gamma=0.1, feature_name='pca_gdf_que8')
        res3 = gdf_res.train_svm(stock='9064', C=0.1, gamma=1.0, feature_name='pca_gdf_que3')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'scaled'
        return df_results

    @staticmethod
    def get_small_rs_results(data_length):
        r = 0.1
        s = 0.1
        data_dir = '../gaussian_filter/data_gdf_feature_scaling'
        gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50_feature_scaling'
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                data_dir=data_dir, gdf_filename_pattern=gdf_filename_pattern)

        res1 = gdf_res.train_svm(stock='9265', C=1000.0, gamma=0.1, feature_name='pca_gdf_que_prev3')
        res2 = gdf_res.train_svm(stock='9061', C=1.0, gamma=0.1, feature_name='gdf_24-26_que_prev')
        res3 = gdf_res.train_svm(stock='9064', C=0.1, gamma=0.1, feature_name='pca_gdf_que3')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'scaled'
        return df_results


class GdfRegularSigmoidResults(object):
    @staticmethod
    def get_regular_results(data_length):
        r = 1.0
        s = 1.0
        gdf_res = SvmGdfResults(
            ['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
            gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(
            stock='9265', C=10.0, gamma=0.01, coef0=0.01, kernel='sigmoid', feature_name='pca_gdf_que6')
        res2 = gdf_res.train_svm(
            stock='9061', C=0.1, gamma=1.0, coef0=0.1, kernel='sigmoid', feature_name='pca_gdf_que5')
        res3 = gdf_res.train_svm(
            stock='9064', C=10.0, gamma=0.1, coef0=0.01, kernel='sigmoid', feature_name='pca_gdf_que_prev4')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results

    @staticmethod
    def get_small_r_results(data_length):
        r = 0.1
        s = 1.0
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(
            stock='9265', C=1000.0, gamma=0.001, coef0=0.01, kernel='sigmoid', feature_name='pca_gdf_que_prev4')
        res2 = gdf_res.train_svm(
            stock='9061', C=10.0, gamma=0.01, coef0=0.01, kernel='sigmoid', feature_name='pca_gdf_que5')
        res3 = gdf_res.train_svm(
            stock='9064', C=100.0, gamma=0.01, coef0=1.0, kernel='sigmoid', feature_name='pca_gdf_que_prev4')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results

    @staticmethod
    def get_small_s_results(data_length):
        r = 1.0
        s = 0.1
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(
            stock='9265', C=1000.0, gamma=0.001, coef0=1.0, kernel='sigmoid', feature_name='pca_gdf_que3')
        res2 = gdf_res.train_svm(
            stock='9061', C=1000.0, gamma=0.001, coef0=0.01, kernel='sigmoid', feature_name='pca_gdf_que5')
        res3 = gdf_res.train_svm(
            stock='9064', C=100.0, gamma=0.01, coef0=1.0, kernel='sigmoid', feature_name='pca_gdf_que_prev5')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results

    @staticmethod
    def get_small_rs_results(data_length):
        r = 0.1
        s = 0.1
        gdf_res = SvmGdfResults(['9061', '9064', '9265'], data_length=data_length, r=r, s=s,
                                gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

        res1 = gdf_res.train_svm(
            stock='9265', C=100.0, gamma=0.01, coef0=0.1, kernel='sigmoid', feature_name='pca_gdf_que3')
        res2 = gdf_res.train_svm(
            stock='9061', C=1000.0, gamma=0.001, coef0=0.1, kernel='sigmoid', feature_name='pca_gdf_que4')
        res3 = gdf_res.train_svm(
            stock='9064', C=10.0, gamma=0.1, coef0=0.01, kernel='sigmoid', feature_name='pca_gdf_que_prev4')
        df_results = pd.DataFrame([res1, res2, res3])
        df_results['r'] = [r] * len(df_results)
        df_results['s'] = [s] * len(df_results)
        df_results['info'] = 'regular'
        return df_results


def get_pca(pca: PCA, df):
    return pd.DataFrame(
        pca.components_, columns=df.columns, index=['PC_{}'.format(i) for i in range(len(pca.n_components))])


def main():
    data_length = 15000
    df_all_results = pd.DataFrame()
    df_all_results = df_all_results.append(GdfRegularSigmoidResults.get_regular_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularSigmoidResults.get_small_r_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularSigmoidResults.get_small_s_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularSigmoidResults.get_small_rs_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularResults.get_regular_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularResults.get_small_r_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularResults.get_small_s_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfRegularResults.get_small_rs_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfScaledResults.get_regular_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfScaledResults.get_small_r_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfScaledResults.get_small_s_results(data_length), ignore_index=True)
    df_all_results = df_all_results.append(GdfScaledResults.get_small_rs_results(data_length), ignore_index=True)
    df_all_results.to_csv('gdf_svm_results.csv')


if __name__ == '__main__':
    main()