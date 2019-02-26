import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from lob_data_utils import lob
from sklearn.metrics import roc_auc_score

from lob_data_utils.svm_calculation import lob_svm


class Overview(object):
    def __init__(self, stock, data_length, results_dir=None, data_dir=None, suffix=None):
        self.stock = stock
        self.data_length = data_length
        if results_dir:
            self.results_dir = results_dir
        else:
            self.results_dir = 'results/'
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = '../gaussian_filter/data_gdf/'

        if suffix:
            self.suffix = suffix
        else:
            self.suffix = ''

    @classmethod
    def is_in_results(cls, df_res, keys_values):
        query = ''
        number_of_keys = len(keys_values)
        i = 0
        for k, v in keys_values.items():
            if type(v) == str:
                v = '\'{}\''.format(v)
            query += '{} == {}'.format(k, v)
            i += 1
            if i != number_of_keys:
                query += ' and '
        return np.any(df_res.query(query))

    def get_svm_sigmoid_results(self, df, df_cv, c=None, coef0=None, gamma=None):
        svm = lob_svm.SVMSigmoid(self.stock, df, c=c, coef0=coef0, gamma=gamma,
                                 data_length=self.data_length)
        cv_score = svm.predict(df_cv, 'cv', check=False)
        train_score = svm.predict(df, 'train', check=False)
        return {'svm': 'sigmoid', 'c': c, 'coef0': coef0, 'gamma': gamma,
                'roc_cv_score': cv_score, 'roc_train_score': train_score}

    def get_svm_rbf_results(self, df, df_cv, c=None, gamma=None):
        svm = lob_svm.SVMRbf(self.stock, df, c=c, gamma=gamma, data_length=self.data_length)
        cv_score = svm.predict(df_cv, 'cv', check=False)
        train_score = svm.predict(df, 'train', check=False)
        return {'svm': 'rbf', 'c': c, 'gamma': gamma,
                'roc_cv_score': cv_score, 'roc_train_score': train_score}

    def get_svm_linear_results(self, df, df_cv, c=None):
        svm = lob_svm.SVMLinear(self.stock, df, c=c, data_length=self.data_length)
        cv_score = svm.predict(df_cv, 'cv', check=False)
        train_score = svm.predict(df, 'train', check=False)
        return {'svm': 'linear', 'c': c,
                'roc_cv_score': cv_score, 'roc_train_score': train_score}

    def write_svm_results(self, df, df_cv, gammas=None, cs=None, coef0s=None):
        # TODO: add K!
        try:
            df_svm_res = pd.read_csv('{}/res_svm_{}.csv'.format(self.results_dir, self.stock))
            print('Results read from file')
        except FileNotFoundError:
            print('Results file does not exist yet')
            df_svm_res = pd.DataFrame(
                columns=['svm', 'c', 'gamma', 'coef0', 'roc_cv_score', 'roc_train_score'])
        svm_results = self.perform_svm(df, df_cv, df_svm_res,
                                       gammas=gammas, cs=cs, coef0s=coef0s)
        for svm_result in svm_results:
            df_svm_res = df_svm_res.append(svm_result, ignore_index=True)
        df_svm_res = df_svm_res.sort_values(by='roc_cv_score', ascending=False)
        df_svm_res.to_csv('{}/res_svm_{}.csv'.format(self.results_dir, self.stock))
        return df_svm_res

    def perform_svm(self, df, df_cv, df_res=None, gammas=None, cs=None, coef0s=None,
                    write_temp=True):
        svm_results = []
        for c in cs:
            for g in gammas:
                for coef0 in coef0s:
                    if self.is_in_results(
                            df_res, {'svm': 'sigmoid', 'c': c, 'gamma': g, 'coef0': coef0}):
                        continue
                    svm_results.append(self.get_svm_sigmoid_results(
                        df, df_cv, c=c, coef0=coef0, gamma=g))
                if self.is_in_results(df_res, {'svm': 'rbf', 'c': c, 'gamma': g}):
                    continue
                svm_results.append(self.get_svm_rbf_results(df, df_cv, c=c, gamma=g))
            if self.is_in_results(df_res, {'c': c, 'svm': 'linear'}):
                continue
            svm_results.append(self.get_svm_linear_results(df, df_cv, c=c))
            if write_temp:
                pd.DataFrame(svm_results).to_csv(
                    '{}/new_svm_{}.csv'.format(self.results_dir, self.stock))
        return svm_results

    @classmethod
    def gdf_svm_classification(cls, df, gdf_columns, C=1000, gamma=1):
        clf = SVC(kernel='rbf', C=C, gamma=gamma)
        X = df.loc[:, gdf_columns]
        y = df['mid_price_indicator'].values.reshape(-1, 1)
        y[0] = 0
        clf.fit(X, y)
        return clf

    def write_svm_gdf(self, K=None, Kn=None, rr=None, ss=None):
        results = []

        try:
            df_gdf_res = pd.read_csv(
                '{}/res_gdf_svm_{}_{}.csv'.format(self.results_dir, self.stock, Kn))
            print('Results read from file')
        except FileNotFoundError:
            print('Results file does not exist yet')
            df_gdf_res = pd.DataFrame(
                columns=['svm', 'c', 'gamma', 'roc_cv_score', 'roc_train_score',
                         'K', 'Kn', 'r', 's'])

        for r in rr:
            for s in ss:
                filename = 'gdf_{}_len{}_r{}_s{}_K{}{}'.format(
                    self.stock, self.data_length, r, s, K, self.suffix)
                dfs, dfs_cv, dfs_test = lob.load_prepared_data(
                    filename, data_dir=self.data_dir,
                    cv=True, length=None)  # we don't care about length here

                for C in [1, 10, 100, 1000, 10000]:
                    for gamma in [1, 10, 100, 1000, 10000]:

                        if self.is_in_results(
                                df_gdf_res, {
                                    'c': C, 'gamma': gamma, 'r': r, 's': s,
                                    'K': K, 'Kn': Kn, 'svm': 'rbf'
                                }):
                            continue
                        res = self.perform_gdf_svm(
                            dfs, dfs_cv, C=C, gamma=gamma, r=r, s=s, K=K, Kn=Kn)
                        results.append(res)
                        pd.DataFrame(results).to_csv(
                            '{}/new_res_gdf_svm_{}_{}.csv'.format(self.results_dir, self.stock, Kn))
        for result in results:
            df_gdf_res = df_gdf_res.append(result, ignore_index=True)
        df_gdf_res.to_csv('results/res_gdf_svm_{}_{}.csv'.format(self.stock, Kn))
        return df_gdf_res

    def perform_gdf_svm(self, dfs, dfs_cv, C=None, gamma=None, r=None, s=None, K=None, Kn=None):
        res = {'c': C, 'gamma': gamma, 'r': r, 's': s, 'stock': self.stock, 'K': K,
               'svm': 'rbf', 'Kn': Kn}
        gdf_columns = ['gdf_' + str(i) for i in range(0, Kn)]
        clf = self.gdf_svm_classification(dfs, gdf_columns, C=C, gamma=gamma)
        predictions = clf.predict(dfs.loc[:, gdf_columns])
        try:
            roc_train = roc_auc_score(predictions, dfs['mid_price_indicator'])
            res['roc_train_score'] = roc_train
        except Exception as e:
            print(e, r, s, C, gamma)
        predictions = clf.predict(dfs_cv.loc[:, gdf_columns])
        try:
            roc_cv = roc_auc_score(predictions, dfs_cv['mid_price_indicator'])
            res['roc_cv_score'] = roc_cv
        except Exception as e:
            print(e, r, s, C, gamma)
        return res

    @classmethod
    def gdf_log_classification(cls, df, gdf_columns, C=1000):
        clf = LogisticRegression(C=C)
        X = df.loc[:, gdf_columns]
        y = df['mid_price_indicator'].values.reshape(-1, 1)
        y[0] = 0
        clf.fit(X, y)
        return clf

    def write_gdf_logistic(self, K=None, Kn=None, rr=None, ss=None, Cs=None):
        try:
            df_res = pd.read_csv(
                '{}/res_gdf_log_{}_{}.csv'.format(self.results_dir, self.stock, Kn))
            print('Results read from file')
        except FileNotFoundError:
            print('Results file does not exist yet')
            df_res = pd.DataFrame(columns=['c', 'roc_cv_score', 'roc_train_score', 'K', 'Kn',
                                           'r', 's'])
        results = []
        for r in rr:
            for s in ss:

                filename = 'gdf_{}_len{}_r{}_s{}_K{}{}'.format(
                    self.stock, self.data_length, r, s, K, self.suffix)
                dfs, dfs_cv, dfs_test = lob.load_prepared_data(
                    filename, data_dir=self.data_dir, cv=True,
                    length=None)  # we don't care about length here
                for C in Cs:

                    if self.is_in_results(
                            df_res, {'c': C, 'r': r, 's': s, 'K': K, 'Kn': Kn}):
                        continue

                    res = self.perform_gdf_logistic(dfs, dfs_cv, K=K, Kn=Kn, C=C, r=r, s=s)
                    results.append(res)
                    pd.DataFrame(results).to_csv(
                        'new_res_gdf_log_{}_{}.csv'.format(self.stock, Kn))
        for result in results:
            df_res = df_res.append(result, ignore_index=True)
        df_res.to_csv('{}/res_gdf_log_{}_{}.csv'.format(self.results_dir, self.stock, Kn))
        return df_res

    def perform_gdf_logistic(self, dfs, dfs_cv, K=None, Kn=None, C=None, r=None, s=None):
        gdf_columns = ['gdf_' + str(i) for i in range(0, Kn)]
        res = {'c': C, 'r': r, 's': s, 'stock': self.stock, 'K': K, 'Kn': Kn}
        clf = self.gdf_log_classification(dfs, gdf_columns, C=C)
        predictions = clf.predict(dfs.loc[:, gdf_columns])
        try:
            roc_train = roc_auc_score(predictions, dfs['mid_price_indicator'])
            res['roc_train_score'] = roc_train
        except Exception as e:
            print(e, r, s, C)
        predictions = clf.predict(dfs_cv.loc[:, gdf_columns])
        try:
            roc_cv = roc_auc_score(predictions, dfs_cv['mid_price_indicator'])
            res['roc_cv_score'] = roc_cv
        except Exception as e:
            print(e, r, s, C)
        return res
