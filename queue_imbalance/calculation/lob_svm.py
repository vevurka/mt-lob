import pandas as pd
from sklearn.svm import SVC

from calculation.calculation import SVMCalculationPerformer


class SVMLinear(object):
    def __init__(self, stock, df_train: pd.DataFrame, c=1, data_length=None):
        self.classifier = SVC(kernel='linear', C=c)
        self.df_train = df_train
        self.stock = stock
        self.data_length = data_length

    def fit(self):
        X = self.df_train['queue_imbalance'].values.reshape(-1, 1)
        y = self.df_train['mid_price_indicator'].values.reshape(-1, 1)

        self.classifier.fit(X, y)

    def predict(self, df_test: pd.DataFrame, data_type):
        cp = SVMCalculationPerformer(self.stock, self.classifier, self.data_length)
        X = df_test['queue_imbalance'].values.reshape(-1, 1)
        y = df_test['mid_price_indicator'].values.reshape(-1, 1)
        response = cp.execute(X, y, data_type, name='svm-linear')
        return response


class SVMRbf(object):
    def __init__(self, stock, df_train: pd.DataFrame, c=1, gamma='auto', data_length=None):
        self.classifier = SVC(kernel='rbf', C=c, gamma=gamma)
        self.df_train = df_train
        self.stock = stock
        self.data_length = data_length

    def fit(self):
        X = self.df_train['queue_imbalance'].values.reshape(-1, 1)
        y = self.df_train['mid_price_indicator'].values.reshape(-1, 1)

        self.classifier.fit(X, y)

    def predict(self, df_test: pd.DataFrame, data_type):
        cp = SVMCalculationPerformer(self.stock, self.classifier, self.data_length)
        X = df_test['queue_imbalance'].values.reshape(-1, 1)
        y = df_test['mid_price_indicator'].values.reshape(-1, 1)
        response = cp.execute(X, y, data_type, name='svm-linear')
        return response
