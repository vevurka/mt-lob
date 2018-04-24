import pandas as pd
from sklearn.svm import SVC

from calculation.calculation import SVMCalculationPerformer


class SVM(object):
    def __init__(self, stock, df_train: pd.DataFrame, data_length=None):
        self.df_train = df_train
        self.stock = stock
        self.data_length = data_length
        self.classifier = None

    def fit(self):
        X = self.df_train['queue_imbalance'].values.reshape(-1, 1)
        y = self.df_train['mid_price_indicator'].values.reshape(-1, 1)

        self.classifier.fit(X, y)

    def predict(self, df_test: pd.DataFrame, data_type):
        cp = SVMCalculationPerformer(self.stock, self.classifier, self.data_length)
        X = df_test['queue_imbalance'].values.reshape(-1, 1)
        y = df_test['mid_price_indicator'].values.reshape(-1, 1)
        response = cp.execute(X, y, data_type, name='svm-sigmoid')
        return response


class SVMLinear(SVM):
    def __init__(self, stock, df_train: pd.DataFrame, c=1, data_length=None):
        super().__init__(stock, df_train, data_length)
        self.classifier = SVC(kernel='linear', C=c)


class SVMRbf(SVM):
    def __init__(self, stock, df_train: pd.DataFrame, c=1, gamma='auto', data_length=None):
        super().__init__(stock, df_train, data_length)
        self.classifier = SVC(kernel='rbf', C=c, gamma=gamma)


class SVMSigmoid(SVM):
    def __init__(self, stock, df_train: pd.DataFrame, c=1, coef0=0, gamma='auto', data_length=None):
        super().__init__(stock, df_train, data_length)
        self.classifier = SVC(kernel='sigmoid', C=c, coef0=coef0, gamma=gamma)

