import pandas as pd
from sklearn.svm import SVC

from calculation.calculation import SVMCalculationPerformer


class SVMLinear(object):
    def __init__(self, stock, df_train: pd.DataFrame, c=1):
        self.classifier = SVC(kernel='linear', C=c)
        self.df_train = df_train
        self.stock = stock

    def fit(self):
        X = self.df_train['queue_imbalance'].values.reshape(-1, 1)
        y = self.df_train['mid_price_indicator'].values.reshape(-1, 1)

        self.classifier.fit(X, y)

    def predict(self, df_test: pd.DataFrame, data_type):
        cp = SVMCalculationPerformer(self.stock, self.classifier)
        X = df_test['queue_imbalance'].values.reshape(-1, 1)
        y = df_test['mid_price_indicator'].values.reshape(-1, 1)
        response = cp.execute(X, y, data_type, name='svm-linear')
        return response
