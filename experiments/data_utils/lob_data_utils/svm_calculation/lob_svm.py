import pandas as pd
from lob_data_utils.svm_calculation.calculation import SVMCalculationPerformer

from sklearn.svm import SVC
import requests


class SVM(object):
    def __init__(self, stock, df_train: pd.DataFrame, data_length=None):
        self.df_train = df_train
        self.stock = stock
        if not data_length:
            data_length = 0
        self.data_length = data_length
        self.classifier = None

    def predict(self, df_test: pd.DataFrame, data_type, check=True):
        params = 'stock={}&data-length={}&data-type={}&algorithm=svm&'
        params_svm = 'kernel={}&gamma={}&coef0={}&c={}'
        gamma = self.classifier.gamma
        if gamma == 'auto':
            gamma = -1
        params = params.format(self.stock, self.data_length, data_type)
        params_svm = params_svm.format(self.classifier.kernel, gamma,
                                       self.classifier.coef0, self.classifier.C)
        if check:
            try:
                r = requests.get('http://localhost:8000/get-result?' + params + params_svm)
                print(r.status_code)
                if r.status_code == 200:
                    if r.json():
                        return r
            except requests.exceptions.ConnectionError:
                print('Connection error')

        X = self.df_train['queue_imbalance'].values.reshape(-1, 1)
        y = self.df_train['mid_price_indicator'].values.reshape(-1, 1)

        self.classifier.fit(X, y)
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

