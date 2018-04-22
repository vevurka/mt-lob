import pandas as pd

import requests
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


class SVMCalculationPerformer(object):
    def __init__(self, stock, classification: SVC):
        self.classification = classification
        self.stock = stock

    def prepare_data(self, roc_score, data_type, name=''):
        gamma = self.classification.gamma
        if self.classification.gamma == 'auto':
            gamma = -1
        return {
                'algorithm': {
                    'name': name,
                    'svm': {
                        'kernel': self.classification.kernel,
                        'c': self.classification.C,
                        'gamma': gamma,
                        'coef0': self.classification.coef0
                    },
                },
                'roc_auc_score': roc_score,
                'data_type': data_type,
                'stock': self.stock
            }

    def execute(self, df: pd.DataFrame, y_true, data_type, name=''):
        prediction = self.classification.predict(df)
        score = roc_auc_score(y_true, prediction)
        data = self.prepare_data(score, data_type, name=name)
        r = requests.post('http://localhost:8000/result/', json=data)
        print(r.json())
        return r.json()
