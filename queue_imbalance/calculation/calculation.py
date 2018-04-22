import pandas as pd

import requests
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


class SVMCalculationPerformer(object):
    def __init__(self, classification: SVC):
        self.classification = classification

    def prepare_data(self, roc_score, name=''):
        return {
                'algorithm': {
                    'name': name,
                    'svm': {
                        'kernel': self.classification.kernel,
                        'c': self.classification.C,
                        'gamma': self.classification.gamma,
                        'coef0': self.classification.coef0
                    },
                },
                'roc_auc_score': roc_score
            }

    def execute(self, df: pd.DataFrame, y_true, name=''):
        prediction = self.classification.predict(df)
        score =  roc_auc_score(y_true, prediction)
        data = self.prepare_data(score, name)
        r = requests.post('http://localhost:8000/result/', json=data)
        print(r.json())
        return r.json()
