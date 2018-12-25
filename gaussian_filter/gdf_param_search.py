import os

from lob_data_utils import lob, model
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np


def svm_classification(df, gdf_columns, C=1000, gamma=1):
    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    X = df.loc[:, gdf_columns]
    y = df['mid_price_indicator'].values.reshape(-1, 1)
    y[0] = 0
    clf.fit(X, y)
    return clf


def main(stock):
    """
    This gets gdf_data
    :return:
    """
    K = 50
    length = 15000
    rr = [0.01, 0.05, 0.1, 0.5, 1.0]
    ss = [0.01, 0.05, 0.1, 0.5, 1.0]
    for r in rr:
        for s in ss:
            # TODO: if file exists
            if os.path.exists('res_{}_len{}_r{}_s{}_K{}.csv'.format(stock, length, r, s, K)):
                print('Exists ', 'res_{}_len{}_r{}_s{}_K{}.csv'.format(stock, length, r, s, K))
                continue
            else:
                print('Will create ', 'res_{}_len{}_r{}_s{}_K{}.csv'.format(stock, length, r, s, K))
            filename = 'gdf_{}_len{}_r{}_s{}_K{}'.format(stock, length, r, s, K)
            dfs, dfs_cv, dfs_test = lob.load_prepared_data(
                filename, data_dir='data_gdf_/', cv=True, length=length)
            gdf_columns = ['gdf_' + str(i) for i in range(0, 50)]

            results = []
            for C in [1, 10, 100, 1000, 10000]:
                for gamma in [1, 10, 100, 1000, 10000]:
                    res = {}
                    res['C'] = C
                    res['gamma'] = gamma
                    res['r'] = r
                    res['s'] = s
                    res['stock'] = stock
                    res['K'] = K

                    print('********************************************')
                    print('C', C, 'gamma', gamma)
                    lob.mo
                    clf = svm_classification(dfs, gdf_columns, C=C, gamma=gamma)
                    predictions = clf.predict(dfs.loc[:, gdf_columns])
                    try:
                        roc_train = roc_auc_score(predictions, dfs['mid_price_indicator'])
                        res['roc_train'] = roc_train
                        print('train', s, roc_train)
                    except Exception as e:
                        print(e)
                        pd.DataFrame(results).to_csv(
                            'res_{}_len{}_r{}_s{}_K{}.csv_partial'.format(stock, length, r, s, K))
                    predictions = clf.predict(dfs_cv.loc[:, gdf_columns])
                    try:
                        roc_cv = roc_auc_score(predictions, dfs_cv['mid_price_indicator'])
                        res['roc_cv'] = roc_cv
                        print('test ', s, roc_cv)
                    except Exception as e:
                        print(e)
                        pd.DataFrame(results).to_csv(
                            'res_{}_len{}_r{}_s{}_K{}.csv_partial'.format(stock, length, r, s, K))
                    results.append(res)
            pd.DataFrame(results).to_csv(
                'res_{}_len{}_r{}_s{}_K{}.csv'.format(stock, length, r, s, K))


if __name__ == "__main__":
    main('9061')
    main('9064')
    main('9265')