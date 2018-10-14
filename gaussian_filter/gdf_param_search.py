from lob_data_utils import lob
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


def main():
    """
    This gets gdf_data
    :return:
    """
   # for r in np.arange(0.01, 0.11, step=0.01):
        # for s in np.arange(0.1, 0.6, step=0.1):
    r = 0.01
    #s = 0.1
    K = 50
    stock = '13003'
    length = 10000
    rr = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    ss = [0.1, 0.2, 0.3, 0.4, 0.5]
    for r in rr:
        for s in ss:

            filename = 'gdf_{}_len{}_r{}_s{}_K{}'.format(stock, length, r, s, K)
            dfs, dfs_cv, dfs_test = lob.load_prepared_data(
                filename, data_dir='data_gdf/', cv=True, length=length)
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
                    clf = svm_classification(dfs, gdf_columns, C=C, gamma=gamma)
                    predictions = clf.predict(dfs.loc[:, gdf_columns])
                    try:
                        roc_train = roc_auc_score(predictions, dfs['mid_price_indicator'])
                        res['roc_train'] = roc_train
                        print('train', s, roc_train)
                    except Exception as e:
                        print(e)
                    predictions = clf.predict(dfs_cv.loc[:, gdf_columns])
                    try:
                        roc_cv = roc_auc_score(predictions, dfs_cv['mid_price_indicator'])
                        res['roc_cv'] = roc_cv
                        print('test ', s, roc_cv)
                    except Exception as e:
                        print(e)
                    results.append(res)
            pd.DataFrame(results).to_csv(
                'res_{}_len{}_r{}_s{}_K{}.csv'.format(stock, length, r, s, K))


if __name__ == "__main__":
    main()

#
# ********************************************
# C 1 gamma 1
# Only one class present in y_true. ROC AUC score is not defined in that case.
# Only one class present in y_true. ROC AUC score is not defined in that case.
# ********************************************
# C 1 gamma 10
# train gdf_9061_len10000_r0.01_s0.1_K50 0.547906769491
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.533386633188
# ********************************************
# C 1 gamma 100
# train gdf_9061_len10000_r0.01_s0.1_K50 0.58333231724
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.538570427341
# ********************************************
# C 1 gamma 1000
# train gdf_9061_len10000_r0.01_s0.1_K50 0.889939452439
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.532651410079
# ********************************************
# C 1 gamma 10000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.742621310655
# ********************************************
# C 10 gamma 1
# train gdf_9061_len10000_r0.01_s0.1_K50 0.546293637307
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.534552755347
# ********************************************
# C 10 gamma 10
# train gdf_9061_len10000_r0.01_s0.1_K50 0.55727743962
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.534863670467
# ********************************************
# C 10 gamma 100
# train gdf_9061_len10000_r0.01_s0.1_K50 0.671572415567
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.53094629156
# ********************************************
# C 10 gamma 1000
# train gdf_9061_len10000_r0.01_s0.1_K50 0.999503903839
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.52431843321
# ********************************************
# C 10 gamma 10000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.742621310655
# ********************************************
# C 100 gamma 1
# train gdf_9061_len10000_r0.01_s0.1_K50 0.553723146128
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.529322590459
# ********************************************
# C 100 gamma 10
# train gdf_9061_len10000_r0.01_s0.1_K50 0.583958475211
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.536727429619
# ********************************************
# C 100 gamma 100
# train gdf_9061_len10000_r0.01_s0.1_K50 0.804930935438
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.534893027053
# ********************************************
# C 100 gamma 1000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.52061685132
# ********************************************
# C 100 gamma 10000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.742621310655
# ********************************************
# C 1000 gamma 1
# train gdf_9061_len10000_r0.01_s0.1_K50 0.55907465886
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.540626578169
# ********************************************
# C 1000 gamma 10
# train gdf_9061_len10000_r0.01_s0.1_K50 0.635509933991
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.528707735833
# ********************************************
# C 1000 gamma 100
# train gdf_9061_len10000_r0.01_s0.1_K50 0.954284962462
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.523163501156
# ********************************************
# C 1000 gamma 1000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.52061685132
# ********************************************
# C 1000 gamma 10000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.742621310655
# ********************************************
# C 10000 gamma 1
# train gdf_9061_len10000_r0.01_s0.1_K50 0.583176668872
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.540493801794
# ********************************************
# C 10000 gamma 10
# train gdf_9061_len10000_r0.01_s0.1_K50 0.712716466067
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.532572239422
# ********************************************
# C 10000 gamma 100
# train gdf_9061_len10000_r0.01_s0.1_K50 0.999837451235
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.510332909006
# ********************************************
# C 10000 gamma 1000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.52061685132
# ********************************************
# C 10000 gamma 10000
# train gdf_9061_len10000_r0.01_s0.1_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.1_K50 0.742621310655


# ********************************************
# C 1 gamma 1
# Only one class present in y_true. ROC AUC score is not defined in that case.
# Only one class present in y_true. ROC AUC score is not defined in that case.
# ********************************************
# C 1 gamma 10
# train gdf_9061_len10000_r0.01_s0.2_K50 0.54267988008
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.528886393794
# ********************************************
# C 1 gamma 100
# train gdf_9061_len10000_r0.01_s0.2_K50 0.584479576039
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.530686250029
# ********************************************
# C 1 gamma 1000
# train gdf_9061_len10000_r0.01_s0.2_K50 0.864135202779
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.523805961154
# ********************************************
# C 1 gamma 10000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# Only one class present in y_true. ROC AUC score is not defined in that case.
# ********************************************
# C 10 gamma 1
# train gdf_9061_len10000_r0.01_s0.2_K50 0.539667828871
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.530864197531
# ********************************************
# C 10 gamma 10
# train gdf_9061_len10000_r0.01_s0.2_K50 0.558066250617
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.534123563218
# ********************************************
# C 10 gamma 100
# train gdf_9061_len10000_r0.01_s0.2_K50 0.650761346741
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.520098202526
# ********************************************
# C 10 gamma 1000
# train gdf_9061_len10000_r0.01_s0.2_K50 0.999162045707
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.517724701446
# ********************************************
# C 10 gamma 10000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.492492492492
# ********************************************
# C 100 gamma 1
# train gdf_9061_len10000_r0.01_s0.2_K50 0.550664136622
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.538116194307
# ********************************************
# C 100 gamma 10
# train gdf_9061_len10000_r0.01_s0.2_K50 0.580576529257
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.540117766774
# ********************************************
# C 100 gamma 100
# train gdf_9061_len10000_r0.01_s0.2_K50 0.775624959807
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.526215903242
# ********************************************
# C 100 gamma 1000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.515262859236
# ********************************************
# C 100 gamma 10000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.492492492492
# ********************************************
# C 1000 gamma 1
# train gdf_9061_len10000_r0.01_s0.2_K50 0.564292117528
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.536204268293
# ********************************************
# C 1000 gamma 10
# train gdf_9061_len10000_r0.01_s0.2_K50 0.620789499567
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.519509545372
# ********************************************
# C 1000 gamma 100
# train gdf_9061_len10000_r0.01_s0.2_K50 0.936603112488
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.503180457986
# ********************************************
# C 1000 gamma 1000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.515262859236
# ********************************************
# C 1000 gamma 10000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.492492492492
# ********************************************
# C 10000 gamma 1
# train gdf_9061_len10000_r0.01_s0.2_K50 0.58021538784
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.535790843669
# ********************************************
# C 10000 gamma 10
# train gdf_9061_len10000_r0.01_s0.2_K50 0.68598486018
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.510301917585
# ********************************************
# C 10000 gamma 100
# train gdf_9061_len10000_r0.01_s0.2_K50 0.999675008125
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.505201646819
# ********************************************
# C 10000 gamma 1000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.515262859236
# ********************************************
# C 10000 gamma 10000
# train gdf_9061_len10000_r0.01_s0.2_K50 1.0
# test  gdf_9061_len10000_r0.01_s0.2_K50 0.492492492492
