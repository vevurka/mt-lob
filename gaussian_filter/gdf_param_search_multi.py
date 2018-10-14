import os

from lob_data_utils import lob
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import pandas as pd
from lob_data_utils import roc_results


def svm_classification(df, gdf_columns, C=1000, gamma=1):
    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    X = df.loc[:, gdf_columns]
    y = df['mid_price_indicator'].values.reshape(-1, 1)
    y[0] = 0
    clf.fit(X, y)
    return clf


def main(stock, stocks):
    """
    This gets gdf_data
    :return:
    """
    K = 50
    length = 10000
    rr = [0.01, 0.03,  0.05,  0.07, 0.09,]
    ss = [0.1] #, 0.2, 0.3, 0.4, 0.5]
    for r in rr:
        for s in ss:

            filename = 'gdf_{}_len{}_r{}_s{}_K{}'.format(stock, length, r, s, K)
            if not os.path.exists(os.path.join('data_gdf',  filename + '.csv')):
                print('GDF file ', filename, 'does not exist')
                continue
            try:
                dfs, dfs_cv, dfs_test = lob.load_prepared_data(
                    filename, data_dir='data_gdf/', cv=True, length=length)
                if dfs is None:
                    continue
            except Exception as e:
                print(filename, 'not read', e)
                continue
            K1 = 20
            K2 = 40
            gdf_columns = ['gdf_' + str(i) for i in range(0, 50)][K1:K2]

            results = []
            f = 'data_res/res_{}_len{}_r{}_s{}_K{}-{}.csv'.format(stock, length, r, s, K1, K2)
            if os.path.exists(f):
                # print('Skipping, file', f, 'exists')
                continue
            Cs = [1, 10, 100, 1000, 10000]
            gammas = [1, 10, 100, 1000, 10000]
            for C in Cs: #[1, 1000]:
                for gamma in gammas: #[1, 100]:
                    res = {}
                    res['C'] = C
                    res['gamma'] = gamma
                    res['r'] = r
                    res['s'] = s
                    res['stock'] = stock
                    res['K'] = K

                    print(stock, 'C', C, 'gamma', gamma)
                    clf = svm_classification(dfs, gdf_columns, C=C, gamma=gamma)
                    predictions = clf.predict(dfs.loc[:, gdf_columns])
                    try:
                        roc_train = roc_auc_score(predictions, dfs['mid_price_indicator'])
                        res['roc_train'] = roc_train
                        print(stock, 'train', s, roc_train)
                    except Exception as e:
                        print(e)
                    predictions = clf.predict(dfs_cv.loc[:, gdf_columns])
                    try:
                        roc_cv = roc_auc_score(predictions, dfs_cv['mid_price_indicator'])
                        res['roc_cv'] = roc_cv
                        print(stock, 'test ', s, roc_cv)
                    except Exception as e:
                        print(e)
                    results.append(res)

            print(stock, 'writing', f)
            pd.DataFrame(results).to_csv(f)
    print('********************************************', stocks.index(stock), len(stocks))
    print('calculated stock', stock, stocks.index(stock))
    return stock


if __name__ == "__main__":
    from multiprocessing import Pool
    stocks = set(roc_results.results.keys())
    files_list = os.listdir('data_gdf')
    stocks_to_calculate = []
    for s in stocks:
        n = 0
        for l in files_list:
            if s in l:
                n += 1
        if n == 51:
            stocks_to_calculate.append(s)
    print(stocks_to_calculate)

    pool = Pool(processes=3)
    res = [pool.apply_async(main, [s, list(stocks_to_calculate)]) for s in stocks_to_calculate]
    print([r.get() for r in res])
    print(stocks_to_calculate)

# calculated stocks
# ['9270', '12456', '9761', '2651', '9069', '9094', '2368', '11399', '2094', '1472', '9926', '2645', '1221', '2748', '2822', '10166', '9034', '12098', '1865', '2051', '1243', '12327', '4060', '1956']
