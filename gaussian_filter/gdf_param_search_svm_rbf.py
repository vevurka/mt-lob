import os
from multiprocessing.pool import Pool

import pandas as pd
from lob_data_utils import lob, model
from sklearn.svm import SVC


def svm_classification(df, gdf_columns, C=1000, gamma=1) -> dict:
    clf = SVC(kernel='rbf', C=C, gamma=gamma)
    X = df.loc[:, gdf_columns]
    y = df['mid_price_indicator'].values.reshape(-1, 1)
#     y[0] = 0
    scores = model.validate_model(clf, X, y)
    return scores


def main(stock):
    """
    This gets gdf_data
    :return:
    """
    K = 50
    length = 15000
    rr = [0.01, 0.05, 0.1, 0.5, 1.0]
    ss = [0.01, 0.05, 0.1, 0.5, 1.0]
    gdf_data_dir = 'data_gdf_feature_scaling'
    results_dir = 'data_res_gdf_feature_scaling'
    gdf_start = 24
    gdf_end = 26
    algorithm = 'svm_rbf'
    for r in rr:
        for s in ss:
            results_filename = os.path.join(
                results_dir, 'res_{}_len{}_r{}_s{}_K{}-{}.csv'.format(stock, length, r, s, gdf_start, gdf_end))
            results_partial_filename = os.path.join(
                results_dir, 'res_{}_len{}_r{}_s{}_K{}-{}_partial.csv'.format(stock, length, r, s, gdf_start, gdf_end))

            gdf_filename = 'gdf_{}_r{}_s{}_K{}_feature_scaling'.format(stock, r, s, K)

            if os.path.exists(results_filename):
                print('Exists ', results_filename)
                continue
            else:
                print('Will create ', results_filename)
            dfs, dfs_test = lob.load_prepared_data(gdf_filename, data_dir=gdf_data_dir, cv=False, length=length)
            gdf_columns = ['gdf_' + str(i) for i in range(gdf_start, gdf_end)]

            results = []
            for C in [1, 10, 100, 1000, 10000]:
                for gamma in [1, 10, 100, 1000, 10000]:
                    res = {'C': C, 'gamma': gamma, 'r': r, 's': s, 'stock': stock, 'K': K, 'method': algorithm}
                    print('********************************************')
                    print(res)
                    try:
                        scores = svm_classification(dfs, gdf_columns, C=C, gamma=gamma)
                        print(res, scores)
                        results.append({**res, **scores})
                    except Exception as e:
                        print('Exception', e, res)
                        results.append(res)
                    pd.DataFrame(results).to_csv(results_partial_filename)
            pd.DataFrame(results).to_csv(results_filename)


if __name__ == "__main__":
    pool = Pool(processes=3)
    stocks_to_calculate = ['9061', '9064', '9265']
    res = [pool.apply_async(main, [s]) for s in stocks_to_calculate]
    print([r.get() for r in res])
    print(stocks_to_calculate)
