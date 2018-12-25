import os
from multiprocessing.pool import Pool

import pandas as pd
from lob_data_utils import lob, model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def svm_classification(df, gdf_columns) -> dict:
    clf = LogisticRegression()
    X = df.loc[:, gdf_columns]
    y = df['mid_price_indicator'].values.reshape(-1, 1)
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
    gdf_data_dir = 'data_gdf'
    results_dir = 'data_res_logistic'
    gdf_start = 0
    gdf_end = 50
    algorithm = 'logistic'
    results = []
    results_filename = os.path.join(
        results_dir, 'res_log_{}_len{}_K{}-{}.csv'.format(stock, length, gdf_start, gdf_end))
    results_partial_filename = os.path.join(
        results_dir, 'res_log_{}_len{}_K{}-{}_partial.csv'.format(stock, length, gdf_start, gdf_end))
    for r in rr:
        for s in ss:
            gdf_filename = 'gdf_{}_len{}_r{}_s{}_K{}'.format(stock, length, r, s, K)

            dfs, dfs_test = lob.load_prepared_data(gdf_filename, data_dir=gdf_data_dir, cv=False, length=length)
            gdf_columns = ['gdf_' + str(i) for i in range(gdf_start, gdf_end)]

            res = {'r': r, 's': s, 'stock': stock, 'K': K, 'method': algorithm}
            print('********************************************')
            print(res)
            try:
                scores = svm_classification(dfs, gdf_columns)
                print(res, scores)
                results.append({**res, **scores})
            except Exception as e:
                print('Exception', e, res)
                results.append(res)
            pd.DataFrame(results).to_csv(results_partial_filename)
    pd.DataFrame(results).to_csv(results_filename)


if __name__ == "__main__":
    # pool = Pool(processes=3)
    # stocks_to_calculate = ['9061', '9064', '9265']
    # res = [pool.apply_async(main, [s]) for s in stocks_to_calculate]
    # print([r.get() for r in res])
   #  print(stocks_to_calculate)
    stocks_to_calculate = ['9061', '9064', '9265']
    for s in stocks_to_calculate:
        main(s)