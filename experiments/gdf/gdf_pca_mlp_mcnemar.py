import logging
import os

import numpy as np
import pandas as pd
from lob_data_utils import gdf_pca, stocks_numbers
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.contingency_tables import mcnemar

res_dir = 'res_gdf_pca_mlp_mcnemar'


def perform_mcnemar(res):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    data_length = 24000
    r = res['r'].values[0]
    s = res['s'].values[0]
    stock = str(int(res['stock'].values[0]))
    gdf_dfs = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)

    df = gdf_dfs.df
    df_test = gdf_dfs.df_test
    df_mlp = pd.read_csv(os.path.join(res_dir, f'pred_mlp_iter_{stock}_len{data_length}_r{r}_s{s}.csv'))

    reg = LogisticRegression(class_weight=gdf_dfs.get_classes_weights())
    reg.fit(df[['queue_imbalance']], df['mid_price_indicator'])
    log_pred = reg.predict(df_test[['queue_imbalance']])

    df_all = pd.DataFrame()
    df_all['pred_log'] = log_pred[(len(log_pred) - len(df_mlp)):]
    df_all['pred_mlp'] = df_mlp['pred'].values
    df_all['actual'] = df_test['mid_price_indicator'].values[(len(log_pred) - len(df_mlp)):]

    df_all['correct_mlp'] = (df_all['pred_mlp'] == df_all['actual']).astype(np.int64)
    df_all['correct_log'] = (df_all['pred_log'] == df_all['actual']).astype(np.int64)

    table = pd.crosstab(df_all['correct_mlp'], df_all['correct_log'])
    mcnemar_res = mcnemar(table, exact=False, correction=True)

    df_mcnemar = pd.DataFrame()
    df_mcnemar['pvalue'] = [mcnemar_res.pvalue]
    df_mcnemar['statistic'] = [mcnemar_res.statistic]
    df_mcnemar['TN'] = [table[0][0]]
    df_mcnemar['FN'] = [table[0][1]]
    df_mcnemar['FP'] = [table[1][0]]
    df_mcnemar['TP'] = [table[1][1]]
    df_mcnemar['stock'] = stock
    df_mcnemar.to_csv(os.path.join(res_dir, f'mcnemar_mlp_log_{stock}_len{data_length}_r{r}_s{s}.csv'))
    return mcnemar_res


def get_best_results(data_dir):
    df_res = pd.DataFrame()
    for f in os.listdir(data_dir):
        if not 'csv' in f:
            continue
        df_res = df_res.append(pd.read_csv(os.path.join(data_dir, f)))
    df_best = df_res.sort_values(by='matthews', ascending=False).groupby(['stock']).head(1)
    df_best.index = df_best['stock']
    return df_best


def main():
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    data_dir = 'res_gdf_pca_mlp'
    stocks = stocks_numbers.chosen_stocks
    df_best = get_best_results(data_dir)
    pool = Pool(processes=8)
    res = [pool.apply_async(perform_mcnemar, [df_best[df_best['stock'] == int(s)]]) for s in stocks]
    print([r.get() for r in res])


if __name__ == '__main__':
    main()
