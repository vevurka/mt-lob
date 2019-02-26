import logging
import os

import numpy as np
import pandas as pd
from lob_data_utils import stocks_numbers
from lob_data_utils.lob_classify import LobClassifySimple

logger = logging.getLogger(__name__)


def is_calculated(df, C, gamma):
    if not np.any(df):
        return False
    df_g = df[df['gamma'] == gamma]
    if np.any(df_g):
        df_c = df_g[df_g['C'] == C]
        return np.any(df_c)


def main(stock):
    results_dir = 'res_prev_que_svm'
    data_length = 24000
    svm_gdf_res = LobClassifySimple(stock, data_length=data_length, data_dir='../data/prepared')
    weights = svm_gdf_res.get_classes_weights()
    filename_partial = os.path.join(results_dir, 'svm_rbf_{}_len{}.csv_partial'.format(stock, data_length))
    filename = os.path.join(results_dir, 'svm_rbf_{}_len{}.csv'.format(stock, data_length))

    df_partial = pd.DataFrame()
    if os.path.exists(filename_partial):
        df_partial = pd.read_csv(filename_partial)
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for g in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            if is_calculated(df_partial, C, g):
                logger.info('Already calculated %s %s %s', stock, C, g)
                continue
            scores = svm_gdf_res.train_svm(C=C, gamma=g, kernel='rbf', feature_name='que_prev', class_weight=weights)
            df_partial = df_partial.append([scores])
            df_partial.to_csv(filename_partial)
    df_partial.to_csv(filename)


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=3)
    stocks = stocks_numbers.chosen_stocks
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])
