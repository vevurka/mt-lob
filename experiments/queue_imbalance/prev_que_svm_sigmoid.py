import logging
import os

import numpy as np
import pandas as pd
from lob_data_utils import stocks_numbers
from lob_data_utils.lob_classify import LobClassifySimple

logger = logging.getLogger(__name__)


def is_calculated(df, C, gamma, coef0):
    if not np.any(df):
        return False
    df_g = df[df['gamma'] == gamma]
    if np.any(df_g):
        df_c = df_g[df_g['C'] == C]
        if np.any(df_c):
            return np.any(df_c[df_c['coef0'] == coef0])


def main(stock):
    results_dir = 'res_prev_que_svm'
    data_length = 24000
    svm_gdf_res = LobClassifySimple(stock, data_length=data_length, data_dir='../data/prepared')

    filename_partial = os.path.join(results_dir, 'svm_sigmoid_{}_len{}.csv_partial'.format(stock, data_length))
    filename = os.path.join(results_dir, 'svm_sigmoid_{}_len{}.csv'.format(stock, data_length))

    df_partial = pd.DataFrame()
    if os.path.exists(filename_partial):
        df_partial = pd.read_csv(filename_partial)

    weights = svm_gdf_res.get_classes_weights()
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        for g in [0.001, 0.01, 0.1, 1, 10, 100]:
            for coef0 in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                if is_calculated(df_partial, C, g, coef0):
                    logger.info('Already calculated %s %s %s %s', stock, C, g, coef0)
                    continue
                scores = svm_gdf_res.train_svm(C=C, gamma=g, kernel='sigmoid', feature_name='que_prev',
                                               coef0=coef0, class_weight=weights)
                df_partial = df_partial.append([scores])
                df_partial.to_csv(filename_partial)
    df_partial.to_csv(filename)


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=8)
    stocks = stocks_numbers.chosen_stocks
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])
