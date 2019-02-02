import logging
import os

import pandas as pd
from lob_data_utils import roc_results, stocks_numbers
from lob_data_utils.lob_classify import LobClassify

logger = logging.getLogger(__name__)


def main(stock):
    results_dir = 'res_svm_prev'
    data_length = 24000
    svm_gdf_res = LobClassify(stock, data_length=data_length,
                              data_dir='../data/prepared')
    weights = svm_gdf_res.get_classes_weights()
    results = []
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for g in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            scores = svm_gdf_res.train_svm(C=C, gamma=g, kernel='rbf', feature_name='que_prev', class_weight=weights)
            results.append(scores)
        pd.DataFrame(results).to_csv(
            os.path.join(results_dir, 'svm_rbf_{}_len{}.csv_partial'.format(stock, data_length)))
    pd.DataFrame(results).to_csv(
        os.path.join(results_dir, 'svm_prev_rbf_{}_len{}.csv'.format(stock, data_length)))


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=8)
    stocks = stocks_numbers.chosen_stocks
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])
