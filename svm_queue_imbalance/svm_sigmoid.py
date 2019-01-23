import logging
import os

import pandas as pd
from lob_data_utils import roc_results
from lob_data_utils.gdf_pca import SvmGdfResults

logger = logging.getLogger(__name__)


def main(stock):
    results_dir = 'res_svm_prev'
    data_length = 10000
    r = 0.1
    s = 0.1
    svm_gdf_res = SvmGdfResults(
        stock, data_length=data_length, r=r, s=s,
        gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')

    results = []
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for g in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            for coef0 in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                scores = svm_gdf_res.train_svm(C=C, gamma=g, kernel='sigmoid', feature_name='que_prev', coef0=coef0)
                results.append(scores)
        pd.DataFrame(results).to_csv(
            os.path.join(results_dir, 'svm_prev_sigmoid_{}_len{}.csv_partial'.format(stock, data_length)))
    pd.DataFrame(results).to_csv(
        os.path.join(results_dir, 'svm_prev_sigmoid_{}_len{}.csv'.format(stock, data_length)))


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=3)
    stocks = list(roc_results.results_10000.keys())
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])



