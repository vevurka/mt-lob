import logging
import os

import pandas as pd
from lob_data_utils import roc_results
from lob_data_utils.lob_classify import LobClassify

logger = logging.getLogger(__name__)


def main(stock):
    results_dir = 'res_svm'
    data_length = 6929
    svm_gdf_res = LobClassify(stock, data_length=data_length,
                              data_dir='../queue_imbalance/data/prepared_removed')

    results = []
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        for g in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            scores = svm_gdf_res.train_svm(C=C, gamma=g, kernel='rbf', feature_name='que')
            results.append(scores)
        pd.DataFrame(results).to_csv(
            os.path.join(results_dir, 'svm_rbf_{}_len{}.csv_partial'.format(stock, data_length)))
    pd.DataFrame(results).to_csv(
        os.path.join(results_dir, 'svm_rbf_{}_len{}.csv'.format(stock, data_length)))


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=4)
    cluster1 = ['9061', '3459', '4549', '9761', '4851']
    cluster2 = ['9062', '11869', '12255', '2748', '4320']
    cluster3 = ['11583', '4799', '9268', '10470', '9058']
    stocks = cluster1 + cluster2 + cluster3
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])
