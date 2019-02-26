import logging
import os

import pandas as pd
from lob_data_utils import stocks_numbers
from lob_data_utils.lob_classify import LobClassifySimple

logger = logging.getLogger(__name__)


def main(stock):
    results_dir = 'res_prev_que_svm'
    data_length = 24000
    svm_gdf_res = LobClassifySimple(stock, data_length=data_length, data_dir='../data/prepared')
    filename = os.path.join(results_dir, 'svm_linear_{}_len{}.csv'.format(stock, data_length))
    if os.path.exists(filename):
        logger.info('Exists %s', filename)
        return {stock: False}
    results = []
    weights = svm_gdf_res.get_classes_weights()
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        scores = svm_gdf_res.train_svm(C=C, kernel='linear', feature_name='que_prev', class_weight=weights)
        results.append(scores)
    pd.DataFrame(results).to_csv(filename)
    return {stock: True}


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    stocks = stocks_numbers.chosen_stocks
    pool = Pool(processes=1)
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])



