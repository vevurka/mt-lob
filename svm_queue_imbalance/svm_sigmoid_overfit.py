import logging
import os

import pandas as pd
from lob_data_utils import roc_results, stocks_numbers
from lob_data_utils.lob_classify import LobClassify

logger = logging.getLogger(__name__)


def main(stock, C, gamma, coef0):
    results_dir = 'res_svm_overfit'
    data_length = 24000
    svm_gdf_res = LobClassify(stock, data_length=data_length,
                              data_dir='../data/prepared')
    weights = svm_gdf_res.get_classes_weights()
    results = []

    scores = svm_gdf_res.train_svm(C=C, gamma=gamma, kernel='sigmoid', feature_name='que_prev', coef0=coef0,
                                   class_weight=weights, should_validate=True)
    results.append(scores)

    pd.DataFrame(results).to_csv(
        os.path.join(results_dir, 'svm_overfit_sigmoid_{}_len{}.csv'.format(stock, data_length)))


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stocks = ['9268', '4549']
    main('9268', 0.001, 1000.0, 100.0)
    main('4549', 0.001, 1000.0, 100.0)