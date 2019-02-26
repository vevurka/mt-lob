import logging
import os

import pandas as pd
from lob_data_utils.lob_classify import LobClassifySimple
from lob_data_utils import stocks_numbers
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def main(stock):
    results_dir = 'res_que_log'
    data_length = 24000
    log_classify = LobClassifySimple(stock, data_length=data_length, data_dir='../..//data/prepared')

    clf = LogisticRegression(class_weight='balanced')
    scores = log_classify.train_clf(clf, feature_name='que', should_validate=True)
    pd.DataFrame([scores]).to_csv(
       os.path.join(results_dir, 'que_log_{}_len{}.csv'.format(stock, data_length)))
    return {stock: True}


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    stocks = stocks_numbers.chosen_stocks
    pool = Pool(processes=1)
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])



