import logging
import os
from ast import literal_eval

import numpy as np
import pandas as pd
from lob_data_utils import stocks_numbers

logger = logging.getLogger(__name__)


def transform_to_orders(df: pd.DataFrame) -> pd.DataFrame:
    orders = []
    for idx, row in df.iterrows():
        bid = np.array([literal_eval(row.get('bid'))][0])
        ask = np.array([literal_eval(row.get('ask'))][0])
        # [(price, volume)]
        mid_price = row['mid_price']

        bid_vol_norm = bid[:, 1] * bid[:, 0]
        ask_vol_norm = ask[:, 1] * ask[:, 0]

        total_capital = bid_vol_norm.sum() + ask_vol_norm.sum()
        bid_vol_norm /= total_capital
        ask_vol_norm /= total_capital

        bid_price_norm = 1000 * (bid[:, 0] - mid_price) / mid_price
        ask_price_norm = 1000 * (ask[:, 0] - mid_price) / mid_price

        o = {}
        o['bid_norm'] = list(zip(bid_price_norm, bid_vol_norm))
        o['ask_norm'] = list(zip(ask_price_norm, ask_vol_norm))
        o['mid_price'] = mid_price
        o['mid_price_indicator'] = row['mid_price_indicator']
        o['datetime'] = row['Unnamed: 0']
        orders.append(o)
    return pd.DataFrame(orders)


def main(stock):
    filename = '{}_normalized.csv'.format(stock)
    f = os.path.join('data_normalized', filename)
    data_file = os.path.join('data/prepared', stock + '.csv')
    logger.info('Will normalize %s from filename %s to %s ', stock, data_file, f)
    if not os.path.exists(data_file):
        logger.info('Data file does not exists %s, please create one', data_file)
        return False
    if not os.path.exists(f):
        logger.info('Preparing %s', filename)

        df = pd.read_csv(data_file)
        df = transform_to_orders(df)
        logger.info('Writing %s', filename)
        df.to_csv(f)
    return True


if __name__ == "__main__":
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    all_stocks = stocks_numbers.chosen_stocks
    pool = Pool(processes=7)

    res = [pool.apply_async(main, [s]) for s in all_stocks]
    print([r.get() for r in res])
