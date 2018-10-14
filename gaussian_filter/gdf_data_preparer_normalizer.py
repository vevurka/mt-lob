import os
from ast import literal_eval
from datetime import datetime

import pandas as pd
import numpy as np

from scipy.stats import norm


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
        orders.append(o)
    return pd.DataFrame(orders)


def main(stock):
    print(stock, datetime.now().isoformat())

    length = 10000
    filename = '{}_len{}_normalized.csv'.format(stock, length)
    print('preparing', filename, datetime.now().isoformat())

    df = pd.read_csv(os.path.join('data/', stock + '.csv'))
    df = transform_to_orders(df.iloc[0:length])
    print('writing', filename, datetime.now().isoformat())
    df.to_csv(os.path.join('data_gdf/', filename))
    return True


if __name__ == "__main__":
    from multiprocessing import Pool
    stocks = ['9061', '9062', '9063', '9064', '9065']
    pool = Pool(processes=len(stocks))

    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])


