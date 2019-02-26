import os
import unittest
from ast import literal_eval
from datetime import datetime

import pandas as pd
import numpy as np
from lob_data_utils.roc_results import results_15000
from lob_data_utils import stocks


class TestOrdersNormalizers(unittest.TestCase):
    def test_z_score(self):
        data = [{
            'Unnamed: 0': datetime(2018, 12, 12, 12, 12, 12),
            'bid': '[(1, 2), (2, 3), (3, 4)]',
            'ask': '[(5, 2), (6, 4), (7, 6)]',
            'mid_price': 5,
            'mid_price_indicator': 1
        }]
        df_order = transform_to_orders_feature_scaling(pd.DataFrame(data))
        np.testing.assert_array_almost_equal(df_order.to_dict()['ask_price_norm'][0], [0.66666667, 0.83333333, 1.0])
        np.testing.assert_array_almost_equal(df_order.to_dict()['bid_price_norm'][0], [0.0, 0.166667, 0.333333])
        np.testing.assert_array_almost_equal(df_order.to_dict()['ask_volume_norm'][0], [0.104167, 0.25, 0.4375])
        np.testing.assert_array_almost_equal(df_order.to_dict()['bid_volume_norm'][0], [0.020833, 0.0625, 0.125])


def transform_to_orders_feature_scaling(df: pd.DataFrame) -> pd.DataFrame:
    orders = []
    for idx, row in df.iterrows():
        bid = np.array([literal_eval(row.get('bid'))][0])
        ask = np.array([literal_eval(row.get('ask'))][0])
        bid_prices = bid[:, 0]
        ask_prices = ask[:, 0]
        bid_volumes = bid[:, 1]
        ask_volumes = ask[:, 1]

        bid_volumes = bid_prices * bid_volumes
        ask_volumes = ask_prices * ask_volumes

        total_capital = bid_volumes.sum() + ask_volumes.sum()
        bid_vol_norm = bid_volumes / total_capital
        ask_vol_norm = ask_volumes / total_capital

        min_price = min(bid_prices)
        max_price = max(ask_prices)

        bid_price_norm = (bid_prices - min_price) / (max_price - min_price)
        ask_price_norm = (ask_prices - min_price) / (max_price - min_price)

        o = {}

        o['bid_price_norm'] = bid_price_norm.tolist()
        o['bid_volume_norm'] = bid_vol_norm.tolist()
        o['ask_price_norm'] = ask_price_norm.tolist()
        o['ask_volume_norm'] = ask_vol_norm.tolist()
        o['mid_price'] = row['mid_price']
        o['mid_price_indicator'] = row['mid_price_indicator']
        o['datetime'] = row['Unnamed: 0']
        orders.append(o)
    return pd.DataFrame(orders)


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


def main(stock, all_stocks: list):
    print(stock, datetime.now().isoformat(), all_stocks.index(stock))
    filename = '{}_normalized_balanced.csv'.format(stock)
    f = os.path.join('data_normalized_balanced/', filename)
    data_file = os.path.join('../data/prepared_balanced', stock + '.csv')
    if not os.path.exists(data_file):
        print(data_file)
        return False
    if not os.path.exists(f):
        print('preparing', filename, datetime.now().isoformat())

        df = pd.read_csv(data_file)
        df = transform_to_orders(df)
        print('writing', filename, datetime.now().isoformat())
        df.to_csv(f)
        print(len(df))
    return True


if __name__ == "__main__":
    from multiprocessing import Pool

    cluster1 = ['9061', '3459', '4549', '9761', '4851']
    cluster2 = ['9062', '11869', '12255', '2748', '4320']
    cluster3 = ['11583', '4799', '9268', '10470', '9058']
    all_stocks = cluster1 + cluster2 + cluster3
    pool = Pool(processes=7)

    res = [pool.apply_async(main, [s, all_stocks]) for s in all_stocks]
    print([r.get() for r in res])


