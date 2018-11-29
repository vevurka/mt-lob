import os
import unittest
from ast import literal_eval
from datetime import datetime

import pandas as pd
import numpy as np
from lob_data_utils import roc_results
from numpy.testing import assert_array_equal


class TestTransformOrders(unittest.TestCase):
    def test(self):
        df = pd.read_csv(os.path.join('data/9061.csv'))
        print(df.iloc[0:5])
        o = transform_to_orders_z_score(df.iloc[0:1000])
        print(o)
        #assert_array_equal(1, 2)

    def test_mean_ask_bid_for_one_order(self):
        data = [{
            'Unnamed: 0': datetime(2018, 12, 12, 12, 12, 12),
            'bid': '[(1, 2), (2, 3), (3, 4)]',
            'ask': '[(5, 2), (6, 4), (7, 6)]',
        }]
        order = add_means_and_stds(pd.DataFrame(data))
        assert_array_equal(order['ask_mean_price'].values, [6])
        assert_array_equal(order['bid_mean_price'].values, [2])
        assert_array_equal(order['ask_mean_volume'].values, [4])
        assert_array_equal(order['bid_mean_volume'].values, [3])

    def test_mean_ask_bid_for_two_orders_different_days(self):
        data = [{
            'Unnamed: 0': datetime(2018, 12, 12, 12, 12, 12),
            'bid': '[(1, 2), (2, 3), (3, 4)]',
            'ask': '[(5, 2), (6, 4), (7, 6)]',
        },
            {
                'Unnamed: 0': datetime(2018, 12, 13, 12, 12, 12),
                'bid': '[(2, 3), (3, 4), (4, 5)]',
                'ask': '[(6, 6), (7, 8), (8, 10)]',
            }
        ]
        order = add_means_and_stds(pd.DataFrame(data))

        assert_array_equal(order['ask_mean_price'].values, [6, 7])
        assert_array_equal(order['bid_mean_price'].values, [2, 3])
        assert_array_equal(order['ask_mean_volume'].values, [4, 8])
        assert_array_equal(order['bid_mean_volume'].values, [3, 4])

        assert_array_equal(order['prev_ask_mean_price'].values, [np.nan, 6])
        assert_array_equal(order['prev_bid_mean_price'].values, [np.nan, 2])
        assert_array_equal(order['prev_ask_mean_volume'].values, [np.nan, 4])
        assert_array_equal(order['prev_bid_mean_volume'].values, [np.nan, 3])

    def test_mean_ask_bid_for_two_orders_same_days(self):
        data = [{
            'Unnamed: 0': datetime(2018, 12, 12, 12, 12, 12),
            'bid': '[(1, 2), (2, 3)]',
            'ask': '[(5, 2), (6, 4)]',
        }, {
            'Unnamed: 0': datetime(2018, 12, 12, 12, 13, 12),
            'bid': '[(3, 4)]',
            'ask': '[(7, 6)]',
        }]
        order = add_means_and_stds(pd.DataFrame(data))

        assert_array_equal(order['ask_mean_price'].values, [6, 6])
        assert_array_equal(order['bid_mean_price'].values, [2, 2])
        assert_array_equal(order['ask_mean_volume'].values, [4, 4])
        assert_array_equal(order['bid_mean_volume'].values, [3, 3])


def _means_std(df: pd.DataFrame, d: datetime) -> dict:
    bids = []
    asks = []
    bid_mean_price = None
    bid_std_price = None
    bid_mean_volume = None
    bid_std_volume = None
    ask_mean_price = None
    ask_std_price = None
    ask_mean_volume = None
    ask_std_volume = None
    for i, row in df.loc[str(d.date())].iterrows():
        bids += [literal_eval(row.get('bid'))][0]
        asks += [literal_eval(row.get('ask'))][0]
    bids = np.array(bids)
    asks = np.array(asks)
    if np.any(bids):
        bid_mean_price = bids[:, 0].mean()
        bid_mean_volume = bids[:, 1].mean()
        bid_std_price = bids[:, 0].std()
        bid_std_volume = bids[:, 1].std()
    if np.any(asks):
        ask_mean_price = asks[:, 0].mean()
        ask_mean_volume = asks[:, 1].mean()
        ask_std_price = asks[:, 0].std()
        ask_std_volume = asks[:, 1].std()

    return {
        'bid_mean_price': bid_mean_price,
        'bid_std_price': bid_std_price,
        'bid_mean_volume': bid_mean_volume,
        'bid_std_volume': bid_std_volume,
        'ask_mean_price': ask_mean_price,
        'ask_std_price': ask_std_price,
        'ask_mean_volume': ask_mean_volume,
        'ask_std_volume': ask_std_volume
    }


def convert_to_previous(d: dict) -> dict:
    new_d = {}
    for k, v in d.items():
        new_d['prev_' + k] = v
    return new_d


def add_means_and_stds(df: pd.DataFrame) -> pd.DataFrame:
    print(len(df))
    df.index = df['Unnamed: 0']
    df.index = pd.to_datetime(df.index)
    rng = pd.date_range(min(df.index), max(df.index), freq='d')
    mean_stds = []
    previous_stds = []
    previous_std = {}
    for d in rng:
        if d.date().strftime('%A') == 'Saturday' or d.date().strftime('%A') == 'Sunday':
            continue
        number_of_samples = len(df.loc[str(d.date())])
        mean_std = _means_std(df, d)
        for i in range(number_of_samples):
            mean_stds.append(mean_std)
            if previous_std:
                previous_stds.append(previous_std)
            else:
                previous_std = convert_to_previous(mean_std)
                for k, v in previous_std.items():
                    previous_std[k] = None
                previous_stds.append(previous_std)
        previous_std = convert_to_previous(mean_std)
    df_mean_std = pd.DataFrame(mean_stds, index=df.index)
    df_prev_mean_std = pd.DataFrame(previous_stds, index=df.index)
    df = pd.concat([df, df_mean_std], axis=1, join_axes=[df.index])
    df = pd.concat([df, df_prev_mean_std], axis=1, join_axes=[df.index])
    print(len(df))
    return df


def transform_to_orders_z_score(df: pd.DataFrame) -> pd.DataFrame:
    df = add_means_and_stds(df)
    bid_norms = []
    ask_norms = []
    for i, row in df.iterrows():
        bid = np.array([literal_eval(row.get('bid'))][0])
        ask = np.array([literal_eval(row.get('ask'))][0])
        bids_norm = []
        asks_norm = []
        for b in bid:
            price_norm = (b[0] - row['prev_bid_mean_price']) / row['prev_bid_std_price']
            volume_norm = (b[1] - row['prev_bid_mean_volume']) / row['prev_bid_std_volume']
            bids_norm.append((price_norm, volume_norm))
        for b in ask:
            price_norm = (b[0] - row['prev_ask_mean_price']) / row['prev_ask_std_price']
            volume_norm = (b[1] - row['prev_ask_mean_volume']) / row['prev_ask_std_volume']
            asks_norm.append((price_norm, volume_norm))
        bid_norms.append(bids_norm)
        ask_norms.append(asks_norm)
    df['bid_norm'] = bid_norms
    df['ask_norm'] = ask_norms
    return df


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
    print(stock, datetime.now().isoformat())

    length = 15000
    filename = '{}_len{}.csv'.format(stock, length)
    f = os.path.join('data_gdf/', filename)
    if not os.path.exists(f):
        print('preparing', filename, datetime.now().isoformat())

        df = pd.read_csv(os.path.join('data/', stock + '.csv'))
        df = transform_to_orders(df.iloc[0:length])
        print('writing', filename, datetime.now().isoformat())
        df.to_csv(f)
        print(len(df))
    return True


if __name__ == "__main__":
    main('9061')

    from multiprocessing import Pool

   # stocks = list(roc_results.results_10000.keys())
    stocks = ['9061', '9064', '9265']
    pool = Pool(processes=5)

    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])


