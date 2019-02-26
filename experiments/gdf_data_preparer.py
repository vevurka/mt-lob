import logging
import os
from ast import literal_eval

import numpy as np
import pandas as pd
from lob_data_utils import stocks_numbers
from scipy.stats import norm

logger = logging.getLogger(__name__)


def gdf_representation(buy_orders, sell_orders, gdf):
    buy_price, buy_volume = buy_orders
    sell_price, sell_volume = sell_orders
    buy_gdf_y = gdf[0] * norm.pdf(buy_price, loc=gdf[1], scale=gdf[2])
    sell_gdf_y = gdf[0] * norm.pdf(sell_price, loc=gdf[1], scale=gdf[2])
    return np.clip(buy_volume, 0.0, buy_gdf_y).sum() + np.clip(sell_volume, 0.0, sell_gdf_y).sum()


def transform_to_orders(df: pd.DataFrame, gdfs, K) -> pd.DataFrame:
    order_list = []
    df.index = df['Unnamed: 0']
    df.index = pd.to_datetime(df.index)

    # rng = pd.date_range(min(df.index), max(df.index), freq='d')
    # df = df.loc[str(rng[1].date()):]
    for idx, row in df.iterrows():
        try:

            d_bid = np.array([literal_eval(row.get('bid_norm'))][0])
            d_ask = np.array([literal_eval(row.get('ask_norm'))][0])

            d_bid_prices = d_bid[:, 0]
            d_ask_prices = d_ask[:, 0]
            d_bid_volumes = d_bid[:, 1]
            d_ask_volumes = d_ask[:, 1]

        except Exception as e:
            print(e)
            raise e
        new_row_dict = {}
        for i in range(0, K):
            gdf_repr = gdf_representation((d_bid_prices, d_bid_volumes),
                                          (d_ask_prices, d_ask_volumes),
                                          gdfs[i, :])
            new_row_dict['gdf_' + str(i)] = gdf_repr
        new_row_dict['mid_price'] = row.get('mid_price')
        new_row_dict['mid_price_indicator'] = row.get('mid_price_indicator')
        new_row_dict['datetime'] = row.get('datetime')
        new_row_dict['queue_imbalance'] = row.get('queue_imbalance')

        order_list.append(new_row_dict)
    order_df = pd.DataFrame(order_list)
    return order_df


def main(stock):
    data_dir_in = 'data/data_normalized'
    data_dir_out = 'data/data_gdf'
    normalized_file = '{}_normalized.csv'.format(stock)

    rs = [(0.1, 0.1), (0.1, 0.5), (0.01, 0.1), (0.01, 0.5), (0.25, 0.25)]
    K = 50
    for r, s in rs:
        filename = 'gdf_{}_r{}_s{}_K{}.csv'.format(stock, r, s, K)
        if os.path.exists(os.path.join(data_dir_out, filename)):
            logger.info('Already exists %s', filename)
            continue

        gdfs_r = r * np.ones(K)
        gdfs_m = 0.1000 * np.hstack([np.arange(- K // 2, 0), np.arange(1, K // 2 + 1)])
        gdfs_s = s * np.ones(K)
        gdfs = np.vstack([gdfs_r, gdfs_m, gdfs_s]).T

        logger.info('Preparing %s from %s', filename, normalized_file)

        df = pd.read_csv(os.path.join(data_dir_in, normalized_file))
        df = transform_to_orders(df, gdfs, K)
        logger.info('Writing to %s  length %s', filename, len(df))
        df.to_csv(os.path.join(data_dir_out, filename))
    return True


if __name__ == "__main__":
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stocks = stocks_numbers.chosen_stocks

    pool = Pool(processes=16)

    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])
