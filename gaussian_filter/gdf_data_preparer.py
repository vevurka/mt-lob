import os
from ast import literal_eval
from datetime import datetime

import pandas as pd
import numpy as np

from scipy.stats import norm


def gdf_representation(buy_orders, sell_orders, gdf):
    buy_price, buy_volume = buy_orders
    sell_price, sell_volume = sell_orders
    buy_gdf_y = gdf[0] * norm.pdf(buy_price, loc=gdf[1], scale=gdf[2])
    sell_gdf_y = gdf[0] * norm.pdf(sell_price, loc=gdf[1], scale=gdf[2])
    return np.clip(buy_volume, 0.0, buy_gdf_y).sum() + np.clip(sell_volume, 0.0, sell_gdf_y).sum()


def transform_to_orders(df: pd.DataFrame, gdfs, K) -> pd.DataFrame:
    order_list = []
    for idx, row in df.iterrows():
        d_bid = np.array([literal_eval(row.get('bid_norm'))][0])
        d_ask = np.array([literal_eval(row.get('ask_norm'))][0])

        new_row_dict = {}
        for i in range(0, K):
            gdf_repr = gdf_representation((d_bid[:, 0], d_bid[:, 1]),
                                          (d_ask[:, 0], d_ask[:, 1]),
                                          gdfs[i, :])
            new_row_dict['gdf_' + str(i)] = gdf_repr
        new_row_dict['mid_price'] = row.get('mid_price')
        new_row_dict['mid_price_indicator'] = row.get('mid_price_indicator')

        order_list.append(new_row_dict)
    order_df = pd.DataFrame(order_list)
    return order_df


def main(stock):
    rr = [0.01, 0.05, 0.1, 0.5, 1]
    ss = [0.01, 0.05, 0.1, 0.5, 1]
    print(stock, datetime.now().isoformat())
    for r in rr:
        for s in ss:
            K = 50
            length = 10000
            filename = 'gdf_{}_len{}_r{}_s{}_K{}.csv'.format(stock, length, r, s, K)
            if os.path.exists(os.path.join('data_gdf/', filename)):
                print('already exists ', filename, datetime.now().isoformat())
                continue

            gdfs_r = r * np.ones(K)
            gdfs_m = 0.1000 * np.hstack([np.arange(- K // 2, 0), np.arange(1, K // 2 + 1)])
            gdfs_s = s * np.ones(K)
            gdfs = np.vstack([gdfs_r, gdfs_m, gdfs_s]).T

            print('preparing', filename, datetime.now().isoformat())

            df = pd.read_csv(
                os.path.join('data_gdf/', stock + '_len{}_normalized.csv'.format(length)))
            df = transform_to_orders(df, gdfs, K)
            print('writing', filename, len(df), datetime.now().isoformat())
            df.to_csv(os.path.join('data_gdf/', filename))
    return True


results_10000 = {'9069': 0.538344342307573, '9063': 0.5650819783767761, '1472': 0.5608978983311691,
                 '9094': 0.57639224321991778, '9270': 0.55729080468416958,
                 '10166': 0.56101190476190477,
                 '9061': 0.55754178760884798, '2651': 0.55447470817120625,
                 '4060': 0.57672119635188301,
                 '12456': 0.55965708885899157, '9064': 0.54277410591578512,
                 '9761': 0.5526978971242964,
                 '1956': 0.57600772437849457, '11244': 0.53146751173341922,
                 '1113': 0.56879299562226393,
                 '13061': 0.56761419035475891, '10887': 0.56532969918231601,
                 '9062': 0.53387239423881161,
                 '7858': 0.57210413240995239, '13003': 0.56472528176229508,
                 '9067': 0.56418706510903138,
                 '10508': 0.57231305825091983, '1907': 0.53570056352459017,
                 '4549': 0.57546111340404471,
                 '4618': 0.54425063075346058, '3035': 0.58067060670126214,
                 '11867': 0.56158470903579749,
                 '9265': 0.5730084485660869, '4799': 0.58432600437896021,
                 '1388': 0.56857602251030048,
                 '9086': 0.569627298353514, '9058': 0.5472748810424245,
                 '11583': 0.54777986241400878,
                 '9268': 0.58192538155793316, '9065': 0.56242530247792566,
                 '2602': 0.56653758141172328,
                 '3161': 0.57977229962254351, '9074': 0.55719458898649643,
                 '9269': 0.57400457400457405,
                 '12059': 0.57784080226888157, '3879': 0.58023302330233029,
                 '5836': 0.55913284695775178,
                 '10484': 0.54923271162582699, '12417': 0.5791528486426516,
                 '9266': 0.54375070135782888,
                 '4320': 0.56145574458037695, '3022': 0.56426065162907268,
                 '8080': 0.57732443311082782,
                 '1431': 0.54649190192689323, '12255': 0.57466165413533843,
                 '11946': 0.58129692668936084,
                 '2813': 0.56522837753284882, '11869': 0.56136200458427588}


if __name__ == "__main__":
    from multiprocessing import Pool
    stocks = list(results_10000.keys())
    pool = Pool(processes=6)

    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])


