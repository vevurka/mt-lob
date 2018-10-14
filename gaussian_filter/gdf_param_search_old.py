from ast import literal_eval

import numpy as np
import pandas as pd
from lob_data_utils import lob
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


def gdf_representation(buy_orders, sell_orders, gdf):
    buy_price, buy_volume = buy_orders
    sell_price, sell_volume = sell_orders
    buy_gdf_y = gdf[0] * norm.pdf(buy_price, loc=gdf[1], scale=gdf[2])
    sell_gdf_y = gdf[0] * norm.pdf(sell_price, loc=gdf[1], scale=gdf[2])
    return np.clip(buy_volume, 0.0, buy_gdf_y).sum() + np.clip(sell_volume, 0.0, sell_gdf_y).sum()


def transform_to_orders(df: pd.DataFrame, n, gdfs) -> pd.DataFrame:
    order_list = []
    for idx, row in df.iterrows():
        l_bid = [literal_eval(row.get('bid'))][0]
        bid_dict = [{'price': p, 'volume': v} for p, v in l_bid]
        d_bid = pd.DataFrame(bid_dict, columns=['price', 'volume'])
        d_bid = d_bid.sort_values(by='price', ascending=False).iloc[0:n]
        d_bid = d_bid.sort_values(by='price', ascending=True)

        l_ask = [literal_eval(row.get('ask'))][0]
        ask_dict = [{'price': p, 'volume': v} for p, v in l_ask]
        d_ask = pd.DataFrame(ask_dict, columns=['price', 'volume'])
        d_ask = d_ask.sort_values(by='price', ascending=True).iloc[0:n]

        mid_price = row['mid_price']

        d_bid['volume_norm'] = d_bid['volume'] * d_bid['price']
        d_ask['volume_norm'] = d_ask['volume'] * d_ask['price']

        total_capital = d_bid['volume_norm'].sum() + d_ask['volume_norm'].sum()
        d_bid['volume_norm'] /= total_capital
        d_ask['volume_norm'] /= total_capital

        d_bid['price_norm'] = 1000 * (d_bid['price'] - mid_price) / mid_price
        d_ask['price_norm'] = 1000 * (d_ask['price'] - mid_price) / mid_price
        new_row_dict = {}
        for i in range(0, n):
            gdf_repr = gdf_representation((d_bid['price_norm'], d_bid['volume_norm']),
                                          (d_ask['price_norm'], d_ask['volume_norm']),
                                          gdfs[i, :])
            new_row_dict['gdf_' + str(i)] = gdf_repr
            new_row_dict['ask_vol' + str(i)] = d_ask.iloc[i]['volume_norm']
            new_row_dict['ask_p' + str(i)] = d_ask.iloc[i]['price_norm']
            new_row_dict['bid_vol' + str(i)] = d_bid.iloc[i]['volume_norm']
            new_row_dict['bid_p' + str(i)] = d_bid.iloc[i]['price_norm']
        new_row_dict['mid_price'] = row.get('mid_price')
        new_row_dict['mid_price_indicator'] = row.get('mid_price_indicator')

        order_list.append(new_row_dict)
    order_df = pd.DataFrame(order_list)
    return order_df


def svm_classification(df, gdf_columns):
    clf = SVC(C=1000)
    X = df.loc[:, gdf_columns]
    y = df['mid_price_indicator'].values.reshape(-1, 1)
    y[0] = 0
    clf.fit(X, y)
    return clf


def main(r=0.02, s=0.2, n=15, K=50):
    print('*****************************************************')
    print('r', r, 's', s)
    gdf_columns = ['gdf_' + str(i) for i in range(0, n)]
    gdfs_r = r * np.ones(K)
    gdfs_m = 0.1000 * np.hstack([np.arange(- K // 2, 0), np.arange(1, K // 2 + 1)])
    gdfs_s = s * np.ones(K)
    gdfs = np.vstack([gdfs_r, gdfs_m, gdfs_s]).T

    data_length = 5050

    stocks = ['9061']
    stocks = ['9062', '9063', '9064', '9065']

    for s in stocks:
        try:
            d, d_cv, d_test = lob.load_prepared_data(s, data_dir='data/', cv=True, length=data_length)
            dfs = transform_to_orders(d, n, gdfs)
            clf = svm_classification(dfs, gdf_columns)
            predictions = clf.predict(dfs.loc[:, gdf_columns])
            print('train', s, roc_auc_score(predictions, dfs['mid_price_indicator']))

            dfs_test = transform_to_orders(d_test, n, gdfs)
            predictions = clf.predict(dfs_test.loc[:, gdf_columns])
            print('test ', s, roc_auc_score(predictions, dfs_test['mid_price_indicator']))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main(r=0.9, s=0.2)
    main(r=0.9, s=0.3)
    main(r=0.7, s=0.3)

# C = 1
# r 0.9 s 0.2
# 9061 0.500330250991
# *****************************************************
# r 0.9 s 0.3
# 9061 0.750412677451
# *****************************************************
# r 0.7 s 0.3
# 9061 0.750412677451
# # *****************************************************



#C=1000


# *****************************************************
# r 0.9 s 0.2
# train 9061 0.535179713147
# test  9061 0.504001038283
# *****************************************************
# r 0.9 s 0.3
# train 9061 0.535523053216
# test  9061 0.538986297042
# *****************************************************
# r 0.7 s 0.3
# train 9061 0.534123453049
# test  9061 0.520670863042

# *****************************************************
# r 0.9 s 0.2
# train 9062 0.527841290522
# test  9062 0.473537549407
# train 9063 0.560700919674
# test  9063 0.526436272338
# train 9064 0.549560803361
# test  9064 0.542213038283
# train 9065 0.536490072154
# test  9065 0.500082255783
# *****************************************************
# r 0.9 s 0.3
# train 9062 0.542139169406
# test  9062 0.494610356963
# train 9063 0.546734463038
# test  9063 0.532351731314
# train 9064 0.556254984818
# test  9064 0.547674194171
# train 9065 0.527716625572
# test  9065 0.538655404211
# *****************************************************
# r 0.7 s 0.3
# train 9062 0.529190358264
# test  9062 0.482042729218
# train 9063 0.555873823731
# test  9063 0.539064985682
# train 9064 0.551474970996
# test  9064 0.529947038619
# train 9065 0.530086719405
# test  9065 0.535009534778
