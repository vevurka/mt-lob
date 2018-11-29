import pandas as pd
from lob_data_utils import lob
import numpy as np


def compare(a, b) -> int:
    if a > b:
        return 1
    elif a == b:
        return 0
    else:
        return -1


def add_state(df: pd.DataFrame):
    """
    state will be bid_price, ask_price, spread, sum_sell_ask, sum_buy_bid, mid_price
    queue imbalance
    TODO: spread does not seem to have any sense!
    """
    df['spread'] = df['ask_price'] - df['bid_price']
    states = {}
    previous_row = None
    for i, row in df.iterrows():
        if i == df.index[0]:
            states['state_bid'] = [np.nan]
            states['state_ask'] = [np.nan]
            states['state_sum_ask'] = [np.nan]
            states['state_sum_bid'] = [np.nan]
            states['state_spread'] = [np.nan]
            states['state_imbalance'] = [np.nan]
            states['state_mid'] = [np.nan]
            previous_row = row
            continue
        states['state_bid'].append(compare(row['bid_price'], previous_row['bid_price']))
        states['state_ask'].append(compare(row['ask_price'], previous_row['ask_price']))
        states['state_sum_ask'].append(compare(row['sum_sell_ask'], previous_row['sum_sell_ask']))
        states['state_sum_bid'].append(compare(row['sum_buy_bid'], previous_row['sum_buy_bid']))
        states['state_spread'].append(compare(row['spread'], previous_row['spread']))
        states['state_imbalance'].append(compare(row['queue_imbalance'], previous_row['queue_imbalance']))
        states['state_mid'].append(compare(row['mid_price'], previous_row['mid_price']))
        previous_row = row
    df['state_ask'] = states['state_ask']
    df['state_bid'] = states['state_bid']
    df['state_sum_ask'] = states['state_sum_ask']
    df['state_sum_bid'] = states['state_sum_bid']
    df['state_spread'] = states['state_spread']
    df['state_imbalance'] = states['state_imbalance']
    df['state_mid'] = states['state_mid']
    return df


def get_state_number(df: pd.DataFrame, idx: int) -> int:
    state_labels = sorted([c for c in df.columns if 'state' in c])
    num = 0
    for i in range(len(state_labels)):
        num += 3 ** i * (df.loc[idx][state_labels[i]] + 1)
    return int(num)


def get_state_number_for_labels(state_labels: dict) -> int:
    num = 0
    for i in range(len(state_labels.keys())):
        num += 3 ** i * (state_labels[list(state_labels.keys())[i]] + 1)
    return int(num)


def unhash_state(state_num, state_labels):
    state = {}
    for i in range(len(state_labels)-1, -1, -1):
        if state_num // (3 ** i) >= 1:
            n = state_num // (3 ** i)
            state_num -= n * 3 ** i
            state[state_labels[i]] = n - 1
        else:
            state[state_labels[i]] = - 1
    return state


def main():
    stock = '13003'
    data_length = 10000
    df, df_cv, df_test = lob.load_prepared_data(
        stock, data_dir='../queue_imbalance/data/prepared', cv=True, length=data_length)
    df = add_state(df)
    print(df[['state_bid', 'bid_price']].iloc[0:3])


if __name__ == "__main__":
    main()