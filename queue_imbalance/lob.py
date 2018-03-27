import pandas as pd

from typing import Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np

import warnings

warnings.filterwarnings('ignore')


def parse_order_book_helper(order_book: str) -> Tuple[datetime, list, list]:
    """
    Input order_book row has format: 20130904 09080000 BID 2 2000 3 3000 ASK 4 4000 5 5000
    Returns datetime of LOB, bid as a tuple (prize, number of items),
    ask as tuple (prize, number of items)
    """
    parts = order_book.split()
    if not parts:
        return None, [], []

    dt = datetime.strptime(parts[0] + parts[1], '%Y%m%d%H%M%S%f')
    idx = 2 + 1  # skip string 'BID'

    bid = []
    while idx + 1 < len(parts) and parts[idx] != 'ASK':
        bid.append((float(parts[idx]), float(parts[idx + 1])))
        idx += 2

    idx += 1  # skip string 'ASK'
    ask = []
    while idx + 1 < len(parts) and idx < len(parts):
        ask.append((float(parts[idx]), float(parts[idx + 1])))
        idx += 2

    return dt, bid, ask


def parse_data(filename: str) -> pd.DataFrame:
    with open(filename) as f:
        order_books = f.read().split('\n')
    parsed_order_book = []
    for o in order_books:
        parsed_o = parse_order_book_helper(o)
        if parsed_o[0]:
            parsed_order_book.append(parse_order_book_helper(o))

    df = pd.DataFrame([p[1:] for p in parsed_order_book],
                      index=[p[0] for p in parsed_order_book], columns=['bid', 'ask'])
    return df


def load_data(company: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = {}
    train_dates = ['0901', '0916', '1001', '1016']
    test_date = '1101'
    for date in train_dates:
        df = parse_data('data/LOB/OrderBookSnapshots_{}_{}.csv'.format(company, date))
        dfs[date] = prepare_dataframe(df)
    df_test = parse_data('data/LOB/OrderBookSnapshots_{}_{}.csv'.format(company, test_date))
    df_test = prepare_dataframe(df_test)
    df = pd.concat(dfs.values()).sort_index()

    print('Training set length:', len(df))
    print('Testing set length:', len(df_test))
    return df, df_test


def get_bid_price(df: pd.DataFrame, index: int) -> float:
    bid_list = df['bid'][index]
    if not bid_list:
        return 0
    return max([price for price, vol in bid_list])


def get_ask_price(df: pd.DataFrame, index: int) -> float:
    ask_list = df['ask'][index]
    if not ask_list:
        return 0
    return min([price for price, vol in ask_list])


def get_mid_price(df: pd.DataFrame, index: int) -> float:
    ask_price = get_ask_price(df, index)
    bid_price = get_bid_price(df, index)
    return (ask_price + bid_price) / 2


def sum_buy_active_orders(price: float, df: pd.DataFrame, index: int) -> float:
    if not df['bid'][index]:
        return 0
    return sum([vol for p, vol in df['bid'][index] if p == price])


def sum_sell_active_orders(price: float, df: pd.DataFrame, index: int) -> float:
    if not df['ask'][index]:
        return 0
    return sum([vol for p, vol in df['ask'][index] if p == price])


def queue_imbalance(df: pd.DataFrame, index: int):
    diff_bid_ask = sum_buy_active_orders(get_bid_price(df, index), df, index) - \
                   sum_sell_active_orders(get_ask_price(df, index), df, index)
    sum_bid_ask = sum_buy_active_orders(get_bid_price(df, index), df, index) + \
                  sum_sell_active_orders(get_ask_price(df, index), df, index)
    if sum_bid_ask == 0:
        return 0  # as the lists are the same length 0
    return diff_bid_ask / sum_bid_ask


def add_mid_price_indicator(df: pd.DataFrame) -> pd.DataFrame:
    y = []
    current_mid_price = get_mid_price(df, 0)
    for i in range(0, len(df)-1):
        future_mid_price = get_mid_price(df, i+1)
        y.append(int(future_mid_price > current_mid_price))
        current_mid_price = future_mid_price
    df['mid_price_indicator'] = y + [None]
    df = df.dropna()
    return df


def add_queue_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    queue_imbalances = []
    for i in range(0, len(df)):
        queue_imbalances.append(queue_imbalance(df, i))
    df['queue_imbalance'] = queue_imbalances
    return df


def prepare_dataframe(df):
    df = df.between_time('10:00', '15:00')
    df['bid_price'] = [get_bid_price(df, i) for i in range(len(df))]
    df['ask_price'] = [get_ask_price(df, i) for i in range(len(df))]
    df['mid_price'] = [get_mid_price(df, i) for i in range(len(df))]
    df['sum_sell_ask'] = [sum_sell_active_orders(get_ask_price(df, i), df, i) for i in
                          range(len(df))]
    df['sum_buy_bid'] = [sum_buy_active_orders(get_bid_price(df, i), df, i) for i in
                         range(len(df))]
    rows_to_remove = []
    for i in range(len(df) - 1, 0, -1):
        if df['mid_price'].iloc[i] == df['mid_price'].iloc[i - 1]:
            rows_to_remove.append(i)

    for r in rows_to_remove:  # rows_to_remove is reversed so we can just remove
        df = df.drop(df.index[r])
    df = add_mid_price_indicator(df)
    df = add_queue_imbalance(df)
    return df


def svm_classification(df, start_idx, end_idx):
    clf = svm.SVC(probability=True)
    X = df['queue_imbalance'][start_idx:end_idx].values.reshape(-1, 1)
    y = df['mid_price_indicator'][start_idx:end_idx].values.reshape(-1, 1)
    y[0] = 0
    clf.fit(X, y)
    return clf


def logistic_regression(df, start_idx, end_idx):
    clf = LogisticRegression()
    X = df['queue_imbalance'][start_idx:end_idx].values.reshape(-1, 1)
    y = df['mid_price_indicator'][start_idx:end_idx].values.ravel()
    y[0] = 0
    clf.fit(X, y)
    return clf


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_density_imbalance_vs_mid(df, st, end):    
    y = df['queue_imbalance'].iloc[st:end].values
    x = df['mid_price'].iloc[st:end].values
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # that most dense points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50, edgecolor='')
    plt.figure()