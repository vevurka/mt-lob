import os

import pandas as pd

from typing import Tuple, Sequence
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import svm
import numpy as np

import warnings

from sklearn.metrics import roc_auc_score, roc_curve

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


def load_prepared_data(stock: str, data_dir=None, cv=False, length=5050) -> Sequence[pd.DataFrame]:
    if data_dir is None:
        data_dir = 'data/prepared/'
    df = pd.read_csv(os.path.join(data_dir, stock + '.csv'))
    print('Len of data for ', stock, 'is', len(df))
    if length:
        if length >= len(df):
            print('Not enough data for {} actual len: {}, wanted len: {}'.format(stock, len(df), length))
            return None, None, None
        return prepare_dataset(stock, df[0:length], cv=cv)
    else:
        return prepare_dataset(stock, df, cv=cv)


def prepare_dataset(stock:str, df: pd.DataFrame, cv=False) -> Sequence[pd.DataFrame]:
    idx = len(df) // 5
    train = df.iloc[idx:len(df)]
    test = df.iloc[0:idx]
    if cv:
        df_cv = train.iloc[len(train) - idx:len(train)]
        train = train.iloc[0:len(train) - idx]

    print('Training set length for {}: {}'.format(stock, len(train)))
    print('Testing set length for {}: {}'.format(stock, len(test)))

    if cv:
        print('Cross-validation set length for {}: {}'.format(stock, len(df_cv)))
        return train, df_cv, test
    else:
        return train, test


def load_data(stock: str, data_dir=None, cv=False) ->  Sequence[pd.DataFrame]:
    if data_dir is None:
        data_dir = 'data/LOB/'
    train_dates = ['0901', '0916', '1001', '1016', '1101']
    df = pd.DataFrame()
    for date in train_dates:
        if not np.any(df):
            df = parse_data(data_dir + 'OrderBookSnapshots_{}_{}.csv'.format(stock, date))
            df = df.between_time('9:00', '15:00')
        else:
            dfs = parse_data(data_dir + 'OrderBookSnapshots_{}_{}.csv'.format(stock, date))
            dfs = dfs.between_time('9:00', '15:00')
            df = df.append(dfs)
    
    df = df.sort_index()
    df = prepare_dataframe(df)
    
    return prepare_dataset(stock, df, cv=cv)


def get_bid_price(df: pd.DataFrame, index: int) -> float:
    bid_list = df['bid'][index]
    if not np.any(bid_list):
        return 0
    return max([price for price, vol in bid_list])


def get_ask_price(df: pd.DataFrame, index: int) -> float:
    ask_list = df['ask'][index]
    if not np.any(ask_list):
        return 0
    return min([price for price, vol in ask_list])


def get_mid_price(df: pd.DataFrame, index: int) -> float:
    ask_price = get_ask_price(df, index)
    bid_price = get_bid_price(df, index)
    return (ask_price + bid_price) / 2


def sum_buy_active_orders(price: float, df: pd.DataFrame, index: int) -> float:
    if not np.any(df['bid'][index]):
        return 0
    return sum([vol for p, vol in df['bid'][index] if p == price])


def sum_sell_active_orders(price: float, df: pd.DataFrame, index: int) -> float:
    if not np.any(df['ask'][index]):
        return 0
    return sum([vol for p, vol in df['ask'][index] if p == price])


def queue_imbalance(df: pd.DataFrame, index: int):
    sum_buy = sum_buy_active_orders(get_bid_price(df, index), df, index)
    sum_ask = sum_sell_active_orders(get_ask_price(df, index), df, index)
    diff_bid_ask = sum_buy - sum_ask
    sum_bid_ask = sum_buy + sum_ask
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
    clf.fit(X, y)
    return clf


def logistic_regression(df, start_idx, end_idx):
    clf = LogisticRegressionCV(fit_intercept=True, solver='sag')
    X = df['queue_imbalance'][start_idx:end_idx].values.reshape(-1, 1)
    y = df['mid_price_indicator'][start_idx:end_idx].values.ravel()
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


def plot_roc(df: pd.DataFrame, clf, stock='', title='', c=None, linestyle=None) -> float:
    prediction = clf.predict(df['queue_imbalance'].values.reshape(-1, 1))

    roc_score = roc_auc_score(df['mid_price_indicator'], prediction)
    fpr, tpr, thresholds = roc_curve(df['mid_price_indicator'].values, prediction)

    if c and linestyle:
        plt.plot(fpr, tpr, label='{} (area = {})'.format(stock, roc_score), c=c, linestyle=linestyle)
    else:
        plt.plot(fpr, tpr, label='{} (area = {})'.format(stock, roc_score))

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return roc_score