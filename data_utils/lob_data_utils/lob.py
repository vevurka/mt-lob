import os

import pandas as pd

from typing import Tuple, Sequence, Optional
from datetime import datetime

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import learning_curve
import numpy as np

import warnings

from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve
import logging

logger = logging.getLogger(__name__)

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
    logger.debug('Started parsing %s', filename)
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


def load_prepared_data(stock: str, data_dir=None, length=None,
                       include_test=True) -> Sequence[Optional[pd.DataFrame]]:
    if data_dir is None:
        data_dir = '../data/prepared/'
    df = pd.read_csv(os.path.join(data_dir, stock + '.csv'))

    logger.debug('Len of data for ', stock, 'is', len(df))
    if length:
        if length > len(df):
            logger.warning('Not enough data for {} actual len: {}, wanted len: {}'.format(stock, len(df), length))
            return None, None
        return prepare_dataset(stock, df[0:length], include_test=include_test)
    else:
        return prepare_dataset(stock, df, include_test=include_test)


def prepare_dataset(stock: str, df: pd.DataFrame, include_test=True) -> Sequence[pd.DataFrame]:
    idx = len(df) // 5
    train = df.iloc[0:4*idx]
    test = df.iloc[4*idx:len(df)]

    if include_test:
        return train, test
    else:
        return df, None


def load_data(stock: str, data_dir=None, include_test=True) -> Sequence[pd.DataFrame]:
    if data_dir is None:
        data_dir = 'data/LOB/'
    train_dates = ['0901', '0916', '1001', '1016', '1101']
    df = pd.DataFrame()
    for date in train_dates:
        dfs = parse_data(data_dir + 'OrderBookSnapshots_{}_{}.csv'.format(stock, date))
        dfs = dfs.between_time('8:30', '16:00')
        df = df.append(dfs)
    logger.info('Finished parsing %s', stock)
    df = df.sort_index()
    df = df.reindex()
    df = prepare_dataframe(df)
    
    return prepare_dataset(stock, df, include_test=include_test)


def get_bid_price(df: pd.DataFrame, index: int) -> float:
    bid_list = df['bid'][index]
    if not np.any(bid_list):
        return 0
    return np.max([price for price, vol in bid_list])


def get_ask_price(df: pd.DataFrame, index: int) -> float:
    ask_list = df['ask'][index]
    if not np.any(ask_list):
        return 0
    return np.min([price for price, vol in ask_list])


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
    return df


def add_queue_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    queue_imbalances = []
    for i in range(0, len(df)):
        queue_imbalances.append(queue_imbalance(df, i))
    df['queue_imbalance'] = queue_imbalances
    return df


def prepare_dataframe(df: pd.DataFrame):
    df['bid_price'] = [get_bid_price(df, i) for i in range(len(df))]
    df['ask_price'] = [get_ask_price(df, i) for i in range(len(df))]
    df['mid_price'] = [get_mid_price(df, i) for i in range(len(df))]
    df['sum_sell_ask'] = [sum_sell_active_orders(get_ask_price(df, i), df, i) for i in
                          range(len(df))]
    df['sum_buy_bid'] = [sum_buy_active_orders(get_bid_price(df, i), df, i) for i in
                         range(len(df))]
    df['mid_price_prev'] = df['mid_price'].shift(-1)
    df = df[df['mid_price_prev'] != df['mid_price']]
    df.drop(columns=['mid_price_prev'], inplace=True)
    df = add_mid_price_indicator(df)
    df = add_queue_imbalance(df)
    return df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_density_imbalance_vs_mid(df, st, end):    
    y = df['queue_imbalance'].iloc[st:end].values
    x = df['mid_price'].iloc[st:end].values
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # that most dense points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50, edgecolor='')
    plt.figure()


def plot_f1(df: pd.DataFrame, clf, stock='', title='', c=None, linestyle=None,
             label=None, alpha=None, ax=None, features=None) -> float:
    if not features:
        prediction = clf.predict(df['queue_imbalance'].values.reshape(-1, 1))
    else:
        prediction = clf.predict(df[features])
    f1 = f1_score(df['mid_price_indicator'], prediction)
    precision, recall, thresholds = precision_recall_curve(df['mid_price_indicator'].values, prediction)
    kwargs = {'linestyle': linestyle, 'c': c, 'label': label, 'alpha': alpha}
    non_empty_kwargs = {}
    for k, v in kwargs.items():
        if v is not None:
            non_empty_kwargs[k] = v
    if non_empty_kwargs['label'] is None:
        non_empty_kwargs['label'] = ''
    non_empty_kwargs['label'] = ' '.join(
        [non_empty_kwargs['label'], '{} (f1 score = {:.4f})'.format(stock, f1)])
    if not ax:
        f, ax = plt.subplots(1, 1)
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.1)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    if ax:
        ax.step(recall, precision, alpha=0.5, where='post',  lw=2, **non_empty_kwargs)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="best")

    return f1


def plot_roc(df: pd.DataFrame, clf, stock='', title='', c=None, linestyle=None,
             label=None, alpha=None, ax=None) -> float:
    prediction = clf.predict(df['queue_imbalance'].values.reshape(-1, 1))

    roc_score = roc_auc_score(df['mid_price_indicator'], prediction)
    fpr, tpr, thresholds = roc_curve(df['mid_price_indicator'].values, prediction)
    kwargs = {'linestyle': linestyle, 'c': c, 'label': label, 'alpha': alpha}
    non_empty_kwargs = {}
    for k, v in kwargs.items():
        if v is not None:
            non_empty_kwargs[k] = v
    if non_empty_kwargs['label'] is None:
        non_empty_kwargs['label'] = ''
    non_empty_kwargs['label'] = ' '.join(
        [non_empty_kwargs['label'], '{} (area = {:.4f})'.format(stock, roc_score)])
    if ax:
        ax.plot(fpr, tpr, **non_empty_kwargs)
        ax.plot([0, 1], [0, 1], 'r--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
    else:
        plt.plot(fpr, tpr, **non_empty_kwargs)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    return roc_score


def plot_roc_proba(df: pd.DataFrame, clf, stock='', title='', c=None, linestyle=None) -> float:
    prediction = clf.predict_proba(df['queue_imbalance'].values.reshape(-1, 1))[:, 1]

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


def plot_learning_curve(estimator, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), title='', ax=None,
                        scoring='roc_auc'):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    if ax:
        ax.grid()

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Validation score")

        ax.legend(loc="best")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
    else:
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")


def prepare_summary(stocks, dfs):
    df_summary = pd.DataFrame(index=stocks)
    sum_sell_ask_mean = []
    sum_buy_bid_mean = []
    max_trade_price = []
    min_trade_price = []
    bid_ask_spread = []
    bid_len = []
    ask_len = []
    mean_bid_ask_len = []
    mean_bid_len = []
    mean_ask_len = []

    for s in stocks:
        sum_sell_ask_mean.append(dfs[s]['sum_sell_ask'].mean())
        sum_buy_bid_mean.append(dfs[s]['sum_buy_bid'].mean())
        max_trade_price.append(max(dfs[s]['bid_price'].max(), dfs[s]['ask_price'].max()))
        min_trade_price.append(max(dfs[s]['bid_price'].min(), dfs[s]['ask_price'].min()))
        bid_ask_spread.append((dfs[s]['ask_price'] - dfs[s]['bid_price']).mean())

        max_len_bid = 0
        max_len_ask = 0

        for i, row in dfs[s].iterrows():
            if len(row['bid']) > max_len_bid:
                max_len_bid = len(row['bid'])
            if len(row['ask']) > max_len_ask:
                max_len_ask = len(row['ask'])
        bid_len.append(max_len_bid)
        ask_len.append(max_len_ask)

        sum_len_bid_ask = 0
        sum_len_bid = 0
        sum_len_ask = 0
        for i, row in dfs[s].iterrows():
            sum_len_bid_ask += (len(row['ask']) + len(row['bid']))
            sum_len_bid += len(row['bid'])
            sum_len_ask += len(row['ask'])
        mean_bid_ask_len.append(sum_len_bid_ask / (2 * len(dfs[s])))
        mean_bid_len.append(sum_len_bid / len(dfs[s]))
        mean_ask_len.append(sum_len_ask / len(dfs[s]))
    df_summary['sum_sell_ask_mean'] = sum_sell_ask_mean
    df_summary['sum_buy_bid_mean'] = sum_buy_bid_mean
    df_summary['max_trade_price'] = max_trade_price
    df_summary['min_trade_price'] = min_trade_price
    df_summary['bid_ask_spread'] = bid_ask_spread
    df_summary['max_len_ask'] = ask_len
    df_summary['max_len_bid'] = bid_len
    df_summary['mean_bid_ask_len'] = mean_bid_ask_len
    df_summary['mean_bid_len'] = mean_bid_len
    df_summary['mean_ask_len'] = mean_ask_len
    return df_summary
