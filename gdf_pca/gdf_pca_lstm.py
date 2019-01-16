import functools
import logging
import os

import pandas as pd
import sklearn
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from lob_data_utils import gdf_pca, roc_results
from numpy.random import seed
seed(1)


def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


auc_roc = as_keras_metric(tf.metrics.auc)
mcc = as_keras_metric(sklearn.metrics.matthews_corrcoef)


def train_lstm(stock, r, s, data_length):
    gdf_filename_pattern = 'gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50'

    gdf_dfs = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length,
        gdf_filename_pattern=gdf_filename_pattern)

    units = [1, 64, 128]
    scores = []
    for unit in units:
        # model = Sequential()
        # model.add(LSTM(unit, input_shape=(1, 10)))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
        model = Sequential()
        model.add(LSTM(unit, input_shape=(1, 10)))
        model.add(Dense(unit, input_shape=(unit, 1), activation='tanh'))
        #  model.add(Dropout(0.1, input_shape=(128, 1)))
        model.add(Dense(1, input_shape=(unit, 1), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
        plot_name = f'plot_lstm/{stock}_u{unit}_r{r}_s{s}'
        score = gdf_dfs.train_lstm(model, feature_name='pca_gdf_que_prev10',
                                   fit_kwargs={'epochs': 50, 'batch_size': 100, 'verbose': 0}, plot_name=plot_name)
        score = {**score, 'unit': unit}
        scores.append(score)
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(
        os.path.join('res_lstm', f'res_lstm_{gdf_dfs.stock}_len{gdf_dfs.data_length}_r{gdf_dfs.r}_s{gdf_dfs.s}.csv'))
    return df_scores


def main():
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    data_length = 10000
    rs = [(0.1, 0.1)]  # (1.0, 1.0), (0.1, 1.0), (1.0, 0.1), (0.01, 0.1)]
    stocks = list(roc_results.result_cv_10000.keys())
    pool = Pool(processes=5)

    res = [pool.apply_async(train_lstm, [s, 0.1, 0.1, data_length]) for s in stocks]
    print([r.get() for r in res])


if __name__ == '__main__':
    main()
