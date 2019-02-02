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
from keras import regularizers
from lob_data_utils import gdf_pca, roc_results, stocks_numbers
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
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    gdf_dfs = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, data_dir='~/shit2/',
        gdf_filename_pattern=gdf_filename_pattern)
    pca = gdf_dfs.get_pca('pca_n_gdf_que_prev')
    units = [1, pca.n_components_, 2 * pca.n_components_, 20]
    number_of_hidden_layers = [0, 1, 2]
    regularization = [0, 0.1, 0.01]
    weights = gdf_dfs.get_classes_weights()

    epochs = 50
    batch_size = 10

    for unit in units:
        scores = []
        filename = os.path.join(
            'res_lstm', f'res_lstm_pca_n_{stock}_u{unit}_len{data_length}_r{r}_s{s}.csv')
        if os.path.exists(filename):
            print(f'Exists {filename}.')
            return None
        for hidden_layer, reg in zip(number_of_hidden_layers, regularization):

            model = Sequential()
            model.add(LSTM(unit, input_shape=(1, pca.n_components_)))

            for i in range(hidden_layer):
                if reg == 0:
                    model.add(Dense(unit, input_shape=(unit, 1), activation='tanh'))
                else:
                    model.add(Dense(unit, input_shape=(unit, 1), activation='tanh',
                                    kernel_regularizer=regularizers.l2(reg)))

            model.add(Dense(1, input_shape=(unit, 1), activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
            plot_name = f'plot_lstm/{stock}_u{unit}_h{hidden_layer}_reg{reg}_pca_n_r{r}_s{s}'
            score = gdf_dfs.train_lstm(model, feature_name='pca_n_gdf_que_prev',
                                       fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0,
                                                   'shuffle': False},
                                       compile_kwargs={'loss': 'binary_crossentropy',
                                                       'optimizer': 'adam',
                                                       'metrics': [auc_roc]},
                                       plot_name=plot_name, class_weight=weights)
            score = {**score, 'r': r, 's': s,  'unit': unit, 'hidden_layer': hidden_layer, 'regularization': reg,
                     'epochs': epochs, 'batch_size': batch_size }
            scores.append(score)
        df_scores = pd.DataFrame(scores)
        df_scores.to_csv(filename)
    return None


def main():
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    data_length = 24000
    stocks = stocks_numbers.chosen_stocks
    pool = Pool(processes=1)

    res = [pool.apply_async(train_lstm, [s, 0.1, 0.1, data_length]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(train_lstm, [s, 0.01, 0.1, data_length]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(train_lstm, [s, 0.1, 0.5, data_length]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(train_lstm, [s, 0.01, 0.5, data_length]) for s in stocks]
    print([r.get() for r in res])


if __name__ == '__main__':
    main()
