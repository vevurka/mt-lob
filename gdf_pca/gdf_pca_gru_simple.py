import functools
import logging
import os
import sys

import numpy as np
import pandas as pd

from lob_data_utils import gdf_pca, stocks_numbers
from numpy.random import seed

seed(1)

logger = logging.getLogger(__name__)


def as_keras_metric(method):
    import tensorflow as tf
    from keras import backend as K
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


def matthews_correlation(y_true, y_pred):
    from keras import backend as K
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def get_model_func(unit, input_shape=None, kernel_regularization=None):
    import tensorflow as tf
    auc_roc = as_keras_metric(tf.metrics.auc)

    from keras.layers import Dense, Dropout
    from keras.layers import GRU, regularizers
    from keras.models import Sequential
    def get_model():
        model = Sequential()
        if kernel_regularization and kernel_regularization != 0.0:
            model.add(GRU(unit, activation='tanh',
                           kernel_regularizer=regularizers.l2(kernel_regularization)))
        else:
            model.add(GRU(unit, activation='tanh'))
        model.add(Dense(1, input_shape=(unit, 1), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc, matthews_correlation])
        return model

    return get_model


def train_lstm(stock, r, s, data_length, units=None, kernel_regularizations=None):
    import tensorflow as tf
    auc_roc = as_keras_metric(tf.metrics.auc)
    r = float(r)
    s = float(s)
    data_length = int(data_length)
    print('running', stock, r, s, data_length)
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    gdf_dfs = gdf_pca.SvmGdfResults(
        str(stock), r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)

    weights = gdf_dfs.get_classes_weights()
    feature = 'pca_n_gdf_que'  # , 'pca_n_gdf_que_prev']
    epochs = 50
    batch_size = 512
    n_steps = 1

    filename = os.path.join('res_gru', f'res_gru_pca_n_one_layer_{stock}_len{data_length}_r{r}_s{s}.csv')
    # if os.path.exists(filename):
    #     logger.info('Exists %s', filename)
    #     return
    partial_filename = filename + '_partial'

    df_partial = pd.DataFrame()
    if os.path.exists(partial_filename):
        df_partial = pd.read_csv(partial_filename)
        if 'kernel_reg' not in df_partial.columns:
            print('Kernel reg not in columns!')
            df_partial['kernel_reg'] = np.zeros(len(df_partial)).astype(np.float)
        df_partial.drop(columns=[c for c in df_partial.columns if 'Unnamed' in c], inplace=True)

    for unit in units:
        unit_str = f'({unit}: tanh, 1)'
        for kernel_reg in kernel_regularizations:

            if np.any(df_partial):
                row = df_partial[df_partial['unit'] == unit_str][df_partial['kernel_reg'] == kernel_reg][df_partial['n_steps'] == n_steps]
                if np.any(row):
                    print(f'Already calculated {stock} {unit_str} {kernel_reg}')
                    continue
            print(f'Will train {stock} r{r} s{s} {unit_str} {kernel_reg}')
            pca = gdf_dfs.get_pca(feature)
            get_model = get_model_func(unit, input_shape=(n_steps, pca.n_components_))
            plot_name = f'plot_lstm/{stock}_one_layer_u{unit}_kr{kernel_reg}_pca_n_r{r}_s{s}'
            score = gdf_dfs.train_lstm(
                get_model, feature_name=feature, method='gru',
                fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
                compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam',
                                'metrics': [matthews_correlation, auc_roc]},
                plot_name=plot_name, class_weight=weights, n_steps=n_steps)
            score = {**score, 'r': r, 's': s, 'unit': unit_str, 'kernel_reg': kernel_reg,
                     'epochs': epochs, 'batch_size': batch_size, 'n_steps': n_steps}
            df_partial = df_partial.append(pd.DataFrame([score]), ignore_index=True)
            df_partial.to_csv(partial_filename)
    df_partial.drop(columns=[c for c in df_partial.columns if 'Unnamed' in c], inplace=True)
    df_partial.to_csv(filename)
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stock = sys.argv[1]
    r = sys.argv[2]
    s = sys.argv[3]
    data_length = sys.argv[4]
    #train_lstm_specific_model(stock, r, s, data_length)

    units = [4, 8] #, 16, 32]
    kernel_regularizations = [0.0, 0.001] #, 0.0001, 0.01]
    train_lstm(stock, r, s, data_length, units=[4, 8], kernel_regularizations=[0.0, 0.001])
    train_lstm(stock, r, s, data_length, units=[16, 32], kernel_regularizations=[0.0, 0.001])
    train_lstm(stock, r, s, data_length, units=[4, 8], kernel_regularizations=[0.0001, 0.01])
    train_lstm(stock, r, s, data_length, units=[16, 32], kernel_regularizations=[0.0001, 0.01])
