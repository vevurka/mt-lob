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
import logging
seed(1)

logger = logging.getLogger(__name__)


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


def matthews_correlation(y_true, y_pred):
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


auc_roc = as_keras_metric(tf.metrics.auc)


def get_model_func(unit, input_shape=None):
    def get_model():
        model = Sequential()
        model.add(LSTM(unit, input_shape=input_shape, activation='relu'))
        model.add(Dense(1, input_shape=(unit, 1), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc, matthews_correlation, 'acc'])
        return model
    return get_model


def train_lstm(stock, r, s, data_length):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    gdf_dfs = gdf_pca.SvmGdfResults(
        str(stock), r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)

    units = [4, 8, 16, 32]
    weights = gdf_dfs.get_classes_weights()
    features = ['pca_n_gdf_que'] #, 'pca_n_gdf_que_prev']
    epochs = 50
    batch_size = 512
    n_steps = 16

    filename = os.path.join('res_lstm', f'res_lstm_pca_n_one_layer_{stock}_len{data_length}_r{r}_s{s}.csv')
    if os.path.exists(filename):
        logger.info('Exists %s', filename)
        return
    partial_filename = filename + '_partial'

    df_partial = pd.DataFrame()
    if os.path.exists(partial_filename):
        df_partial = pd.read_csv(partial_filename)

    for unit in units:
        for feature in features:
            if os.path.exists(filename):
                print(f'Exists {filename}.')
                return None
            pca = gdf_dfs.get_pca(feature)
            get_model = get_model_func(unit, input_shape=(n_steps, pca.n_components_))
            plot_name = f'plot_lstm/{stock}_one_layer_u{unit}_pca_n_r{r}_s{s}'
            score = gdf_dfs.train_lstm(
                get_model, feature_name=feature,
                fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
                compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': [auc_roc]},
                plot_name=plot_name, class_weight=weights, n_steps=n_steps)
            score = {**score, 'r': r, 's': s,  'unit': f'({unit}: relu, 1)',
                     'epochs': epochs, 'batch_size': batch_size, 'n_steps': n_steps}
            df_partial = df_partial.append(pd.DataFrame([score]), ignore_index=True)
            df_partial.to_csv(partial_filename)
    df_partial.to_csv(filename)
    return True


def main():
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    data_length = 24000
    stocks = stocks_numbers.chosen_stocks
    pool = Pool(processes=5)
    # stocks = ['11869', '4799', '9268']
    res = [pool.apply_async(train_lstm, [s, 0.25, 0.25, data_length]) for s in stocks]
    print([r.get() for r in res])
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
