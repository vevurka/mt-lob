import functools
import logging
import os

import pandas as pd
from keras.layers import Dense
from keras import backend as K, regularizers
from lob_data_utils import gdf_pca, stocks_numbers
from sklearn.neural_network import MLPClassifier
from keras import optimizers, Sequential
import tensorflow as tf
import numpy as np

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


auc_roc = as_keras_metric(tf.metrics.auc)


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


def get_model1():  # 10470
    model = Sequential()
    model.add(Dense(16, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
    return model


def get_model2():
    model = Sequential()
    model.add(Dense(4, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
    return model


def get_model3():
    model = Sequential()
    model.add(Dense(8, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(4, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
    return model


def get_model4():
    model = Sequential()
    model.add(Dense(16, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(4, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
    return model


def get_model5():
    model = Sequential()
    model.add(Dense(8, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
    return model


models = {
    '(16: kernel_l2(0.01), 8: kernel_l2(0.01), 1)': get_model1,
    '(4: kernel_l2(0.01), 2: kernel_l2(0.01), 1)': get_model2,
    '(8: kernel_l2(0.01), 4: kernel_l2(0.01), 2: kernel_l2(0.01), 1)': get_model3,
    '(16: kernel_l2(0.01), 8: kernel_l2(0.01), 4: kernel_l2(0.01), 1)': get_model4,
    '(8: kernel_l2(0.01), 1)': get_model5,
}


def main(stock, r=0.1, s=0.1):
    result_dir = 'res_mlp_pca'

    data_length = 24000
    svm_gdf_res = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern='gdf_{}_r{}_s{}_K50')
    feature_name = 'pca_n_gdf_que'

    weights = svm_gdf_res.get_classes_weights()
    epochs = 50
    batch_size = 512
    filename = os.path.join(result_dir, 'mlp_pca_gdf_n_{}_len{}_r{}_s{}.csv'.format(stock, data_length, r, s))
    if os.path.exists(filename):
        print(f'Exists {filename}')
        return
    filename_partial = os.path.join(
        result_dir, 'mlp_pca_n_gdf_{}_len{}_r{}_s{}.csv_partial'.format(stock, data_length, r, s))
    df_partial = pd.DataFrame()
    if os.path.exists(filename_partial):
        print(f'Reading partial file {filename_partial}')
        df_partial = pd.read_csv(filename_partial)
    for k, get_m in models.items():
            if np.any(df_partial):

                print(filename_partial)
                row = df_partial[df_partial['hidden_layer_sizes'] == k]

                if np.any(row):
                    print(f'Read result for hidden layer {k} in {filename_partial}')
                    continue
            print(f'Training {stock} {r} {s} {k}')

            plot_name = f'plot_mlp/{stock}_mlp_pca_gdf_n_r{r}_s{s}'
            score = svm_gdf_res.train_mlp(
                get_m, feature_name=feature_name, method='mlp',
                fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
                compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam',
                                'metrics': [auc_roc, matthews_correlation, 'acc']},
                plot_name=plot_name, class_weight=weights)
            score = {**score, 'r': r, 's': s,
                     'epochs': epochs, 'batch_size': batch_size}
            score = {'solver': 'adam', 'hidden_layer_sizes': k,
                     'learning_rate': 0.001, **score}
            df_partial = df_partial.append(pd.DataFrame([score]), ignore_index=True)
            df_partial.index = list((range(len(df_partial))))
            df_partial.to_csv(filename_partial)
    df_partial.to_csv(filename)
    return True


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    pool = Pool(processes=5)
    stocks = stocks_numbers.chosen_stocks
    res = [pool.apply_async(main, [s, 0.01, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.5]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.01, 0.5]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.25, 0.25]) for s in stocks]
    print([r.get() for r in res])
