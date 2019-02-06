import functools
import logging
import os

import pandas as pd
from keras.layers import Dense
from keras import backend as K
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


def main(stock, r=0.1, s=0.1):
    result_dir = 'res_mlp_pca'

    data_length = 24000
    svm_gdf_res = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern='gdf_{}_r{}_s{}_K50')
    feature_name = 'pca_n_gdf_que'
    n = svm_gdf_res.get_pca(feature_name).n_components
    hidden_layer_sizes = [(n,), (n, n), (2 * n, n), (2 * n, 2 * n), (n, 2 * n), (n, n, n)]

    weights = svm_gdf_res.get_classes_weights()
    epochs = 10
    batch_size = 300
    filename = os.path.join(result_dir, 'mlp_pca_gdf_{}_len{}_r{}_s{}.csv'.format(stock, data_length, r, s))
    if os.path.exists(filename):
        print(f'Exists {filename}')
        return
    filename_partial = os.path.join(result_dir, 'mlp_pca_gdf_{}_len{}_r{}_s{}.csv_partial'.format(stock, data_length, r, s))
    df_partial = pd.DataFrame()
    if os.path.exists(filename_partial):
        print(f'Reading partial file {filename_partial}')
        df_partial = pd.read_csv(filename_partial)
    for hidden_layer_size in hidden_layer_sizes:
        for learning_rate in [0.001]: #[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
            if np.any(df_partial):

                print(filename_partial)
                row = df_partial[df_partial['hidden_layer_sizes'] == hidden_layer_size]
                if np.any(row) and len(row) >= 1:
                    print(row)
                    row = df_partial[df_partial['hidden_layer_sizes'] == hidden_layer_size][df_partial['learning_rate'] == learning_rate]
                    print(row)
                    if np.any(row):
                        print(f'Read result for hidden layer {hidden_layer_size} lr {learning_rate} in {filename_partial}')
                        continue
            print(f'Training {stock} {r} {s} {hidden_layer_size} {learning_rate}')
            solver = optimizers.Adam(lr=learning_rate)
            model = Sequential()
            if isinstance(hidden_layer_size, int):
                model.add(Dense(hidden_layer_size))
            else:
                for h in hidden_layer_size:
                    model.add(Dense(h))
            model.add(Dense(1, activation='sigmoid'))

            plot_name = f'plot_mlp/{stock}_mlp_pca_n_r{r}_s{s}'
            score = svm_gdf_res.train_mlp(
                model, feature_name=feature_name, method='mlp',
                fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
                compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': solver,
                                'metrics': [auc_roc, 'acc']},
                plot_name=plot_name, class_weight=weights)
            score = {**score, 'r': r, 's': s, 'arch': model.to_json(),
                     'epochs': epochs, 'batch_size': batch_size}
            score = {'solver': solver, 'hidden_layer_sizes': hidden_layer_size,
                     'learning_rate': learning_rate, **score}
            df_partial = df_partial.append(pd.DataFrame([score]), ignore_index=True)
            df_partial.index = list((range(len(df_partial))))
          #  df_partial.drop(columns=[[c for c in df_partial.columns if 'Unnamed' in c]], inplace=True)
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
