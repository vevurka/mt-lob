import functools
import json
import logging
import os

import pandas as pd
import sklearn


from lob_data_utils import gdf_pca, stocks_numbers
from numpy.random import seed

import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
from keras.utils import plot_model


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
seed(1)


def train_lstm(res):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    data_length = 24000
    r = res['r'].values[0]
    s = res['s'].values[0]
    feature = res['features'].values[0]
    n_steps = int(res['n_steps'].values[0])
    stock = str(int(res['stock'].values[0]))
    arch = res['arch'].values[0]
    print(stock, n_steps, feature)
    gdf_dfs = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)
    weights = gdf_dfs.get_classes_weights()

    epochs = 25
    batch_size = 300

    filename = os.path.join('predictions', f'pred_lstm_best_{stock}_len{data_length}_r{r}_s{s}.csv')
    if os.path.exists(filename):
        print(f'Exists {filename}.')
        return None

    mod = model_from_json(arch)
    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc])
    plot_model(mod, to_file=f'plot_lstm_best/{stock}_model_best_r{r}_s{s}.png')
    plot_name = f'plot_lstm_best/{stock}_pred_r{r}_s{s}'

    gdf_dfs.train_lstm(
        mod, feature_name=feature,
        fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
        should_validate=False,
        compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': [auc_roc]},
        plot_name=plot_name, class_weight=weights, n_steps=n_steps)
    test_x, test_y = gdf_dfs.get_test_set(feature_name=feature, n_steps=n_steps)
    pred = mod.predict_classes(test_x)
    df_scores = pd.DataFrame()
    df_scores['pred'] = pred.ravel()
    df_scores['actual'] = test_y
    df_scores.to_csv(filename)
    return None


def get_best_results(data_dir):
    df_res = pd.DataFrame()
    for f in os.listdir(data_dir):
        df_res = df_res.append(pd.read_csv(os.path.join(data_dir, f)))
    df_best = df_res.sort_values(by='matthews', ascending=False).groupby(['stock']).head(1)
    df_best.index = df_best['stock']
    return df_best


def main():
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    data_dir = 'res_lstm_weird/'
    stocks = stocks_numbers.chosen_stocks
    df_best = get_best_results(data_dir)
    stocks2 = ['11869', '4799', '9268']# , '9761', '9268', '9062']
    train_lstm(df_best[df_best['stock'] == int('11869')])
    pool = Pool(processes=8)
   # res = [pool.apply_async(train_lstm, [df_best[df_best['stock'] == int(s)]]) for s in stocks2]
   # print([r.get() for r in res])
    res = [pool.apply_async(train_lstm, [df_best[df_best['stock'] == int(s)]]) for s in stocks]
    print([r.get() for r in res])


if __name__ == '__main__':
    main()
