import logging
import os

import pandas as pd
from keras.models import model_from_json
from lob_data_utils import gdf_pca, stocks_numbers
from lob_data_utils.keras_metrics import matthews_correlation, auc_roc


res_dir = 'res_gdf_pca_lstm_mcnemar'


def get_model(arch):
    def _get_model():
        mod = model_from_json(arch)
        mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
        return mod
    return _get_model


def train_lstm(res):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    data_length = 24000
    r = res['r'].values[0]
    s = res['s'].values[0]
    feature = res['features'].values[0]
    n_steps = int(res['n_steps'].values[0])
    stock = str(int(res['stock'].values[0]))
    arch = res['arch'].values[0]

    gdf_dfs = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)
    weights = gdf_dfs.get_classes_weights()

    epochs = 50
    batch_size = 512

    filename = os.path.join(res_dir, f'pred_lstm_iter_{stock}_len{data_length}_r{r}_s{s}.csv')

    get_model_func = get_model(arch)
    s, m = gdf_dfs.train_lstm(
        get_model_func, feature_name=feature, should_return_model=True,
        fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
        should_validate=False,
        compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': [matthews_correlation, auc_roc]},
        class_weight=weights, n_steps=n_steps)
    test_x, test_y = gdf_dfs.get_test_set(feature_name=feature, n_steps=n_steps)
    pred = m.predict_classes(test_x)
    df_scores = pd.DataFrame()
    df_scores['pred'] = pred.ravel()
    df_scores['actual'] = test_y
    df_scores.to_csv(filename)
    return None


def get_best_results(data_dir):
    df_res = pd.DataFrame()
    for f in os.listdir(data_dir):
        if 'csv' in f:
            df_res = df_res.append(pd.read_csv(os.path.join(data_dir, f)))
    df_best = df_res.sort_values(by='matthews', ascending=False).groupby(['stock']).head(1)
    df_best.index = df_best['stock']
    return df_best


def main():
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    data_dir = 'res_gdf_pca_lstm/'
    df_best = get_best_results(data_dir)
    pool = Pool(processes=8)
    stocks = stocks_numbers.chosen_stocks
    res = [pool.apply_async(train_lstm, [df_best[df_best['stock'] == int(s)]]) for s in stocks]
    print([r.get() for r in res])


if __name__ == '__main__':
    main()
