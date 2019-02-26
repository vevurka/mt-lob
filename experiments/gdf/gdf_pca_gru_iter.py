import logging
import os
import sys

import pandas as pd
from lob_data_utils import gdf_pca
from lob_data_utils.keras_metrics import matthews_correlation, auc_roc

logger = logging.getLogger(__name__)

res_dir = 'res_gdf_pca_gru_iter'


def get_gru_model_for_arch(arch):
    def get_model():
        from keras.models import model_from_json
        m = model_from_json(arch)
        m.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
        return m
    return get_model


def train_gru(res):

    data_length = 24000
    r = res['r'].values[0]
    s = res['s'].values[0]
    feature = res['features'].values[0]
    n_steps = int(res['n_steps'].values[0])
    unit = res['unit'].values[0]
    stock = str(int(res['stock'].values[0]))
    arch = res['arch'].values[0]

    gdf_dfs = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern='gdf_{}_r{}_s{}_K50')
    weights = gdf_dfs.get_classes_weights()

    epochs = 50
    batch_size = 512

    filename = os.path.join(res_dir, f'res_gru_iter_{stock}_len{data_length}_r{r}_s{s}.csv')
    partial_filename = filename + '_partial'

    df_partial = pd.DataFrame()
    if os.path.exists(partial_filename):
        df_partial = pd.read_csv(partial_filename)

    if len(df_partial) < 30:
        logger.info('Iteration %s stock %s', len(df_partial), stock)
        get_model = get_gru_model_for_arch(arch)
        try:
            score = gdf_dfs.train_lstm(
                get_model, feature_name=feature,
                fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
                compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam',
                                'metrics': [matthews_correlation, auc_roc]}, class_weight=weights, n_steps=n_steps)
            score = {**score, 'r': r, 's': s, 'unit': unit, 'arch': arch,
                     'epochs': epochs, 'batch_size': batch_size, 'n_steps': n_steps}
            df_partial = df_partial.append([score])
            df_partial.to_csv(partial_filename)
            logger.info('Done %s stock %s', len(df_partial), stock)
        except Exception as e:
            logger.error('%s: iter %s  %s', stock, len(df_partial), e)
            raise Exception(stock, e)
    df_partial.to_csv(filename)
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
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    data_dir = 'res_gdf_pca_gru'
    df_best = get_best_results(data_dir)

    s = sys.argv[1]
    logger.info('Stock %s', s)
    train_gru(df_best[df_best['stock'] == int(s)])


if __name__ == '__main__':
    main()
