import logging
import os
import sys

import numpy as np
import pandas as pd

from lob_data_utils import gdf_pca
from lob_data_utils.keras_metrics import matthews_correlation, auc_roc

logger = logging.getLogger(__name__)

res_dir = 'res_gdf_pca_gru'


def get_model_func(unit, kernel_regularization=None):
    from keras.layers import Dense
    from keras.layers import GRU, regularizers
    from keras.models import Sequential

    def get_model():
        model = Sequential()
        if kernel_regularization and kernel_regularization != 0.0:
            model.add(GRU(
                unit, activation='tanh', kernel_regularizer=regularizers.l2(kernel_regularization)))
        else:
            model.add(GRU(unit, activation='tanh'))
        model.add(Dense(1, input_shape=(unit, 1), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc, matthews_correlation])
        return model

    return get_model


def train_gru(stock: str, r: float, s: float, data_length: int, units: list=None, n_steps: int=4,
              kernel_regularizations: list=None):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    gdf_dfs = gdf_pca.SvmGdfResults(
        str(stock), r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)

    weights = gdf_dfs.get_classes_weights()
    feature = 'pca_n_gdf_que'
    epochs = 50
    batch_size = 512

    filename = os.path.join(res_dir, f'res_gru_pca_n_one_layer_step{n_steps}_{stock}_len{data_length}_r{r}_s{s}.csv')
    partial_filename = filename + '_partial'

    df_partial = pd.DataFrame()
    if os.path.exists(partial_filename):
        df_partial = pd.read_csv(partial_filename)
        df_partial.drop(columns=[c for c in df_partial.columns if 'Unnamed' in c], inplace=True)

    for unit in units:
        unit_str = f'({unit}: tanh, 1)'
        for kernel_reg in kernel_regularizations:

            if np.any(df_partial):
                row = df_partial[df_partial['unit'] == unit_str][df_partial['kernel_reg'] == kernel_reg][df_partial['n_steps'] == n_steps]
                if np.any(row):
                    logger.info(f'Already calculated {stock} {unit_str} {kernel_reg}')
                    continue
            logger.info(f'Will train {stock} r{r} s{s} step{n_steps} {unit_str} {kernel_reg}')
            get_model = get_model_func(unit)
            plot_name = f'plot_gru/{stock}_one_layer_u{unit}_kr{kernel_reg}_pca_n_r{r}_s{s}'
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


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stock = str(sys.argv[1])
    r = float(sys.argv[2])
    s = float(sys.argv[3])
    data_length = int(sys.argv[4])
    n_steps = int(sys.argv[5])

    train_gru(
        stock, r, s, data_length, units=[4, 8, 16, 32], n_steps=n_steps,
        kernel_regularizations=[0.0, 0.1, 0.01, 0.001])


if __name__ == '__main__':
    main()
