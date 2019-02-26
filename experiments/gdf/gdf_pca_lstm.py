import logging
import os
import sys

import numpy as np
import pandas as pd
from lob_data_utils import gdf_pca
from lob_data_utils.keras_metrics import matthews_correlation, auc_roc

logger = logging.getLogger(__name__)
res_dir = 'res_gdf_pca_lstm'
plot_dir = 'plot_lstm'


def get_model_func(unit, input_shape=None, kernel_regularization=None):
    from keras.layers import Dense
    from keras.layers import LSTM, regularizers
    from keras.models import Sequential

    def get_model():
        model = Sequential()
        if kernel_regularization and kernel_regularization != 0.0:
            model.add(LSTM(unit, input_shape=input_shape, activation='tanh',
                           kernel_regularizer=regularizers.l2(kernel_regularization)))
        else:
            model.add(LSTM(unit, input_shape=input_shape, activation='tanh'))
        model.add(Dense(1, input_shape=(unit, 1), activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc_roc, matthews_correlation])
        return model

    return get_model


def train_lstm(stock: str, r: float, s: float, data_length: int, n_steps: int, units: list=None, kernel_regularizations: list=None):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    gdf_dfs = gdf_pca.SvmGdfResults(
        str(stock), r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)

    weights = gdf_dfs.get_classes_weights()
    feature = 'pca_n_gdf_que'
    epochs = 50
    batch_size = 512

    filename = os.path.join(res_dir, f'res_lstm_pca_n_one_layer_{stock}_len{data_length}_r{r}_s{s}.csv')
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
            plot_name = os.path.join(plot_dir, f'{stock}_one_layer_u{unit}_kr{kernel_reg}_pca_n_r{r}_s{s}')
            score = gdf_dfs.train_lstm(
                get_model, feature_name=feature,
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


def get_specific_model_func(input_shape=None):
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM, regularizers
    from keras.models import Sequential

    def get_model0():
        model = Sequential()
        model.add(LSTM(16, activation='relu', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
        return model

    def get_model1():
        model = Sequential()
        model.add(LSTM(16, activation='tanh', input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(8, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
        return model
    return {
        '(LSTM: 16;relu, DROPOUT: 0.25, LSTM: 8;relu;kr0.001, DROPOUT: 0.25, 1: sigmoid)': get_model0,
        '(LSTM: 16;tanh, DROPOUT: 0.25, LSTM: 8;tanh;kr0.001, DROPOUT: 0.25, 1: sigmoid)': get_model1,
    }


def train_lstm_specific_model(stock: str, r: float, s: float, data_length: int):
    gdf_filename_pattern = 'gdf_{}_r{}_s{}_K50'
    gdf_dfs = gdf_pca.SvmGdfResults(
        str(stock), r=r, s=s, data_length=data_length, gdf_filename_pattern=gdf_filename_pattern)

    weights = gdf_dfs.get_classes_weights()
    feature = 'pca_n_gdf_que'
    epochs = 50
    batch_size = 512
    n_steps = 12

    filename = os.path.join(res_dir, f'res_lstm_pca_n_{stock}_len{data_length}_r{r}_s{s}.csv')
    partial_filename = filename + '_partial'

    df_partial = pd.DataFrame()
    if os.path.exists(partial_filename):
        df_partial = pd.read_csv(partial_filename)
        df_partial.drop(columns=[c for c in df_partial.columns if 'Unnamed' in c], inplace=True)
    pca = gdf_dfs.get_pca(feature)
    for model_name, model_to_run in get_specific_model_func(input_shape=(n_steps, pca.n_components_)).items():
        if np.any(df_partial):
            row = df_partial[df_partial['unit'] == model_name]
            if np.any(row):
                if row['n_steps'] == n_steps:
                    print(row)
                    print(f'Already calculated {stock} {model_name}')
                    continue
        logger.info(f'Will train {stock} r{r} s{s} {model_name}')
        plot_name = os.path.join(plot_dir, f'{stock}_func_{model_to_run.__name__}_pca_n_r{r}_s{s}')
        score = gdf_dfs.train_lstm(
            model_to_run, feature_name=feature,
            fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
            compile_kwargs={'loss': 'binary_crossentropy', 'optimizer': 'adam',
                            'metrics': [matthews_correlation, auc_roc]},
            plot_name=plot_name, class_weight=weights, n_steps=n_steps)
        score = {**score, 'r': r, 's': s, 'unit': model_name, 'kernel_reg': np.nan,
                 'epochs': epochs, 'batch_size': batch_size, 'n_steps': n_steps}
        df_partial = df_partial.append(pd.DataFrame([score]), ignore_index=True)
        df_partial.to_csv(partial_filename)
    df_partial.drop(columns=[c for c in df_partial.columns if 'Unnamed' in c], inplace=True)
    df_partial.to_csv(filename)
    return True


def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stock = str(int(sys.argv[1]))
    r = float(sys.argv[2])
    s = float(sys.argv[3])
    data_length = int(sys.argv[4])
    n_steps = int(sys.argv[5])

    units = [4, 8, 16, 32]
    kernel_regularizations = [0.0, 0.001, 0.0001, 0.01]
    train_lstm(stock, r, s, data_length, n_steps, units=units, kernel_regularizations=kernel_regularizations)


if __name__ == '__main__':
    main()

