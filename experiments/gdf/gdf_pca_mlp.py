import logging
import os
import sys

import numpy as np
import pandas as pd
from keras import Sequential
from keras import regularizers
from keras.layers import Dense
from lob_data_utils import gdf_pca
from lob_data_utils.keras_metrics import auc_roc, matthews_correlation

logger = logging.getLogger(__name__)

res_dir = 'res_gdf_pca_mlp'


def get_model31():
    model = Sequential()
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model32():
    model = Sequential()
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model33():
    model = Sequential()
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model34():
    model = Sequential()
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model35():
    model = Sequential()
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model21():
    model = Sequential()
    model.add(Dense(16, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model22():
    model = Sequential()
    model.add(Dense(4, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model23():
    model = Sequential()
    model.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(4, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model24():
    model = Sequential()
    model.add(Dense(16, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(4, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model25():
    model = Sequential()
    model.add(Dense(8, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model11():  # 10470
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model12():
    model = Sequential()
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model13():
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model14():
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model15():
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model1():
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model2():
    model = Sequential()
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model3():
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model4():
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(4, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


def get_model5():
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation, auc_roc])
    return model


models = {
    '(16: kernel_l2(0.01), 8: kernel_l2(0.01), 1)': get_model1,
    '(4: kernel_l2(0.01), 2: kernel_l2(0.01), 1)': get_model2,
    '(8: kernel_l2(0.01), 4: kernel_l2(0.01), 2: kernel_l2(0.01), 1)': get_model3,
    '(16: kernel_l2(0.01), 8: kernel_l2(0.01), 4: kernel_l2(0.01), 1)': get_model4,
    '(8: kernel_l2(0.01), 1)': get_model5,
    '(16: kernel_l2(0.001), 8: kernel_l2(0.001), 1)': get_model11,
    '(4: kernel_l2(0.001), 2: kernel_l2(0.001), 1)': get_model12,
    '(8: kernel_l2(0.001), 4: kernel_l2(0.001), 2: kernel_l2(0.001), 1)': get_model13,
    '(16: kernel_l2(0.001), 8: kernel_l2(0.001), 4: kernel_l2(0.001), 1)': get_model14,
    '(8: kernel_l2(0.001), 1)': get_model15,
    '(16: tanh;kernel_l2(0.001), 8: tanh;kernel_l2(0.001), 1)': get_model21,
    '(4: tanh;kernel_l2(0.001), 2: tanh;kernel_l2(0.001), 1)': get_model22,
    '(8: tanh;kernel_l2(0.001), 4: tanh;kernel_l2(0.001), 2: tanh;kernel_l2(0.001), 1)': get_model23,
    '(16: tanh;kernel_l2(0.001), 8: tanh;kernel_l2(0.001), 4: tanh;kernel_l2(0.001), 1)': get_model24,
    '(8: tanh;kernel_l2(0.001), 1)': get_model25,
    '(16: tanh, 8: tanh, 1)': get_model31,
    '(4: tanh, 2: tanh, 1)': get_model32,
    '(8: tanh, 4: tanh, 2: tanh, 1)': get_model33,
    '(16: tanh, 8: tanh, 4: tanh, 1)': get_model34,
    '(8: tanh, 1)': get_model35,
}


def main(stock, r=0.1, s=0.1):
    data_length = 24000
    svm_gdf_res = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern='gdf_{}_r{}_s{}_K50')
    feature_name = 'pca_n_gdf_que'

    weights = svm_gdf_res.get_classes_weights()
    epochs = 50
    batch_size = 512
    filename = os.path.join(res_dir, 'mlp_pca_gdf_n_{}_len{}_r{}_s{}.csv'.format(stock, data_length, r, s))
    filename_partial = os.path.join(
        res_dir, 'mlp_pca_n_gdf_{}_len{}_r{}_s{}.csv_partial'.format(stock, data_length, r, s))
    df_partial = pd.DataFrame()
    if os.path.exists(filename_partial):
        logger.info(f'Reading partial file {filename_partial}')
        df_partial = pd.read_csv(filename_partial)
        df_partial.drop(columns=[c for c in df_partial.columns if 'Unnamed' in c], inplace=True)
    for k, get_m in models.items():
        if np.any(df_partial):
            row = df_partial[df_partial['hidden_layer_sizes'] == k]
            if np.any(row):
                logger.info(f'Read result for hidden layer {k} in {filename_partial}')
                continue
        logger.info(f'Training {stock} {r} {s} {k}')

        plot_name = f'plot_mlp/{stock}_mlp_pca_gdf_n_r{r}_s{s}'
        score = svm_gdf_res.train_mlp(
            get_m, feature_name=feature_name, method='mlp',
            fit_kwargs={'epochs': epochs, 'batch_size': batch_size, 'verbose': 0, 'shuffle': False},
            compile_kwargs={
                'loss': 'binary_crossentropy', 'optimizer': 'adam',
                'metrics': [auc_roc, matthews_correlation]},
            plot_name=plot_name, class_weight=weights)
        score = {**score, 'r': r, 's': s,
                 'epochs': epochs, 'batch_size': batch_size, 'solver': 'adam', 'hidden_layer_sizes': k,
                 'learning_rate': 0.001}
        df_partial = df_partial.append(pd.DataFrame([score]), ignore_index=True)
        df_partial.index = list((range(len(df_partial))))
        df_partial.to_csv(filename_partial)
    df_partial.to_csv(filename)
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stock = str(int(sys.argv[1]))
    r = float(sys.argv[2])
    s = float(sys.argv[3])
    main(stock, r, s)
