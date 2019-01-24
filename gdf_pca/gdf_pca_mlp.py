import logging
import os

import pandas as pd
from lob_data_utils import gdf_pca, stocks_numbers
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


def main_old(stock, r=0.1, s=0.1):
    result_dir = 'res_mlp_pca'
    data_length = 10000
    svm_gdf_res = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length,
        gdf_filename_pattern='gdf_{}_' + 'len{}'.format(data_length) + '_r{}_s{}_K50')
    results = []
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]:
        for hidden_layer_size in [(8, 16), (16, 16), (16, 8), (8, 8)]:
            activation = 'tanh'
            solver = 'adam'
            clf = MLPClassifier(
                solver=solver, alpha=alpha, activation=activation, hidden_layer_sizes=hidden_layer_size, random_state=1)
            scores = svm_gdf_res.train_clf(clf, feature_name='pca_gdf_que_prev10', method='mlp')
            results.append({**scores, 'alpha': alpha, 'solver': solver, 'hidden_layer_sizes': hidden_layer_size})
    pd.DataFrame(results).to_csv(
        os.path.join(result_dir, 'mlp_pca_gdf_{}_len{}_r{}_s{}.csv'.format(stock, data_length, r, s)))


def main(stock, r=0.1, s=0.1):
    result_dir = 'res_mlp_pca'
    data_length = 24000
    svm_gdf_res = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, data_dir='../gaussian_filter/data_gdf_not_synced/',
        gdf_filename_pattern='gdf_{}_r{}_s{}_K50')
    results = []
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        for hidden_layer_size in [(8, 16), (16, 16), (16, 8), (8, 8), (20, 20), (20, 16, 8), (20, 20, 20)]:
            for learning_rate in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]:
                activation = 'tanh'
                solver = 'adam'
                clf = MLPClassifier(
                    learning_rate_init=learning_rate,
                    solver=solver,
                    alpha=alpha,
                    activation=activation,
                    hidden_layer_sizes=hidden_layer_size,
                    random_state=1)
                scores = svm_gdf_res.train_clf(clf, feature_name='pca_n_gdf_que_prev', method='mlp')
                results.append({**scores, 'alpha': alpha, 'solver': solver,
                                'hidden_layer_sizes': hidden_layer_size,
                                'learning_rate': learning_rate})
        pd.DataFrame(results).to_csv(
            os.path.join(result_dir, 'mlp_pca_gdf_{}_len{}_r{}_s{}.csv_partial'.format(stock, data_length, r, s)))
    pd.DataFrame(results).to_csv(
        os.path.join(result_dir, 'mlp_pca_gdf_{}_len{}_r{}_s{}.csv'.format(stock, data_length, r, s)))
    return True


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    pool = Pool(processes=5)
    stocks = stocks_numbers.chosen_stocks
    stocks = ['9761']
    # res = [pool.apply_async(main, [s, 0.01, 0.1]) for s in stocks]
    # print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.5]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.01, 0.5]) for s in stocks]
    print([r.get() for r in res])


