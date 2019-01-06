import logging
import os

import pandas as pd
from lob_data_utils import roc_results, gdf_pca
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


def main(stock, r=0.1, s=0.1):
    result_dir = 'res_mlp_pca_gdf_que_prev10'
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


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=5)
    stocks = list(roc_results.results_10000.keys())
    res = [pool.apply_async(main, [s, 0.01, 0.1]) for s in stocks]
    print([r.get() for r in res])
    # res = [pool.apply_async(main, [s, 0.1, 1.0]) for s in stocks]
    # print([r.get() for r in res])
    # res = [pool.apply_async(main, [s, 0.1, 0.1]) for s in stocks]
    # print([r.get() for r in res])
    # res = [pool.apply_async(main, [s, 1.0, 1.0]) for s in stocks]
    # print([r.get() for r in res])
    # res = [pool.apply_async(main, [s, 1.0, 0.1]) for s in stocks]
    # print([r.get() for r in res])


