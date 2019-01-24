import logging
import os

import pandas as pd
from lob_data_utils import roc_results, gdf_pca
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


def main(stock, r=0.1, s=0.1):
    try:
        result_dir = 'res_mlp_pca_gdf_que_prev10_15000'
        data_length = 15000
        svm_gdf_res = gdf_pca.SvmGdfResults(
            stock, r=r, s=s, data_length=data_length,
            gdf_filename_pattern='gdf_{}_r{}_s{}_K50',
            data_dir='../gaussian_filter/data_gdf_whole/')
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
    except Exception as e:
        print(e)


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=2)
    stocks = list(roc_results.results_15000.keys())
    res = [pool.apply_async(main, [s, 0.01, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 1.0, 1.0]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 1.0, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 1.0]) for s in stocks]


