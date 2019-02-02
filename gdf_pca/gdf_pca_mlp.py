import logging
import os

import pandas as pd
from lob_data_utils import gdf_pca, stocks_numbers
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


def main(stock, r=0.1, s=0.1):
    result_dir = 'res_mlp_pca'
    data_length = 24000
    svm_gdf_res = gdf_pca.SvmGdfResults(
        stock, r=r, s=s, data_length=data_length, gdf_filename_pattern='gdf_{}_r{}_s{}_K50')
    results = []
    feature_name = 'pca_n_gdf_que_prev'
    n = svm_gdf_res.get_pca(feature_name).n_components
    hidden_layer_sizes = [n, (n, n), (2*n, n), (2*n, 2*n), (n, 2*n)]

    weights = svm_gdf_res.get_classes_weights()

    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        for hidden_layer_size in hidden_layer_sizes:
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
                scores = svm_gdf_res.train_clf(clf, feature_name=feature_name, method='mlp', class_weight=weights)
                results.append({'alpha': alpha, 'solver': solver,
                                'hidden_layer_sizes': hidden_layer_size,
                                'learning_rate': learning_rate, **scores})
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
    res = [pool.apply_async(main, [s, 0.01, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.5]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.01, 0.5]) for s in stocks]
    print([r.get() for r in res])


