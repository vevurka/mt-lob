import logging
import subprocess

from lob_data_utils import stocks_numbers

logger = logging.getLogger(__name__)


def main(stock, r, s, n_steps):
    subprocess.run(['python', 'gdf_pca_lstm.py', stock, str(r), str(s), str(24000), str(n_steps)])


if __name__ == '__main__':
    from multiprocessing import Pool
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    stocks = stocks_numbers.chosen_stocks
    for steps in [1, 8, 4, 16]:
        pool = Pool(processes=5)
        res = [pool.apply_async(main, [s, 0.01, 0.1, steps]) for s in stocks]
        print([r.get() for r in res])
        res = [pool.apply_async(main, [s, 0.1, 0.5, steps]) for s in stocks]
        print([r.get() for r in res])
        res = [pool.apply_async(main, [s, 0.1, 0.1, steps]) for s in stocks]
        print([r.get() for r in res])
        res = [pool.apply_async(main, [s, 0.01, 0.5, steps]) for s in stocks]
        print([r.get() for r in res])
        res = [pool.apply_async(main, [s, 0.25, 0.25, steps]) for s in stocks]
        print([r.get() for r in res])
