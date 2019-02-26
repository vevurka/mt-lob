import logging
import subprocess

from lob_data_utils import stocks_numbers

logger = logging.getLogger(__name__)


def main(stock):
    subprocess.run(['python', 'gdf_pca_gru_iter.py', stock])


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    pool = Pool(processes=3)
    stocks = stocks_numbers.chosen_stocks
    for i in range(0, 30):
        res = [pool.apply_async(main, [s]) for s in stocks]
        print([r.get() for r in res])