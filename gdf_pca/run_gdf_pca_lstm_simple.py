import logging
import logging
import subprocess

from lob_data_utils import stocks_numbers

logger = logging.getLogger(__name__)


def main(stock, r, s):
    subprocess.run(['python', 'gdf_pca_lstm_simple.py', stock, str(r), str(s), str(24000)])


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    pool = Pool(processes=1)
    stocks = stocks_numbers.chosen_stocks
    stocks = ['9062']
    res = [pool.apply_async(main, [s, 0.01, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.5]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.1, 0.1]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.01, 0.5]) for s in stocks]
    print([r.get() for r in res])
    res = [pool.apply_async(main, [s, 0.25, 0.25]) for s in stocks]
    print([r.get() for r in res])
