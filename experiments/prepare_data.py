import logging

from lob_data_utils import lob, stocks_numbers


def main(s):
    errors = []
    df, _ = lob.load_data(str(s), include_test=False)
    df.to_csv('data/prepared/' + str(s) + '.csv')

    return errors


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=3)
    stocks = stocks_numbers.chosen_stocks
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])

