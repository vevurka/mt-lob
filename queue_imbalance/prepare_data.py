import logging

from lob_data_utils import lob, roc_results


def main(s):
    errors = []
    print('************************************', s)
    try:
        df, _ = lob.load_data(str(s), data_dir='data/INDEX/', include_test=False)
        # print(df.head())
        df.to_csv('data/prepared/' + str(s) + '.csv')
    except Exception as e:
        print(e)
        errors.append({'stock': s, 'error': e})
    print(errors)
    return errors


if __name__ == '__main__':
    from multiprocessing import Pool

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    pool = Pool(processes=3)
    stocks = list(roc_results.results.keys())
    res = [pool.apply_async(main, [s]) for s in stocks]
    print([r.get() for r in res])

