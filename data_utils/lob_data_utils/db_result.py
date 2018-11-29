import psycopg2

import psycopg2.extras


def get_svm_results_for_stock(stock, data_length=0, data_type='cv'):
    with psycopg2.connect("dbname=lob_results user=postgres password=postgres") as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''select * from resulter_result r join resulter_algorithm a on (r.algorithm_id = a.id)
                                                   join resulter_svm s on a.svm_id = s.id where
                                                        stock = %s and data_length = %s and data_type = %s''',
                (stock, data_length, data_type))
            return cur.fetchall()


def get_svm_results_for_data_length(data_length, data_type='cv'):
    with psycopg2.connect("dbname=lob_results user=postgres password=postgres") as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''select * from resulter_result r join resulter_algorithm a on (r.algorithm_id = a.id)
                                                   join resulter_svm s on a.svm_id = s.id where
                                                        data_length = %s and data_type = %s ''',
                (data_length, data_type))
            return cur.fetchall()


def get_svm_results_by_kernel(kernel, data_type='cv', data_length=0):
    with psycopg2.connect("dbname=lob_results user=postgres password=postgres") as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''select * from resulter_result r join resulter_algorithm a on (r.algorithm_id = a.id)
                                                   join resulter_svm s on a.svm_id = s.id where
                                                        s.kernel = %s and data_type = %s  and data_length = %s''',
                (kernel, data_type, data_length))
            return cur.fetchall()


def get_svm_results_by_params(stock, kernel, data_type='cv', data_length=0, gamma=-1, c=1, coef0=0):
    with psycopg2.connect("dbname=lob_results user=postgres password=postgres") as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                '''select * from resulter_result r join resulter_algorithm a on (r.algorithm_id = a.id)
                                                   join resulter_svm s on a.svm_id = s.id where
                                                        r.stock = %s and
                                                        s.kernel = %s and data_type = %s  and data_length = %s
                                                        and s.gamma = %s and s.c = %s and s.coef0 = %s''',
                (stock, kernel, data_type, data_length, gamma, c, coef0))
            return cur.fetchall()

