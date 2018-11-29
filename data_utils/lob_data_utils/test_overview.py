import unittest

import pandas as pd
from lob_data_utils.overview import Overview


class Test(unittest.TestCase):
    def test_is_in_results_for_one_key_value(self):
        overview = Overview('1', 2)
        df = pd.DataFrame()
        df['a'] = [1]
        self.assertTrue(overview.is_in_results(df, {'a': 1}))

    def test_is_in_results_for_more_key_value(self):
        overview = Overview('1', 2)
        df = pd.DataFrame()
        df['a'] = [1, 2, 3]
        df['b'] = [1, 2, 3]
        self.assertTrue(overview.is_in_results(df, {'a': 1, 'b': 1}))
        self.assertFalse(overview.is_in_results(df, {'a': 1, 'b': 2}))

    def test_is_in_results_for_more_key_value_strings(self):
        overview = Overview('1', 2)
        df = pd.DataFrame()
        df['a'] = [1, 2, 3]
        df['b'] = ['str1', 'str2', 'str3']
        self.assertTrue(overview.is_in_results(df, {'a': 1, 'b': 'str1'}))
        self.assertFalse(overview.is_in_results(df, {'a': 1, 'b': 'str2'}))
