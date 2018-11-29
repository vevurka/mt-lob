import unittest
from collections import OrderedDict

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from states import add_state, get_state_number, unhash_state, get_state_number_for_labels


class TestStates(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame()
        self.df['bid_price'] = [0, 2, 1, 1, 3]
        self.df['ask_price'] = [10, 12, 13, 11, 11]
        self.df['sum_sell_ask'] = [200, 202, 201, 200, 200]
        self.df['sum_buy_bid'] = [100, 102, 102, 101, 100]
        self.df['spread'] = self.df['ask_price'] - self.df['bid_price']
        self.df['queue_imbalance'] = (self.df['sum_buy_bid'] - self.df['sum_sell_ask']) / \
                                     (self.df['sum_buy_bid'] + self.df['sum_sell_ask'])
        self.df['mid_price'] = (self.df['ask_price'] - self.df['bid_price']) / 2

    def test_state_bid(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_bid'].values, [np.nan, 1, -1, 0, 1])

    def test_state_ask(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_ask'].values, [np.nan, 1, 1, -1, 0])

    def test_state_sum_ask(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_sum_ask'].values, [np.nan, 1, -1, -1, 0])

    def test_state_sum_bid(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_sum_bid'].values, [np.nan, 1, 0, -1, -1])

    def test_state_spread(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_spread'].values, [np.nan, 0, 1, -1, -1])

    def test_state_imbalance(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_imbalance'].values, [np.nan, 1, 1, -1, -1])

    def test_state_mid(self):
        test_df = add_state(self.df)
        assert_array_almost_equal(test_df['state_mid'].values, [np.nan, 0, 1, -1, -1])

    def test_state_number(self):
        df = add_state(self.df)
        state = get_state_number(df, 1)

        self.assertEqual(state, 2078)

        state = get_state_number(df, 2)
        self.assertEqual(state, 965)

        state = get_state_number(df, 3)
        self.assertEqual(state, 3)

        state = get_state_number(df, 4)
        self.assertEqual(state, 250)

    def test_unhash_for_2(self):
        labels = ['a', 'b']
        state = unhash_state(6, labels)
        state_hash = get_state_number_for_labels({'a': -1, 'b': 1})
        self.assertEqual(unhash_state(state_hash, labels), state)

    def test_unhash_for_3_states(self):
        labels = ['a', 'b', 'c']
        values = {'a': 0, 'b': -1, 'c': 1}
        state_hash = get_state_number_for_labels(OrderedDict(values))
        unhashed_state = unhash_state(state_hash, labels)
        self.assertEqual(unhashed_state, values)

        values = [-1, 0, 1]
        for i in values:
            for j in values:
                for k in values:
                    state = {'a': i, 'b': j, 'c': k}
                    state_hash = get_state_number_for_labels(OrderedDict(state))
                    unhashed_state = unhash_state(state_hash, labels)
                    self.assertEqual(unhashed_state, state)

    def test_unhash(self):
       self.assertEqual(
           unhash_state(181, sorted([c for c in self.df.columns if 'state' in c])), 1)