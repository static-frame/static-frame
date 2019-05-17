
import unittest
import os
import sys
from itertools import zip_longest


import numpy as np
import pytest


from static_frame import TypeBlocks


skip_win = pytest.mark.skipif(sys.platform == 'win32', reason='Windows default dtypes.')

class TestCase(unittest.TestCase):
    '''
    TestCase specialized for usage with StaticFrame
    '''

    def setUp(self):
        pass

    @staticmethod
    def get_test_input(file_name: str):
        # input dir should be a sibling of this module
        fp_module = os.path.join(os.getcwd(), __file__)
        fp = os.path.join(os.path.dirname(fp_module), 'input', file_name)
        if not os.path.isfile(fp):
            raise RuntimeError('file not found', fp)
        return fp


    def assertTypeBlocksArrayEqual(self, tb: TypeBlocks, match, match_dtype=None):
        '''
        Args:
            tb: a TypeBlocks instance
            match: can be anything that can be used to create an array.
        '''
        # could use np.testing
        if not isinstance(match, np.ndarray):
            match = np.array(match, dtype=match_dtype)
        self.assertTrue((tb.values == match).all())


    def assertAlmostEqualValues(self, values1, values2):
        for v1, v2 in zip_longest(values1, values2):
            if isinstance(v1, float) and np.isnan(v1) and isinstance(v2, float) and np.isnan(v2):
                continue
            self.assertEqual(v1, v2)

    def assertAlmostEqualItems(self, pairs1, pairs2):
        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)

            if isinstance(v1, float) and np.isnan(v1) and isinstance(v2, float) and np.isnan(v2):
                continue

            self.assertEqual(v1, v2)


    def assertAlmostEqualFramePairs(self, pairs1, pairs2):
        '''
        For comparing nested tuples returned by Frame.to_pairs()
        '''
        for (k1, v1), (k2, v2) in zip_longest(pairs1, pairs2):
            self.assertEqual(k1, k2)
            self.assertAlmostEqualItems(v1, v2)
