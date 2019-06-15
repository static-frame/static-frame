
import unittest
import os
import sys
from itertools import zip_longest
import typing as tp
import itertools as it
import string
import cmath

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
    def get_arrays_a() -> tp.Generator[np.ndarray, None , None]:
        '''
        Return sample array suitable for TypeBlock creation, testing. Unique values required.
        '''

        a1 = np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
        a1.flags.writeable = False

        a2 = np.array([[4], [5], [6]])
        a2.flags.writeable = False

        a3 = np.array([[None, 'a', None], ['q', 'x', 'c'], ['f', 'y', 'e']])
        a3.flags.writeable = False

        a4 = np.array([1.2, np.nan, 30.5])
        a4.flags.writeable = False

        for arrays in it.permutations((a1, a2, a3, a4)):
            yield arrays


    @staticmethod
    def get_arrays_b() -> tp.Generator[np.ndarray, None , None]:
        '''
        Return sample array suitable for TypeBlock creation, testing. Many NaNs.
        '''

        a1 = np.array([[1, 2, 3], [10, 20, 30], [100, 200, 300]])
        a1.flags.writeable = False

        a2 = np.array([[4], [5], [6]])
        a2.flags.writeable = False

        a3 = np.array([[None, 'a', None], [None, None, 'c'], ['f', None, 'e']])
        a3.flags.writeable = False

        a4 = np.array([np.nan, np.nan, np.nan])
        a4.flags.writeable = False

        for arrays in it.permutations((a1, a2, a3, a4)):
            yield arrays


    @staticmethod
    def get_letters(*slice_args) -> tp.Generator[str, None, None]:
        for letter in string.ascii_lowercase[slice(*slice_args)]:
            yield letter

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
            if (isinstance(v1, float) and np.isnan(v1) and
                    isinstance(v2, float) and np.isnan(v2)):
                continue
            if (isinstance(v1, complex) and cmath.isnan(v1) and
                    isinstance(v2, complex) and cmath.isnan(v2)):
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
