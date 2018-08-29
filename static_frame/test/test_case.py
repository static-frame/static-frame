
import unittest
from itertools import zip_longest


import numpy as np

from static_frame import TypeBlocks


class TestCase(unittest.TestCase):
    '''
    TestCase specialized for usage with StaticFrame
    '''

    def setUp(self):
        pass

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
