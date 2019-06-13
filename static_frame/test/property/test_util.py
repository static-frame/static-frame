
import unittest

import numpy as np

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example
from hypothesis import reproduce_failure

from static_frame.test.property.strategies import get_array_1d
from static_frame.test.property.strategies import get_array_1d2d
from static_frame.test.property.strategies import get_dtype_pairs

from static_frame.test.test_case import TestCase
from static_frame.core import util


class TestUnit(TestCase):


    @given(get_array_1d2d())
    def test_mloc(self, array):

        x = util.mloc(array)
        self.assertTrue(isinstance(x, int))


    @given(get_dtype_pairs())
    def test_resolve_dtype(self, dtype_pair):

        x = util.resolve_dtype(*dtype_pair)
        self.assertTrue(isinstance(x, np.dtype))



if __name__ == '__main__':
    unittest.main()
