
import typing as tp
import unittest
import operator

import numpy as np  # type: ignore

# from hypothesis import strategies as st
from hypothesis import given  # type: ignore


from static_frame.test.property import strategies as sfst

from static_frame.test.test_case import TestCase

from static_frame import Series



class TestUnit(TestCase):


    @given(sfst.get_series())  # type: ignore
    def test_basic_attributes(self, s1: Series) -> None:

        self.assertEqual(s1.dtype, s1.values.dtype)
        # self.assertEqual(s1.shape, s1.shape)
        self.assertEqual(s1.ndim, 1)

        if s1.shape[0] > 0:
            self.assertTrue(s1.size > 0)
            self.assertTrue(s1.nbytes > 0)
