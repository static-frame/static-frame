
import unittest

import numpy as np

from hypothesis import strategies as st
from hypothesis import given
from hypothesis import example
from hypothesis import reproduce_failure

from static_frame.test.property import strategies as sfst

from static_frame.test.test_case import TestCase

from static_frame import TypeBlocks
from static_frame import Index

from static_frame import IndexDate
from static_frame import IndexYear
from static_frame import IndexYearMonth
from static_frame import IndexSecond
from static_frame import IndexMillisecond

from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO

from static_frame import IndexGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO


class TestUnit(TestCase):


    @given(sfst.get_labels())
    def test_get_labels(self, values):
        for value in values:
            self.assertTrue(isinstance(hash(value), int))

    @given(sfst.get_dtypes())
    def test_get_dtypes(self, dtypes):
        for dt in dtypes:
            self.assertTrue(isinstance(dt, np.dtype))

    @given(sfst.get_spacing(10))
    def test_get_spacing_10(self, spacing):
        self.assertEqual(sum(spacing), 10)

    @given(sfst.get_shape_1d2d())
    def test_get_shape_1d2d(self, shape):
        self.assertTrue(isinstance(shape, tuple))
        self.assertTrue(len(shape) in (1, 2))

    @given(sfst.get_array_1d2d())
    def test_get_array_1d2d(self, array):
        self.assertTrue(isinstance(array, np.ndarray))
        self.assertTrue(array.ndim in (1, 2))

    @given(sfst.get_blocks())
    def test_get_blocks(self, blocks):
        self.assertTrue(isinstance(blocks, tuple))
        for b in blocks:
            self.assertTrue(isinstance(b, np.ndarray))
            self.assertTrue(b.ndim in (1, 2))

    @given(sfst.get_type_blocks())
    def test_get_type_blocks(self, tb):
        self.assertTrue(isinstance(tb, TypeBlocks))
        rows, cols = tb.shape
        col_count = 0
        for b in tb._blocks:
            if b.ndim == 1:
                self.assertEqual(len(b), rows)
                col_count += 1
            else:
                self.assertEqual(b.ndim, 2)
                self.assertEqual(b.shape[0], rows)
                col_count += b.shape[1]

        self.assertEqual(col_count, cols)



if __name__ == '__main__':
    unittest.main()
