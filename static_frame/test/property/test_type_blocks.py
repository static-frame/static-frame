

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


    @given(sfst.get_array_1d2d())
    def test_shape_filter(self, shape):
        self.assertTrue(len(TypeBlocks.shape_filter(shape)), 2)


    @given(sfst.get_type_blocks())
    def test_basic_attributes(self, tb):
        self.assertTrue(len(tb.dtypes), len(tb))
        self.assertTrue(len(tb.shapes), len(tb.mloc))


    @given(sfst.get_type_blocks())
    def test_values(self, tb):
        values = tb.values

        self.assertEqual(values.shape, tb.shape)
        self.assertEqual(values.dtype, tb._row_dtype)


    @given(sfst.get_type_blocks())
    def test_element_items(self, tb):
        count = 0
        for k, v in tb.element_items():
            count += 1
            # tb.iloc[list(k)]

        self.assertEqual(count, tb.size)
