
import typing as tp
import unittest

import numpy as np  # type: ignore

from hypothesis import strategies as st
from hypothesis import given  # type: ignore
from hypothesis import example  # type: ignore
from hypothesis import reproduce_failure  # type: ignore

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


    @given(sfst.get_array_1d2d())  # type: ignore
    def test_shape_filter(self, shape: np.ndarray) -> None:
        self.assertTrue(len(TypeBlocks.shape_filter(shape)), 2)


    @given(sfst.get_type_blocks())  # type: ignore
    def test_basic_attributes(self, tb: TypeBlocks) -> None:
        self.assertTrue(len(tb.dtypes), len(tb))
        self.assertTrue(len(tb.shapes), len(tb.mloc))


    @given(sfst.get_type_blocks())  # type: ignore
    def test_values(self, tb: TypeBlocks) -> None:
        values = tb.values
        self.assertEqual(values.shape, tb.shape)
        self.assertEqual(values.dtype, tb._row_dtype)


    @given(sfst.get_type_blocks())  # type: ignore
    def test_element_items(self, tb: TypeBlocks) -> None:

        # NOTE: this found a flaw in _extract_iloc where we tried to optimize selection with a unified array

        count = 0
        for k, v in tb.element_items():
            count += 1
            v_extract = tb.iloc[k]

            self.assertEqualWithNaN (v, v_extract)

        self.assertEqual(count, tb.size)


    @given(sfst.get_type_blocks_aligned_array())  # type: ignore
    def test_append(self, tb_aligned_array: tp.Tuple[TypeBlocks, np.ndarray]) -> None:
        tb, aa = tb_aligned_array
        shape_original = tb.shape
        tb.append(aa)
        if aa.ndim == 1:
            self.assertEqual(tb.shape[1], shape_original[1] + 1)
        else:
            self.assertEqual(tb.shape[1], shape_original[1] + aa.shape[1])

    @given(sfst.get_type_blocks_aligned_type_blocks(min_size=2, max_size=2))  # type: ignore
    def test_extend(self, tbs: tp.List[TypeBlocks]) -> None:
        front = tbs[0]
        back = tbs[1]
        shape_original = front.shape
        # extend with type blocks
        front.extend(back) 
        self.assertEqual(front.shape,
                (shape_original[0], shape_original[1] + back.shape[1]))

        # extend with iterable of arrays
        front.extend(back._blocks)
        self.assertEqual(front.shape,
                (shape_original[0], shape_original[1] + back.shape[1] * 2))



