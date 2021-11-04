
# import typing as tp
import unittest
# import operator
# import os
# import sqlite3
# import gc

# import numpy as np
from hypothesis import given
# from arraykit import isna_element

from static_frame.core.frame import Frame
from static_frame.core.index_datetime import IndexDate

# from static_frame.core.frame import FrameGO
# from static_frame.core.interface import UFUNC_AXIS_SKIPNA
# from static_frame.core.interface import UFUNC_BINARY_OPERATORS
# from static_frame.core.interface import UFUNC_UNARY_OPERATORS
# from static_frame.core.series import Series
from static_frame.test.property import strategies as sfst

# from static_frame.test.test_case import skip_win
from static_frame.test.test_case import temp_file
from static_frame.test.test_case import TestCase



class TestUnit(TestCase):

    @given(sfst.get_frame(dtype_group=sfst.DTGroup.ALL_NO_OBJECT,
            index_dtype_group=sfst.DTGroup.BASIC,
            columns_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_npz_a(self, f1: Frame) -> None:
        # if f1.columns.dtype.kind != 'O' and f1.index.dtype.kind != 'O':
        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True))


    @given(sfst.get_frame(dtype_group=sfst.DTGroup.ALL_NO_OBJECT,
            index_cls=IndexDate,
            index_dtype_group=sfst.DTGroup.DATE,
            columns_cls=IndexDate,
            columns_dtype_group=sfst.DTGroup.DATE,
            ))
    def test_frame_to_npz_b(self, f1: Frame) -> None:
        # if f1.columns.dtype.kind != 'O' and f1.index.dtype.kind != 'O':
        with temp_file('.npz') as fp:
            f1.to_npz(fp)
            f2 = Frame.from_npz(fp)
            self.assertTrue(f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True))



if __name__ == '__main__':
    unittest.main()
