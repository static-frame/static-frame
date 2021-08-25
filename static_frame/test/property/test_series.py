# import typing as tp
import unittest
import operator

# from hypothesis import strategies as st
from hypothesis import given
from arraykit import isna_element

from static_frame.core.interface import UFUNC_UNARY_OPERATORS
from static_frame.core.interface import UFUNC_BINARY_OPERATORS
from static_frame.core.interface import UFUNC_AXIS_SKIPNA

from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import TestCase

from static_frame import Series



class TestUnit(TestCase):


    @given(sfst.get_series())
    def test_basic_attributes(self, s1: Series) -> None:

        self.assertEqual(s1.dtype, s1.values.dtype)
        # self.assertEqual(s1.shape, s1.shape)
        self.assertEqual(s1.ndim, 1)

        if s1.shape[0] > 0:
            self.assertTrue(s1.size > 0)
            self.assertTrue(s1.nbytes > 0)


    @given(sfst.get_series(dtype_group=sfst.DTGroup.NUMERIC, min_size=1))
    def test_unary_operators_numeric(self, s1: Series) -> None:
        for op in UFUNC_UNARY_OPERATORS:
            if op == '__invert__': # invalid on non Boolean
                continue
            func = getattr(operator, op)
            a = func(s1).values
            b = func(s1.values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_series(dtype_group=sfst.DTGroup.BOOL, min_size=1))
    def test_unary_operators_boolean(self, s1: Series) -> None:
        for op in UFUNC_UNARY_OPERATORS:
            if op != '__invert__': # valid on Boolean
                continue
            func = getattr(operator, op)
            a = func(s1).values
            b = func(s1.values)
            self.assertAlmostEqualArray(a, b)



    @given(sfst.get_series(dtype_group=sfst.DTGroup.NUMERIC))
    def test_binary_operators_numeric(self, s1: Series) -> None:
        for op in UFUNC_BINARY_OPERATORS:
            if op in {
                    '__matmul__',
                    '__pow__',
                    '__lshift__',
                    '__rshift__',
                    '__and__',
                    '__xor__',
                    '__or__',
                    '__mod__',
                    }:
                continue
            func = getattr(operator, op)
            values = s1.values
            a = func(s1, s1).values
            b = func(values, values)
            self.assertAlmostEqualArray(a, b)

    @given(sfst.get_series(dtype_group=sfst.DTGroup.BOOL))
    def test_binary_operators_boolean(self, s1: Series) -> None:
        for op in UFUNC_BINARY_OPERATORS:
            if op not in {
                    '__and__',
                    '__xor__',
                    '__or__',
                    }:
                continue
            func = getattr(operator, op)
            values = s1.values
            a = func(s1, s1).values
            b = func(values, values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_series(dtype_group=sfst.DTGroup.NUMERIC, min_size=1))
    def test_ufunc_axis(self, s1: Series) -> None:
        for attr, attrs in UFUNC_AXIS_SKIPNA.items():
            a = getattr(s1, attr)()
            func = attrs.ufunc_skipna
            b = func(s1.values)
            self.assertEqualWithNaN(a, b)

    @given(sfst.get_series(min_size=1))
    def test_isin(self, s1: Series) -> None:

        value = s1.iloc[0]
        if not isna_element(value):
            self.assertTrue(s1.isin((value,)).iloc[0])


#     @given(sfst.get_series(min_size=1), sfst.get_series(min_size=1), sfst.get_series(min_size=1))
#     def test_from_overlay(self,
#                 s1: Series,
#                 s2: Series,
#                 s3: Series,
#                 ) -> None:

#         # NOTE: this fails dues to numpy doing this:
# #         In : np.array(9007199268722005).astype(float).tolist()
#  # 9007199268722004.0
#         # this happens in calls to np.union1d

#         post = Series.from_overlay((s1, s2, s3))
#         self.assertTrue(post.index.equals(s1.index.union(s2.index, s3.index)))





if __name__ == '__main__':
    unittest.main()
