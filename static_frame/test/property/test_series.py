
# import typing as tp
import unittest
import operator

# import numpy as np  # type: ignore

# from hypothesis import strategies as st
from hypothesis import given  # type: ignore

from static_frame.core.container import _UFUNC_UNARY_OPERATORS
from static_frame.core.container import _UFUNC_BINARY_OPERATORS
from static_frame.core.container import UFUNC_AXIS_SKIPNA

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


    @given(sfst.get_series(dtype_group=sfst.DTGroup.NUMERIC, min_size=1))  # type: ignore
    def test_unary_operators_numeric(self, s1: Series) -> None:
        for op in _UFUNC_UNARY_OPERATORS:
            if op == '__invert__': # invalid on non Boolean
                continue
            func = getattr(operator, op)
            a = func(s1).values
            b = func(s1.values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_series(dtype_group=sfst.DTGroup.BOOL, min_size=1))  # type: ignore
    def test_unary_operators_boolean(self, s1: Series) -> None:
        for op in _UFUNC_UNARY_OPERATORS:
            if op != '__invert__': # valid on Boolean
                continue
            func = getattr(operator, op)
            a = func(s1).values
            b = func(s1.values)
            self.assertAlmostEqualArray(a, b)



    @given(sfst.get_series(dtype_group=sfst.DTGroup.NUMERIC))  # type: ignore
    def test_binary_operators_numeric(self, s1: Series) -> None:
        for op in _UFUNC_BINARY_OPERATORS:
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

    @given(sfst.get_series(dtype_group=sfst.DTGroup.BOOL))  # type: ignore
    def test_binary_operators_boolean(self, s1: Series) -> None:
        for op in _UFUNC_BINARY_OPERATORS:
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


    @given(sfst.get_series(dtype_group=sfst.DTGroup.NUMERIC, min_size=1))  # type: ignore
    def test_ufunc_axis(self, s1: Series) -> None:
        for attr, attrs in UFUNC_AXIS_SKIPNA.items():

            a = getattr(s1, attr)().values # call the method
            func = attrs.funcna
            b = func(s1.values)
            self.assertAlmostEqualArray(a, b)



if __name__ == '__main__':
    unittest.main()


