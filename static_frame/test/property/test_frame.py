
import typing as tp
import unittest
import operator

import numpy as np  # type: ignore

# from hypothesis import strategies as st
from hypothesis import given  # type: ignore


from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.series import Series
from static_frame.core.container import _UFUNC_UNARY_OPERATORS
from static_frame.core.container import _UFUNC_BINARY_OPERATORS
from static_frame.core.container import UFUNC_AXIS_SKIPNA

from static_frame.test.property import strategies as sfst
# from static_frame.test.test_case import temp_file

from static_frame.test.test_case import TestCase

from static_frame import Frame



class TestUnit(TestCase):


    @given(sfst.get_frame_or_frame_go())  # type: ignore
    def test_basic_attributes(self, f1: Frame) -> None:

        self.assertEqual(len(f1.dtypes), f1.shape[1])
        # self.assertEqual(f1.shape, f1.shape)
        self.assertEqual(f1.ndim, 2)
        # self.assertEqual(f1.unified, len(f1.mloc) <= 1)

        if f1.shape[0] > 0 and f1.shape[1] > 0:
            self.assertTrue(f1.size > 0)
            self.assertTrue(f1.nbytes > 0)
        else:
            self.assertTrue(f1.size == 0)
            self.assertTrue(f1.nbytes == 0)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.NUMERIC))  # type: ignore
    def test_unary_operators_numeric(self, f1: Frame) -> None:
        for op in _UFUNC_UNARY_OPERATORS:
            if op == '__invert__': # invalid on non Boolean
                continue
            func = getattr(operator, op)
            values = f1.values
            # must coerce all blocks to same type to compare to what NP does
            a = func(f1.astype(values.dtype)).values
            b = func(values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.BOOL))  # type: ignore
    def test_unary_operators_boolean(self, f1: Frame) -> None:
        for op in _UFUNC_UNARY_OPERATORS:
            if op != '__invert__': # valid on Boolean
                continue
            func = getattr(operator, op)
            a = func(f1).values
            b = func(f1.values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.NUMERIC))  # type: ignore
    def test_binary_operators_numeric(self, f1: Frame) -> None:
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
            values = f1.values
            # must coerce all blocks to same type to compare to what NP does
            f2 = f1.astype(values.dtype)
            a = func(f2, f2).values
            b = func(values, values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.BOOL))  # type: ignore
    def test_binary_operators_boolean(self, f1: Frame) -> None:
        for op in _UFUNC_BINARY_OPERATORS:
            if op not in {
                    '__and__',
                    '__xor__',
                    '__or__',
                    }:
                continue
            func = getattr(operator, op)
            a = func(f1, f1).values
            values = f1.values
            b = func(values, values)
            self.assertAlmostEqualArray(a, b)

    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.NUMERIC, # type: ignore
            min_rows=3,
            max_rows=3,
            min_columns=3,
            max_columns=3)
            )
    def test_binary_operators_matmul(self,
            f1: Frame,
            ) -> None:

        f2 = f1.relabel(columns=f1.index)
        f3 = f2 @ f1
        self.assertAlmostEqualArray(f3.values, f2.values @ f1.values)

    # from hypothesis import reproduce_failure
    # NOTE: was able to improve many of these, but continued to get compliated type cases, and complications
    @given(sfst.get_frame_or_frame_go( # type: ignore
            dtype_group=sfst.DTGroup.NUMERIC_REAL,
            min_rows=1,
            min_columns=1))
    def test_ufunc_axis(self, f1: Frame) -> None:

        for attr, attrs in UFUNC_AXIS_SKIPNA.items():

            if attr in ('std', 'var'):
                continue

            for axis in (0, 1):
                values = f1.values
                # must coerce all blocks to same type to compare to what NP does
                # f2 = f1.astype(values.dtype)

                a = getattr(f1, attr)(axis=axis).values # call the method
                b = attrs.ufunc_skipna(values, axis=axis)

    #             if a.dtype != b.dtype:
    #                 continue
    #             try:
    #                 self.assertAlmostEqualArray(a, b)
    #             except:
    #                 import ipdb; ipdb.set_trace()
    #                 raise



    # # TODO: intger tests with pow, mod

    #---------------------------------------------------------------------------

    @given(sfst.get_frame_go(), sfst.get_label())  # type: ignore
    def test_frame_go_setitem(self, f1: Frame, label: tp.Hashable) -> None:

        shape = f1.shape
        f1['foo'] = label # type: ignore
        self.assertEqual(shape[1] + 1, f1.shape[1])


    @given(sfst.get_arrays_2d_aligned_rows(min_size=2, max_size=2))  # type: ignore
    def test_frame_go_extend(self, arrays: tp.Sequence[np.ndarray]) -> None:
        f1 = FrameGO(arrays[0], columns=self.get_letters(arrays[0].shape[1]))
        shape = f1.shape
        f2 = Frame(arrays[1])
        f1.extend(f2)
        self.assertEqual(f1.shape[1], shape[1] + f2.shape[1])


    @given(sfst.get_arrays_2d_aligned_rows(min_size=3))  # type: ignore
    def test_frame_go_extend_items(self, arrays: tp.Sequence[np.ndarray]) -> None:
        frame_array = arrays[0]
        # just take first columm form 2d arrays
        series_arrays = [a[:, 0] for a in arrays[1:]]

        f1 = FrameGO(frame_array)
        shape = f1.shape

        letters = self.get_letters(len(series_arrays))

        def items() -> tp.Iterator[tp.Tuple[tp.Hashable, Series]]:
            for idx, label in enumerate(letters):
                s = Series(series_arrays[idx], index=f1.index)
                yield label, s

        f1.extend_items(items())

        self.assertEqual(f1.shape[1], shape[1] + len(series_arrays))


    #---------------------------------------------------------------------------
    # exporters

    @given(sfst.get_frame_or_frame_go())  # type: ignore
    def test_frame_to_pairs(self, f1: Frame) -> None:
        for i in range(0, 1):
            post = f1.to_pairs(i)
            if i == 1:
                self.assertEqual(len(post), f1.shape[1]) # type: ignore
            else:
                self.assertEqual(len(post[0][1]), f1.shape[0]) # type: ignore
            self.assertTrue(isinstance(post, tuple))


    @given(sfst.get_frame_or_frame_go( # type: ignore
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_pandas(self, f1: Frame) -> None:
        post = f1.to_pandas()
        self.assertTrue(post.shape == f1.shape)
        if not f1.isna().any().any(): # type: ignore
            self.assertTrue((post.values == f1.values).all())


    # @given(sfst.get_frame_or_frame_go(
    #         dtype_group=sfst.DTGroup.BASIC,
    #         index_dtype_group=sfst.DTGroup.BASIC,
    #         ))  # type: ignore
    # def test_frame_to_parquet(self, f1: Frame) -> None:

    #     with temp_file('.parquet') as fp:
    #         f1.to_parquet(fp)








if __name__ == '__main__':
    unittest.main()
