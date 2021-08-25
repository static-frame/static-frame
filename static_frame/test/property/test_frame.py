
import typing as tp
import unittest
import operator
import os
import sqlite3
import gc

import numpy as np
from hypothesis import given
from arraykit import isna_element

from static_frame.core.frame import Frame
from static_frame.core.frame import FrameGO
from static_frame.core.interface import UFUNC_AXIS_SKIPNA
from static_frame.core.interface import UFUNC_BINARY_OPERATORS
from static_frame.core.interface import UFUNC_UNARY_OPERATORS
from static_frame.core.series import Series
from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import skip_win
from static_frame.test.test_case import temp_file
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):


    @given(sfst.get_frame_or_frame_go())
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


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.NUMERIC))
    def test_unary_operators_numeric(self, f1: Frame) -> None:
        for op in UFUNC_UNARY_OPERATORS:
            if op == '__invert__': # invalid on non Boolean
                continue
            func = getattr(operator, op)
            values = f1.values
            # must coerce all blocks to same type to compare to what NP does
            a = func(f1.astype(values.dtype)).values
            b = func(values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.BOOL))
    def test_unary_operators_boolean(self, f1: Frame) -> None:
        for op in UFUNC_UNARY_OPERATORS:
            if op != '__invert__': # valid on Boolean
                continue
            func = getattr(operator, op)
            a = func(f1).values
            b = func(f1.values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.NUMERIC))
    def test_binary_operators_numeric(self, f1: Frame) -> None:
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
            values = f1.values
            # must coerce all blocks to same type to compare to what NP does
            f2 = f1.astype(values.dtype)
            a = func(f2, f2).values
            b = func(values, values)
            self.assertAlmostEqualArray(a, b)


    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.BOOL))
    def test_binary_operators_boolean(self, f1: Frame) -> None:
        for op in UFUNC_BINARY_OPERATORS:
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

    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.NUMERIC,
            index_dtype_group=sfst.DTGroup.STRING,
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
    @given(sfst.get_frame_or_frame_go(
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

    @given(sfst.get_frame())
    def test_frame_isin(self, f1: Frame) -> None:
        value = f1.iloc[0, 0]
        if (not isna_element(value) and
                not isinstance(value, np.datetime64) and
                not isinstance(value, np.timedelta64)):
            self.assertTrue(f1.isin((value,)).iloc[0, 0])



    # # TODO: intger tests with pow, mod

    #---------------------------------------------------------------------------

    @given(sfst.get_frame_go(), sfst.get_label())
    def test_frame_go_setitem(self, f1: Frame, label: tp.Hashable) -> None:

        shape = f1.shape
        f1['foo'] = label # type: ignore
        self.assertEqual(shape[1] + 1, f1.shape[1])


    @given(sfst.get_arrays_2d_aligned_rows(min_size=2, max_size=2))
    def test_frame_go_extend(self, arrays: tp.Sequence[np.ndarray]) -> None:
        f1 = FrameGO(arrays[0], columns=self.get_letters(arrays[0].shape[1]))
        shape = f1.shape
        f2 = Frame(arrays[1])
        f1.extend(f2)
        self.assertEqual(f1.shape[1], shape[1] + f2.shape[1])


    @given(sfst.get_arrays_2d_aligned_rows(min_size=3))
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

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_pairs(self, f1: Frame) -> None:
        for i in range(0, 1):
            post = f1.to_pairs(i)
            if i == 1:
                self.assertEqual(len(post), f1.shape[1]) # type: ignore
            else:
                self.assertEqual(len(post[0][1]), f1.shape[0]) # type: ignore
            self.assertTrue(isinstance(post, tuple))


    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_pandas(self, f1: Frame) -> None:
        post = f1.to_pandas()
        self.assertTrue(post.shape == f1.shape)
        if not f1.isna().any().any():
            self.assertTrue((post.values == f1.values).all())


    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_parquet(self, f1: Frame) -> None:
        import pyarrow
        with temp_file('.parquet') as fp:
            try:
                f1.to_parquet(fp)
                self.assertTrue(os.stat(fp).st_size > 0)
            except pyarrow.lib.ArrowNotImplementedError:
                # could be Byte-swapped arrays not supported
                pass


    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.CORE,
            index_dtype_group=sfst.DTGroup.CORE,
            ))
    def test_frame_to_msgpack(self, f1: Frame) -> None:
        msg = f1.to_msgpack()

        f2 = Frame.from_msgpack(msg)
        assert f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)

        f2 = Frame.from_msgpack(f1.to_msgpack())
        assert f1.equals(f2, compare_name=True, compare_dtype=True, compare_class=True)


    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_xarray(self, f1: Frame) -> None:
        xa = f1.to_xarray()
        self.assertTrue(tuple(xa.keys()) == tuple(f1.columns))


    @given(sfst.get_frame(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_frame_go(self, f1: Frame) -> None:
        f2 = f1.to_frame_go()
        f2['__new__'] = 10
        self.assertTrue(len(f2.columns) == len(f1.columns) + 1)

    @skip_win  # type: ignore # get UnicodeEncodeError: 'charmap' codec can't encode character '\u0162' in position 0: character maps to <undefined>
    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_csv(self, f1: Frame) -> None:
        with temp_file('.csv') as fp:
            f1.to_csv(fp)
            self.assertTrue(os.stat(fp).st_size > 0)

            # not yet validating result, as edge cases with unusual unicode and non-unique indices are a problem
            # f2 = Frame.from_csv(fp,
            #         index_depth=f1.index.depth,
            #         columns_depth=f1.columns.depth)


    @skip_win  # type: ignore # UnicodeEncodeError
    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_tsv(self, f1: Frame) -> None:
        with temp_file('.txt') as fp:
            f1.to_tsv(fp)
            self.assertTrue(os.stat(fp).st_size > 0)


    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_xlsx(self, f1: Frame) -> None:
        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            self.assertTrue(os.stat(fp).st_size > 0)

    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            ))
    def test_frame_to_sqlite(self, f1: Frame) -> None:
        with temp_file('.sqlite') as fp:

            try:
                f1.to_sqlite(fp)
                self.assertTrue(os.stat(fp).st_size > 0)
            except (sqlite3.IntegrityError, sqlite3.OperationalError, OverflowError):
                # some indices, after translation, are not unique
                # SQLite is no case sensitive, and does not support unicide
                # OverflowError: Python int too large to convert to SQLite INTEGER
                pass


    @given(sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            columns_dtype_group=sfst.DTGroup.STRING,
            index_dtype_group=sfst.DTGroup.STRING
            ))
    def test_frame_to_hdf5(self, f1: Frame) -> None:
        f1 = f1.rename('f1')
        with temp_file('.hdf5') as fp:

            try:
                f1.to_hdf5(fp)
                self.assertTrue(os.stat(fp).st_size > 0)
            except ValueError:
                # will happen for empty strings and unicde that cannot be handled by HDF5
                pass


    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_html(self, f1: Frame) -> None:
        post = f1.to_html()
        self.assertTrue(len(post) > 0)

    @skip_win  # type: ignore # UnicodeEncodeError
    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_html_datatables(self, f1: Frame) -> None:
        post = f1.to_html_datatables(show=False)
        self.assertTrue(len(post) > 0)

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_rst(self, f1: Frame) -> None:
        post = f1.to_rst()
        self.assertTrue(len(post) > 0)

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_markdown(self, f1: Frame) -> None:
        post = f1.to_markdown()
        self.assertTrue(len(post) > 0)

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_latex(self, f1: Frame) -> None:
        post = f1.to_latex()
        self.assertTrue(len(post) > 0)

    @given(sfst.get_frame_or_frame_go())
    def test_frame_blocks_dont_have_reference_cycles(self, f1: Frame) -> None:
        self.assertEqual([f1], gc.get_referrers(f1._blocks))


if __name__ == '__main__':
    unittest.main()
