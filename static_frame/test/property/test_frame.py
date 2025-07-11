import gc
import operator
import os
import sqlite3

import numpy as np
import typing_extensions as tp
from arraykit import isna_element
from hypothesis import given

from static_frame.core.exception import ErrorInitIndexNonUnique
from static_frame.core.frame import Frame, FrameGO
from static_frame.core.interface import (
    UFUNC_AXIS_SKIPNA,
    UFUNC_BINARY_OPERATORS,
    UFUNC_UNARY_OPERATORS,
)
from static_frame.core.series import Series
from static_frame.core.util import DTYPE_INEXACT_KINDS, TLabel, WarningsSilent
from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import TestCase, skip_win, temp_file


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
            if op == '__invert__':  # invalid on non Boolean
                continue
            func = getattr(operator, op)
            values = f1.values
            # must coerce all blocks to same type to compare to what NP does
            with WarningsSilent():
                a = func(f1.astype(values.dtype)).values
                b = func(values)
                self.assertAlmostEqualArray(a, b)

    @given(sfst.get_frame_or_frame_go(dtype_group=sfst.DTGroup.BOOL))
    def test_unary_operators_boolean(self, f1: Frame) -> None:
        for op in UFUNC_UNARY_OPERATORS:
            if op != '__invert__':  # valid on Boolean
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
                '__floordiv__',
            }:
                continue
            func = getattr(operator, op)
            values = f1.values
            # must coerce all blocks to same type to compare to what NP does
            f2 = f1.astype(values.dtype)
            with WarningsSilent():
                a = func(f2, f2).values
                b = func(values, values)
                if a.dtype.kind in DTYPE_INEXACT_KINDS:
                    if (np.isnan(a) | np.isinf(a)).all() and (
                        np.isnan(b) | np.isinf(b)
                    ).all():
                        return
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

    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.NUMERIC,
            index_dtype_group=sfst.DTGroup.STRING,
            min_rows=3,
            max_rows=3,
            min_columns=3,
            max_columns=3,
        )
    )
    def test_binary_operators_matmul(
        self,
        f1: Frame,
    ) -> None:
        f2 = f1.relabel(columns=f1.index)
        with WarningsSilent():
            f3 = f2 @ f1
            self.assertAlmostEqualArray(f3.values, f2.values @ f1.values)

    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.NUMERIC_REAL, min_rows=1, min_columns=1
        )
    )
    def test_ufunc_axis(self, f1: Frame) -> None:
        for attr, attrs in UFUNC_AXIS_SKIPNA.items():
            if attr in ('std', 'var'):
                continue

            for axis in (0, 1):
                values = f1.values
                # must coerce all blocks to same type to compare to what NP does
                # f2 = f1.astype(values.dtype)
                with WarningsSilent():
                    a = getattr(f1, attr)(axis=axis).values  # call the method
                    b = attrs.ufunc_skipna(values, axis=axis)

    # NOTE: this fails with dt64 types due to odd unitless values from hypothesis
    # from hypothesis import reproduce_failure
    # @reproduce_failure('6.40.0', b'AXicY2DmYGBkwAKQBBkRbCY0KagCLhAJAAS+AB4=')
    @given(
        sfst.get_frame(
            dtype_group=sfst.DTGroup.CORE_NO_OBJECT,
        )
    )
    def test_frame_isin(self, f1: Frame) -> None:
        value = f1.iloc[0, 0]
        if (
            not isna_element(value)
            and not isinstance(value, np.datetime64)
            and not isinstance(value, np.timedelta64)
        ):
            self.assertTrue(f1.isin((value,)).iloc[0, 0])

    # ---------------------------------------------------------------------------

    @given(sfst.get_frame_go(), sfst.get_label())
    def test_frame_go_setitem(self, f1: Frame, label: TLabel) -> None:
        shape = f1.shape
        f1['foo'] = label  # type: ignore
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

        def items() -> tp.Iterator[tp.Tuple[TLabel, Series]]:
            for idx, label in enumerate(letters):
                s = Series(series_arrays[idx], index=f1.index)
                yield label, s

        f1.extend_items(items())

        self.assertEqual(f1.shape[1], shape[1] + len(series_arrays))

    # ---------------------------------------------------------------------------
    # exporters

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_pairs(self, f1: Frame) -> None:
        for i in range(1):
            post = f1.to_pairs(axis=i)
            if i == 1:
                self.assertEqual(len(post), f1.shape[1])  # type: ignore
            else:
                self.assertEqual(len(post[0][1]), f1.shape[0])  # type: ignore
            self.assertTrue(isinstance(post, tuple))

    # from hypothesis import reproduce_failure
    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC_NO_REAL,
        )
    )
    def test_frame_to_pandas(self, f1: Frame) -> None:
        try:
            post = f1.to_pandas()
        except NotImplementedError:
            return

        self.assertTrue(post.shape == f1.shape)
        if not f1.isna().any().any():
            self.assertTrue((post.values == f1.values).all())

    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
        )
    )
    def test_frame_to_parquet(self, f1: Frame) -> None:
        import pyarrow

        with temp_file('.parquet') as fp:
            try:
                f1.to_parquet(fp)
                self.assertTrue(os.stat(fp).st_size > 0)
            except pyarrow.lib.ArrowNotImplementedError:
                # could be Byte-swapped arrays not supported
                pass

    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC_NO_REAL,
        )
    )
    def test_frame_to_xarray(self, f1: Frame) -> None:
        xa = f1.to_xarray()
        self.assertAlmostEqualValues(xa.keys(), f1.columns)

    @given(
        sfst.get_frame(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.BASIC,
        )
    )
    def test_frame_to_frame_go(self, f1: Frame) -> None:
        f2 = f1.to_frame_go()
        f2['__new__'] = 10
        self.assertTrue(len(f2.columns) == len(f1.columns) + 1)

    @skip_win  # get UnicodeEncodeError: 'charmap' codec can't encode character '\u0162' in position 0: character maps to <undefined>
    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
        )
    )
    def test_frame_to_csv(self, f1: Frame) -> None:
        with temp_file('.csv') as fp:
            f1.to_csv(fp, escape_char=r'`')
            self.assertTrue(os.stat(fp).st_size > 0)

            # not yet validating result, as edge cases with unusual unicode and non-unique indices are a problem
            # f2 = Frame.from_csv(fp,
            #         index_depth=f1.index.depth,
            #         columns_depth=f1.columns.depth)

    @skip_win  # UnicodeEncodeError
    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
        )
    )
    def test_frame_to_tsv(self, f1: Frame) -> None:
        with temp_file('.txt') as fp:
            f1.to_tsv(fp, escape_char='`')
            self.assertTrue(os.stat(fp).st_size > 0)

    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
        )
    )
    def test_frame_to_xlsx(self, f1: Frame) -> None:
        with temp_file('.xlsx') as fp:
            f1.to_xlsx(fp)
            self.assertTrue(os.stat(fp).st_size > 0)

    # from hypothesis import reproduce_failure
    # @reproduce_failure('6.92.1', b'AXicY2BlAAImBjBgZIDTQBFmBgYWBkZGFkYgPwokBQAELQB2')
    @given(
        sfst.get_frame_or_frame_go(
            dtype_group=sfst.DTGroup.BASIC,
            index_dtype_group=sfst.DTGroup.STRING,
            columns_dtype_group=sfst.DTGroup.STRING,
        )
    )
    def test_frame_to_sqlite(self, f1: Frame) -> None:
        with temp_file('.sqlite') as fp:
            try:
                f1.to_sqlite(fp, label='foo')
                self.assertTrue(os.stat(fp).st_size > 0)
            except (
                sqlite3.IntegrityError,
                sqlite3.OperationalError,
                sqlite3.ProgrammingError,
                sqlite3.Warning,
                OverflowError,
                ValueError,  # the query contains a null character
            ):
                # some indices, after translation, are not unique
                # SQLite is no case sensitive, and does not support unicide
                # OverflowError: Python int too large to convert to SQLite INTEGER
                pass

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_html(self, f1: Frame) -> None:
        post = f1.to_html()
        self.assertTrue(len(post) > 0)

    @skip_win  # UnicodeEncodeError
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

    # ---------------------------------------------------------------------------

    @given(sfst.get_frame_or_frame_go())
    def test_frame_blocks_dont_have_reference_cycles(self, f1: Frame) -> None:
        self.assertEqual([f1], gc.get_referrers(f1._blocks))

    # ---------------------------------------------------------------------------
    # NOTE: re-encoding from json strings is difficult here as some string representations result in non-unique indices.

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_json_index(self, f1: Frame) -> None:
        msg = f1.to_json_index()
        self.assertIsInstance(msg, str)
        try:
            f2 = Frame.from_json_index(msg)
        except ErrorInitIndexNonUnique:
            pass

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_json_columns(self, f1: Frame) -> None:
        msg = f1.to_json_columns()
        self.assertIsInstance(msg, str)
        try:
            f2 = Frame.from_json_columns(msg)
        except ErrorInitIndexNonUnique:
            pass

    # from hypothesis import reproduce_failure
    # @reproduce_failure('6.92.1', b'AXicXYxBDgAgCMMqoP9/sookOndgKYwxAhp0SL9qiwyVfZkMPaT5fg/1DL5H1Eq7BN8m9+MDJhbzAD4=')
    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_json_split(self, f1: Frame) -> None:
        msg = f1.to_json_split()
        self.assertIsInstance(msg, str)
        try:
            f2 = Frame.from_json_split(msg)
        except ErrorInitIndexNonUnique:
            pass

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_json_records(self, f1: Frame) -> None:
        msg = f1.to_json_records()
        self.assertIsInstance(msg, str)
        try:
            f2 = Frame.from_json_records(msg)
        except ErrorInitIndexNonUnique:
            pass

    @given(sfst.get_frame_or_frame_go())
    def test_frame_to_json_values(self, f1: Frame) -> None:
        msg = f1.to_json_values()
        self.assertIsInstance(msg, str)
        try:
            f2 = Frame.from_json_values(msg)
        except ErrorInitIndexNonUnique:
            pass

    @given(sfst.get_frame_or_frame_go_core_no_object())
    def test_frame_to_json_typed(self, f1: Frame) -> None:
        msg = f1.to_json_typed()
        self.assertIsInstance(msg, str)
        try:
            f2 = Frame.from_json_typed(msg)
        except ErrorInitIndexNonUnique:
            pass


if __name__ == '__main__':
    import unittest

    unittest.main()
