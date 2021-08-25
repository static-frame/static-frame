
import typing as tp
import unittest

import numpy as np

# from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings as hypo_settings

from static_frame.test.property import strategies as sfst

from static_frame.test.test_case import TestCase

from static_frame import TypeBlocks
from static_frame import Index
from static_frame import IndexHierarchy
from static_frame import Series
from static_frame import Frame


class TestUnit(TestCase):


    @given(sfst.get_labels())
    def test_get_labels(self, values: tp.Iterable[tp.Hashable]) -> None:
        for value in values:
            self.assertTrue(isinstance(hash(value), int))

    @given(sfst.get_dtypes())
    def test_get_dtypes(self, dtypes: tp.Iterable[np.dtype]) -> None:
        for dt in dtypes:
            self.assertTrue(isinstance(dt, np.dtype))

    @given(sfst.get_spacing(10))
    def test_get_spacing_10(self, spacing: tp.Iterable[int]) -> None:
        self.assertEqual(sum(spacing), 10)

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_shape_1d2d())
    def test_get_shape_1d2d(self, shape: tp.Tuple[int, ...]) -> None:
        self.assertTrue(isinstance(shape, tuple))
        self.assertTrue(len(shape) in (1, 2))

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_array_1d2d())
    def test_get_array_1d2d(self, array: np.ndarray) -> None:
        self.assertTrue(isinstance(array, np.ndarray))
        self.assertTrue(array.ndim in (1, 2))

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_arrays_2d_aligned_columns(min_size=2))
    def test_get_arrays_2s_aligned_columns(self, arrays: tp.Iterable[np.ndarray]) -> None:
        array_iter = iter(arrays)
        a1 = next(array_iter)
        match = a1.shape[1]
        for array in array_iter:
            self.assertEqual(array.shape[1], match)

    @given(sfst.get_arrays_2d_aligned_rows(min_size=2))
    def test_get_arrays_2s_aligned_rows(self, arrays: tp.Iterable[np.ndarray]) -> None:
        array_iter = iter(arrays)
        a1 = next(array_iter)
        match = a1.shape[0]
        for array in array_iter:
            self.assertEqual(array.shape[0], match)

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_blocks())
    def test_get_blocks(self, blocks: tp.Tuple[np.ndarray]) -> None:
        self.assertTrue(isinstance(blocks, tuple))
        for b in blocks:
            self.assertTrue(isinstance(b, np.ndarray))
            self.assertTrue(b.ndim in (1, 2))

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_type_blocks())
    def test_get_type_blocks(self, tb: TypeBlocks) -> None:
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

    @hypo_settings(max_examples=10) # type: ignore
    @given(sfst.get_index())
    def test_get_index(self, idx: Index) -> None:
        self.assertTrue(isinstance(idx, Index))
        self.assertEqual(len(idx), len(idx.values))

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_index_hierarchy())
    def test_get_index_hierarchy(self, idx: IndexHierarchy) -> None:
        self.assertTrue(isinstance(idx, IndexHierarchy))
        self.assertTrue(idx.depth > 1)
        self.assertEqual(len(idx), len(idx.values))

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_series())
    def test_get_series(self, series: Series) -> None:
        self.assertTrue(isinstance(series, Series))
        self.assertEqual(len(series), len(series.values))

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_frame())
    def test_get_frame(self, frame: Frame) -> None:
        self.assertTrue(isinstance(frame, Frame))
        self.assertEqual(frame.shape, frame.values.shape)

    @hypo_settings(max_examples=10)  # type: ignore
    @given(sfst.get_frame(index_cls=IndexHierarchy, columns_cls=IndexHierarchy))  # type: ignore
    def test_get_frame_hierarchy(self, frame: Frame) -> None:
        self.assertTrue(isinstance(frame, Frame))
        self.assertTrue(frame.index.depth > 1)
        self.assertTrue(frame.columns.depth > 1)
        self.assertEqual(frame.shape, frame.values.shape)


if __name__ == '__main__':
    unittest.main()
