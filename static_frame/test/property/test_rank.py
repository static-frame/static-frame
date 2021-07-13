


# import typing as tp
import unittest
from hypothesis import given
import numpy as np
from scipy.stats import rankdata

from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import TestCase
from static_frame.core.rank import rank_1d
from static_frame.core.rank import rank_2d



class TestUnit(TestCase):


    @given(sfst.get_array_1d(dtype_group=sfst.DTGroup.NUMERIC, max_size=100))
    def test_rank_1d_ordinal(self, value: np.ndarray) -> None:
        a1 = rankdata(value, method='ordinal')
        a2 = rank_1d(value, method='ordinal', start=1)
        self.assertEqual(a1.tolist(), a2.tolist())

        if len(value):
            a3 = rank_1d(value, method='ordinal')
            self.assertEqual(a3.min(), 0)

            a4 = rank_1d(value, method='ordinal', ascending=False)
            self.assertEqual(a4.min(), 0)

    @given(sfst.get_array_1d(dtype_group=sfst.DTGroup.NUMERIC, max_size=100))
    def test_rank_1d_dense(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        a1 = rankdata(value, method='dense')
        a2 = rank_1d(value, method='dense', start=1)
        self.assertEqual(a1.tolist(), a2.tolist())

        if len(value):
            a3 = rank_1d(value, method='dense')
            self.assertEqual(a3.min(), 0)

            a4 = rank_1d(value, method='dense', ascending=False)
            self.assertEqual(a4.min(), 0)

    @given(sfst.get_array_1d(dtype_group=sfst.DTGroup.NUMERIC, max_size=100))
    def test_rank_1d_min(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        a1 = rankdata(value, method='min')
        a2 = rank_1d(value, method='min', start=1)
        self.assertEqual(a1.tolist(), a2.tolist())

        if len(value):
            a3 = rank_1d(value, method='min')
            self.assertEqual(a3.min(), 0)

            a4 = rank_1d(value, method='min', ascending=False)
            self.assertEqual(a4.min(), 0)


    @given(sfst.get_array_1d(dtype_group=sfst.DTGroup.NUMERIC, max_size=100))
    def test_rank_1d_max(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        a1 = rankdata(value, method='max')
        a2 = rank_1d(value, method='max', start=1)
        self.assertEqual(a1.tolist(), a2.tolist())



    @given(sfst.get_array_1d(dtype_group=sfst.DTGroup.NUMERIC, max_size=100))
    def test_rank_1d_average(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        a1 = rankdata(value, method='average')
        a2 = rank_1d(value, method='mean', start=1)
        self.assertEqual(a1.tolist(), a2.tolist())




    @given(sfst.get_array_2d(dtype_group=sfst.DTGroup.NUMERIC, max_rows=20, max_columns=20))
    def test_rank_2d_ordinal(self, value: np.ndarray) -> None:
        for axis in (0, 1):
            a1 = rankdata(value, method='ordinal', axis=axis)
            a2 = rank_2d(value, method='ordinal', start=1, axis=axis)
            self.assertEqual(a1.tolist(), a2.tolist())

    @given(sfst.get_array_2d(dtype_group=sfst.DTGroup.NUMERIC, max_rows=20, max_columns=20))
    def test_rank_2d_dense(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        for axis in (0, 1):
            a1 = rankdata(value, method='dense', axis=axis)
            a2 = rank_2d(value, method='dense', start=1, axis=axis)
            self.assertEqual(a1.tolist(), a2.tolist())

    @given(sfst.get_array_2d(dtype_group=sfst.DTGroup.NUMERIC, max_rows=20, max_columns=20))
    def test_rank_2d_min(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        for axis in (0, 1):
            a1 = rankdata(value, method='min', axis=axis)
            a2 = rank_2d(value, method='min', start=1, axis=axis)
            self.assertEqual(a1.tolist(), a2.tolist())

    @given(sfst.get_array_2d(dtype_group=sfst.DTGroup.NUMERIC, max_rows=20, max_columns=20))
    def test_rank_2d_max(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        for axis in (0, 1):
            a1 = rankdata(value, method='max', axis=axis)
            a2 = rank_2d(value, method='max', start=1, axis=axis)
            self.assertEqual(a1.tolist(), a2.tolist())

    @given(sfst.get_array_2d(dtype_group=sfst.DTGroup.NUMERIC, max_rows=20, max_columns=20))
    def test_rank_2d_average(self, value: np.ndarray) -> None:
        # cannot compare values with NaN as scipy uses quicksort
        if np.isnan(value).any():
            return
        for axis in (0, 1):
            a1 = rankdata(value, method='average', axis=axis)
            a2 = rank_2d(value, method='mean', start=1, axis=axis)
            self.assertEqual(a1.tolist(), a2.tolist())





if __name__ == '__main__':
    unittest.main()
