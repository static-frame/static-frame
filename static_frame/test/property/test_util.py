
import typing as tp
import unittest

import numpy as np  # type: ignore

from hypothesis import strategies as st
from hypothesis import given  # type: ignore
from hypothesis import example  # type: ignore
from hypothesis import reproduce_failure  # type: ignore

from static_frame.test.property.strategies import DTGroup

from static_frame.test.property.strategies import get_shape_1d2d
from static_frame.test.property.strategies import get_array_1d
from static_frame.test.property.strategies import get_array_1d2d
from static_frame.test.property.strategies import get_array_2d
from static_frame.test.property.strategies import get_dtype_pairs

from static_frame.test.property.strategies import get_dtype
from static_frame.test.property.strategies import get_dtypes
from static_frame.test.property.strategies import get_label
from static_frame.test.property.strategies import get_value
from static_frame.test.property.strategies import get_labels
from static_frame.test.property.strategies import get_arrays_2d_aligned_columns
from static_frame.test.property.strategies import get_arrays_2d_aligned_rows

from static_frame.core.operator_delegate import UFUNC_AXIS_SKIPNA

from static_frame.test.test_case import TestCase
from static_frame.core import util


class TestUnit(TestCase):


    @given(get_array_1d2d())  # type: ignore
    def test_mloc(self, array: np.ndarray) -> None:

        x = util.mloc(array)
        self.assertTrue(isinstance(x, int))


    @given(get_dtype_pairs())  # type: ignore
    def test_resolve_dtype(self, dtype_pair: tp.Tuple[np.dtype, np.dtype]) -> None:

        x = util.resolve_dtype(*dtype_pair)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_dtypes(min_size=1))  # type: ignore
    def test_resolve_dtype_iter(self, dtypes: tp.Iterable[np.dtype]) -> None:

        x = util.resolve_dtype_iter(dtypes)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_labels(min_size=1))  # type: ignore
    def test_resolve_type_object_iter(self, objects: tp.Iterable[object]) -> None:

        x = util.resolve_type_object_iter(objects)
        self.assertTrue(x in (None, str, object))

    @given(get_arrays_2d_aligned_columns())  # type: ignore
    def test_concat_resolved_axis_0(self, arrays: tp.List[np.ndarray]) -> None:
        array = util.concat_resolved(arrays, axis=0)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, util.resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_arrays_2d_aligned_rows())  # type: ignore
    def test_concat_resolved_axis_1(self, arrays: tp.List[np.ndarray]) -> None:
        array = util.concat_resolved(arrays, axis=1)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, util.resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_dtype(), get_shape_1d2d(), get_value())  # type: ignore
    def test_full_or_fill(self,
            dtype: np.dtype,
            shape: tp.Union[tp.Tuple[int], tp.Tuple[int, int]],
            value: object) -> None:
        array = util.full_for_fill(dtype, shape, fill_value=value)
        self.assertTrue(array.shape == shape)
        if isinstance(value, (float, complex)) and np.isnan(value):
            pass
        else:
            self.assertTrue(value in array)

    @given(get_dtype())  # type: ignore
    def test_dtype_to_na(self, dtype: util.DtypeSpecifier) -> None:
        post = util.dtype_to_na(dtype)
        self.assertTrue(post in {0, False, None, '', np.nan, util.NAT})


    @given(get_array_1d(min_size=1, dtype_group=DTGroup.NUMERIC))
    def test_ufunc_skipna_1d(self, array):

        has_na = util._isna(array).any()
        for ufunc, ufunc_skipna, dtype in UFUNC_AXIS_SKIPNA.values():
            v1 = ufunc_skipna(array)
            # this should return a single value
            self.assertFalse(isinstance(v1, np.ndarray))

            if has_na:
                v2 = ufunc(array)
                self.assertFalse(isinstance(v2, np.ndarray))

    @given(get_array_1d2d())
    def test_ufunc_unique(self, array):
        post = util.ufunc_unique(array, axis=0)
        self.assertTrue(len(post) <= array.shape[0])

    @given(get_array_1d(min_size=1), st.integers())
    def test_roll_1d(self, array, shift):
        post = util.roll_1d(array, shift)
        self.assertEqual(len(post), len(array))
        self.assertEqualWithNaN(array[-(shift % len(array))], post[0])

    @given(get_array_2d(min_rows=1, min_columns=1), st.integers())
    def test_roll_2d(self, array, shift):
        for axis in (0, 1):
            post = util.roll_2d(array, shift=shift, axis=axis)
            self.assertEqual(post.shape, array.shape)

            start = -(shift % array.shape[axis])

            if axis == 0:
                a = array[start]
                b = post[0]
            else:
                a = array[:, start]
                b = post[:, 0]

            self.assertAlmostEqualValues(a, b)


    @given(get_array_1d(dtype_group=DTGroup.OBJECT))
    def test_collection_to_array(self, array):
        values = array.tolist()
        post = util.collection_to_array(values, discover_dtype=True)
        self.assertAlmostEqualValues(array, post)

    @given(get_array_1d(dtype_group=DTGroup.OBJECT))
    def test_iterable_to_array(self, array):
        values = array.tolist()
        post, _ = util.iterable_to_array(values)
        self.assertAlmostEqualValues(post, values)

    @given(get_array_1d(dtype_group=DTGroup.OBJECT))
    def test_collection_and_dtype_to_1darray(self, array):
        values = array.tolist()
        post = util.collection_and_dtype_to_1darray(values, dtype=util.DTYPE_OBJECT)
        self.assertAlmostEqualValues(post, values)


    @given(st.slices(10))
    def test_slice_to_ascending_slice(self, key):

        post_key = util.slice_to_ascending_slice(key, size=10)
        self.assertEqual(
            set(range(*key.indices(10))),
            set(range(*post_key.indices(10)))
            )


if __name__ == '__main__':
    unittest.main()
