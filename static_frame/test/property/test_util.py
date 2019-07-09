
import typing as tp
import datetime
import unittest
import fractions
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
    def test_resolve_type_iter(self, objects: tp.Iterable[object]) -> None:

        known_types = set((
                None,
                type(None),
                bool,
                str,
                object,
                int,
                float,
                complex,
                datetime.date,
                datetime.datetime,
                fractions.Fraction
                ))
        resolved, has_tuple, values_post = util.resolve_type_iter(objects)
        self.assertTrue(resolved in known_types)



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


    @given(get_array_1d(min_size=1, dtype_group=DTGroup.NUMERIC)) # type: ignore
    def test_ufunc_skipna_1d(self, array: np.ndarray) -> None:

        has_na = util.isna_array(array).any()
        for ufunc, ufunc_skipna, dtype in UFUNC_AXIS_SKIPNA.values():
            v1 = ufunc_skipna(array)
            # this should return a single value
            self.assertFalse(isinstance(v1, np.ndarray))

            if has_na:
                v2 = ufunc(array)
                self.assertFalse(isinstance(v2, np.ndarray))

    @given(get_array_1d2d()) # type: ignore
    def test_ufunc_unique(self, array: np.ndarray) -> None:
        post = util.ufunc_unique(array, axis=0)
        self.assertTrue(len(post) <= array.shape[0])

    @given(get_array_1d(min_size=1), st.integers()) # type: ignore
    def test_roll_1d(self, array: np.ndarray, shift: int) -> None:
        post = util.roll_1d(array, shift)
        self.assertEqual(len(post), len(array))
        self.assertEqualWithNaN(array[-(shift % len(array))], post[0])

    @given(get_array_2d(min_rows=1, min_columns=1), st.integers()) # type: ignore
    def test_roll_2d(self, array: np.ndarray, shift: int) -> None:
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



    @given(get_array_1d(dtype_group=DTGroup.OBJECT)) # type: ignore
    def test_iterable_to_array_a(self, array: np.ndarray) -> None:
        values = array.tolist()
        post, _ = util.iterable_to_array(values)
        self.assertAlmostEqualValues(post, values)

        # explicitly giving object dtype
        post, _ = util.iterable_to_array(values, dtype=util.DTYPE_OBJECT)
        self.assertAlmostEqualValues(post, values)


    @given(get_labels()) # type: ignore
    def test_iterable_to_array_b(self, labels: tp.Iterable[tp.Any]) -> None:
        post, _ = util.iterable_to_array(labels)
        self.assertAlmostEqualValues(post, labels)
        self.assertTrue(isinstance(post, np.ndarray))


    @given(st.slices(10)) # type: ignore
    def test_slice_to_ascending_slice(self, key: slice) -> None:

        post_key = util.slice_to_ascending_slice(key, size=10)
        self.assertEqual(
            set(range(*key.indices(10))),
            set(range(*post_key.indices(10)))
            )

# to_datetime64
# to_timedelta64
# key_to_datetime_key

    @given(get_array_1d2d()) # type: ignore
    def test_array_to_groups_and_locations(self, array: np.ndarray) -> None:

        groups, locations = util.array_to_groups_and_locations(array, 0)

        if len(array) > 0:
            self.assertTrue(len(groups) >= 1)

        # always 1dm locations
        self.assertTrue(locations.ndim == 1)
        self.assertTrue(len(np.unique(locations)) == len(groups))


    @given(get_array_1d2d()) # type: ignore
    def test_isna_array(self, array: np.ndarray) -> None:

        post = util.isna_array(array)
        self.assertTrue(post.dtype == bool)

        values = np.ravel(array)
        count_na = sum(util.isna_element(x) for x in values)

        self.assertTrue(np.ravel(post).sum() == count_na)


    @given(get_array_1d(dtype_group=DTGroup.BOOL)) # type: ignore
    def test_binary_transition(self, array: np.ndarray) -> None:
        post = util.binary_transition(array)

        self.assertTrue(post.dtype == util.DEFAULT_INT_DTYPE)

        # if no True in original array, result will be empty
        if array.sum() == 0:
            self.assertTrue(len(post) == 0)
        # if all True, result is empty
        elif array.sum() == len(array):
            self.assertTrue(len(post) == 0)
        else:
            # the post selection shold always be indices that are false
            self.assertTrue(array[post].sum() == 0)


    # NOTE: temporarily only using numeric types;
    # @given(get_array_1d2d(dtype_group=DTGroup.NUMERIC))
    # def test_array_to_duplicated(self, array: np.ndarray) -> None:
    #     if array.ndim == 2:
    #         for axis in (0, 1):
    #             post = util.array_to_duplicated(array, axis=axis)
    #     else:
    #         post = util.array_to_duplicated(array)
    #         # if not all value are unique, we must have some duplicated
    #         if np.unique(array) < len(array):
    #             self.assertTrue(post.sum() > 0)

    #     self.assertTrue(post.dtype == bool)


if __name__ == '__main__':
    unittest.main()
