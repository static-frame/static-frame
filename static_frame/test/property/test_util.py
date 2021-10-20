
import typing as tp
import datetime
import unittest
import fractions
import numpy as np

from hypothesis import strategies as st
from hypothesis import given

from arraykit import mloc
from arraykit import shape_filter
from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter
from arraykit import isna_element


# from static_frame.test.property.strategies import get_arrays_2d_aligned
# from static_frame.test.property.strategies import get_blocks
# from static_frame.test.property.strategies import get_label
from static_frame.core import util
from static_frame.core.interface import UFUNC_AXIS_SKIPNA
from static_frame.core.util import DTYPE_INEXACT_KINDS
from static_frame.test.property.strategies import DTGroup
from static_frame.test.property.strategies import get_array_1d
from static_frame.test.property.strategies import get_array_1d2d
from static_frame.test.property.strategies import get_array_2d
from static_frame.test.property.strategies import get_arrays_2d_aligned_columns
from static_frame.test.property.strategies import get_arrays_2d_aligned_rows
from static_frame.test.property.strategies import get_dtype
from static_frame.test.property.strategies import get_dtype_pairs
from static_frame.test.property.strategies import get_dtypes
from static_frame.test.property.strategies import get_labels
from static_frame.test.property.strategies import get_shape_1d2d
from static_frame.test.property.strategies import get_value
from static_frame.test.test_case import TestCase

class TestUnit(TestCase):


    @given(get_array_1d2d())
    def test_mloc(self, array: np.ndarray) -> None:

        x = mloc(array)
        self.assertTrue(isinstance(x, int))

    @given(get_array_1d2d())
    def test_shape_filter(self, shape: np.ndarray) -> None:
        self.assertTrue(len(shape_filter(shape)), 2)

    @given(get_dtype_pairs())
    def test_resolve_dtype(self, dtype_pair: tp.Tuple[np.dtype, np.dtype]) -> None:

        x = resolve_dtype(*dtype_pair)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_dtypes(min_size=1))
    def test_resolve_dtype_iter(self, dtypes: tp.Iterable[np.dtype]) -> None:

        x = resolve_dtype_iter(dtypes)
        self.assertTrue(isinstance(x, np.dtype))

    @given(get_labels(min_size=1))
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
        resolved, has_tuple, values_post = util.prepare_iter_for_array(objects)
        self.assertTrue(resolved in known_types)



    @given(get_arrays_2d_aligned_columns())
    def test_concat_resolved_axis_0(self, arrays: tp.List[np.ndarray]) -> None:
        array = util.concat_resolved(arrays, axis=0)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_arrays_2d_aligned_rows())
    def test_concat_resolved_axis_1(self, arrays: tp.List[np.ndarray]) -> None:
        array = util.concat_resolved(arrays, axis=1)
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype, resolve_dtype_iter((x.dtype for x in arrays)))

    @given(get_dtype(), get_shape_1d2d(), get_value())
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

    @given(get_dtype())
    def test_dtype_to_na(self, dtype: util.DtypeSpecifier) -> None:
        post = util.dtype_to_fill_value(dtype)
        self.assertTrue(post in {0, False, None, '', np.nan, util.NAT})


    @given(get_array_1d2d(dtype_group=DTGroup.NUMERIC))
    def test_ufunc_axis_skipna(self, array: np.ndarray) -> None:

        has_na = util.isna_array(array).any()

        for nt in UFUNC_AXIS_SKIPNA.values():
            ufunc = nt.ufunc
            ufunc_skipna = nt.ufunc_skipna
            # dtypes = nt.dtypes
            # composable = nt.composable
            # doc = nt.doc_header
            # size_one_unity = nt.size_one_unity

            with np.errstate(over='ignore', under='ignore', divide='ignore'):

                post = util.array_ufunc_axis_skipna(array=array,
                        skipna=True,
                        axis=0,
                        ufunc=ufunc,
                        ufunc_skipna=ufunc_skipna
                        )
                if array.ndim == 2:
                    self.assertTrue(post.ndim == 1)

    @given(get_array_1d2d())
    def test_ufunc_unique(self, array: np.ndarray) -> None:
        post = util.ufunc_unique(array, axis=0)
        self.assertTrue(len(post) <= array.shape[0])

    @given(get_array_1d(min_size=1), st.integers())
    def test_roll_1d(self, array: np.ndarray, shift: int) -> None:
        post = util.roll_1d(array, shift)
        self.assertEqual(len(post), len(array))
        self.assertEqualWithNaN(array[-(shift % len(array))], post[0])

    @given(get_array_2d(min_rows=1, min_columns=1), st.integers())
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



    @given(get_array_1d(dtype_group=DTGroup.OBJECT))
    def test_iterable_to_array_a(self, array: np.ndarray) -> None:
        values = array.tolist()
        post, _ = util.iterable_to_array_1d(values)
        self.assertAlmostEqualValues(post, values)

        # explicitly giving object dtype
        post, _ = util.iterable_to_array_1d(values, dtype=util.DTYPE_OBJECT)
        self.assertAlmostEqualValues(post, values)


    @given(get_labels())
    def test_iterable_to_array_b(self, labels: tp.Iterable[tp.Any]) -> None:
        post, _ = util.iterable_to_array_1d(labels)
        self.assertAlmostEqualValues(post, labels)
        self.assertTrue(isinstance(post, np.ndarray))


    @given(get_labels())
    def test_iterable_to_array_nd(self, labels: tp.Iterable[tp.Any]) -> None:
        post = util.iterable_to_array_nd(labels)
        self.assertAlmostEqualValues(post, labels)
        self.assertTrue(isinstance(post, np.ndarray))

        if len(labels): #type: ignore
            sample = post[0]
            post = util.iterable_to_array_nd(sample)
            self.assertTrue(isinstance(post, np.ndarray))

    @given(st.slices(10))  #pylint: disable=E1120
    def test_slice_to_ascending_slice(self, key: slice) -> None:

        post_key = util.slice_to_ascending_slice(key, size=10)
        self.assertEqual(
            set(range(*key.indices(10))),
            set(range(*post_key.indices(10)))
            )

# to_datetime64
# to_timedelta64
# key_to_datetime_key

    @given(get_array_1d2d())
    def test_array_to_groups_and_locations(self, array: np.ndarray) -> None:

        groups, locations = util.array_to_groups_and_locations(array, 0)

        if len(array) > 0:
            self.assertTrue(len(groups) >= 1)

        # always 1dm locations
        self.assertTrue(locations.ndim == 1)
        self.assertTrue(len(np.unique(locations)) == len(groups))


    @given(get_array_1d2d())
    def test_isna_array(self, array: np.ndarray) -> None:

        post = util.isna_array(array)
        self.assertTrue(post.dtype == bool)

        values = np.ravel(array)
        count_na = sum(isna_element(x) for x in values)

        self.assertTrue(np.ravel(post).sum() == count_na)


    @given(get_array_1d(dtype_group=DTGroup.BOOL))
    def test_binary_transition(self, array: np.ndarray) -> None:
        post = util.binary_transition(array)

        # could be 32 via result of np.nonzero
        self.assertTrue(post.dtype in (np.int32, np.int64))

        # if no True in original array, result will be empty
        if array.sum() == 0:
            self.assertTrue(len(post) == 0)
        # if all True, result is empty
        elif array.sum() == len(array):
            self.assertTrue(len(post) == 0)
        else:
            # the post selection shold always be indices that are false
            self.assertTrue(array[post].sum() == 0)


    @given(get_array_1d2d())
    def test_array_to_duplicated(self, array: np.ndarray) -> None:
        if array.ndim == 2:
            for axis in (0, 1):
                post = util.array_to_duplicated(array, axis=axis)
                if axis == 0:
                    unique_count = len(set(tuple(x) for x in array))
                else:
                    unique_count = len(set(
                        tuple(array[:, i]) for i in range(array.shape[1]))
                        )
                if unique_count < array.shape[axis]:
                    self.assertTrue(post.sum() > 0)
        else:
            post = util.array_to_duplicated(array)
            # if not all value are unique, we must have some duplicated
            if len(set(array)) < len(array):
                self.assertTrue(post.sum() > 0)

        self.assertTrue(post.dtype == bool)

    @given(get_array_1d2d())
    def test_array_shift(self, array: np.ndarray) -> None:

        for shift in (-1, 1):
            for wrap in (True, False):

                tests = []
                post1 = util.array_shift(
                        array=array,
                        shift=shift,
                        axis=0,
                        wrap=wrap)
                tests.append(post1)

                if array.ndim == 2:
                    post2 = util.array_shift(
                        array=array,
                        shift=shift,
                        axis=1,
                        wrap=wrap)
                    tests.append(post2)

                for post in tests:
                    self.assertTrue(array.shape == post.shape)

                    # type is only always maintained if we are wrapping
                    if wrap:
                        self.assertTrue(array.dtype == post.dtype)


    @given(st.lists(get_array_1d(), min_size=2, max_size=2))
    def test_union1d(self, arrays: tp.Sequence[np.ndarray]) -> None:
        post = util.union1d(
                arrays[0],
                arrays[1],
                assume_unique=False)
        self.assertTrue(post.ndim == 1)

        # the unqiueness of NaNs has changed in newer NP versions, so only compare if non-nans are found
        if post.dtype.kind in ('c', 'f') and not np.isnan(post).any():
            self.assertTrue(len(post) == len(set(arrays[0]) | set(arrays[1])))
        # complex results are tricky to compare after forming sets
        if (post.dtype.kind not in ('O', 'M', 'm', 'c', 'f')
                and not np.isnan(post).any()):
            self.assertSetEqual(set(post), (set(arrays[0]) | set(arrays[1])))


    @given(st.lists(get_array_1d(), min_size=2, max_size=2))
    def test_intersect1d(self, arrays: tp.Sequence[np.ndarray]) -> None:
        post = util.intersect1d(
                arrays[0],
                arrays[1],
                assume_unique=False)
        self.assertTrue(post.ndim == 1)
        # nan values in complex numbers make direct comparison tricky
        self.assertTrue(len(post) == len(set(arrays[0]) & set(arrays[1])))

        if (post.dtype.kind not in ('O', 'M', 'm', 'c', 'f')
                and not np.isnan(post).any()):
            self.assertSetEqual(set(post), (set(arrays[0]) & set(arrays[1])))


    @given(st.lists(get_array_1d(), min_size=2, max_size=2))
    def test_setdiff1d(self, arrays: tp.Sequence[np.ndarray]) -> None:
        post = util.setdiff1d(
                arrays[0],
                arrays[1],
                assume_unique=False)
        self.assertTrue(post.ndim == 1)

        if post.dtype.kind in ('f', 'c', 'i', 'u'):
            # Compare directly to numpy behavior for number values.
            self.assertTrue(len(post) == len(np.setdiff1d(arrays[0], arrays[1], assume_unique=False)))
        else:
            # nan values in complex numbers make direct comparison tricky
            self.assertTrue(len(post) == len(set(arrays[0]).difference(set(arrays[1]))))

        if (post.dtype.kind not in ('O', 'M', 'm', 'c', 'f')
                and not np.isnan(post).any()):
            self.assertSetEqual(set(post), (set(arrays[0]).difference(set(arrays[1]))))


    #---------------------------------------------------------------------------
    @given(get_arrays_2d_aligned_columns(min_size=2, max_size=2))
    def test_union2d(self, arrays: tp.Sequence[np.ndarray]) -> None:
        post = util.union2d(arrays[0], arrays[1], assume_unique=False)
        self.assertTrue(post.ndim == 2)

        if post.dtype.kind in ('f', 'c') and np.isnan(post).any():
            return

        self.assertTrue(len(post) == len(
                set(util.array2d_to_tuples(arrays[0]))
                | set(util.array2d_to_tuples(arrays[1])))
                )

    @given(get_arrays_2d_aligned_columns(min_size=2, max_size=2))
    def test_intersect2d(self, arrays: tp.Sequence[np.ndarray]) -> None:
        post = util.intersect2d(arrays[0], arrays[1], assume_unique=False)
        self.assertTrue(post.ndim == 2)
        self.assertTrue(len(post) == len(
                set(util.array2d_to_tuples(arrays[0]))
                & set(util.array2d_to_tuples(arrays[1])))
                )

    @given(get_arrays_2d_aligned_columns(min_size=2, max_size=2))
    def test_setdiff2d(self, arrays: tp.Sequence[np.ndarray]) -> None:

        for array in arrays:
            if array.dtype.kind in ('f', 'c') and np.isnan(array).any():
                return

        post = util.setdiff2d(arrays[0], arrays[1], assume_unique=False)
        self.assertTrue(post.ndim == 2)
        self.assertTrue(len(post) == len(
                set(util.array2d_to_tuples(arrays[0])).difference(
                set(util.array2d_to_tuples(arrays[1]))))
                )

    @given(get_arrays_2d_aligned_columns())
    def test_array_set_ufunc_many(self, arrays: tp.Sequence[np.ndarray]) -> None:

        for union in (True, False):
            post = util.ufunc_set_iter(arrays, union=union)
            self.assertTrue(post.ndim == 2)

    #---------------------------------------------------------------------------
    @given(get_array_1d2d(min_rows=1, min_columns=1))
    def test_isin(self, array: np.ndarray) -> None:

        container_factory = (list, set, np.array)
        result = None

        if array.ndim == 1:
            sample = array[0]
            if np.array(sample).dtype.kind in DTYPE_INEXACT_KINDS and np.isnan(sample):
                pass
            else:
                for factory in container_factory:
                    result = util.isin(array, factory((sample,)))
                    self.assertTrue(result[0])
        elif array.ndim == 2:
            sample = array[0, 0]
            if np.array(sample).dtype.kind in DTYPE_INEXACT_KINDS and np.isnan(sample):
                pass
            else:
                for factory in container_factory:
                    result = util.isin(array, factory((sample,)))
                    self.assertTrue(result[0, 0])

        if result is not None:
            self.assertTrue(array.shape == result.shape)
            self.assertTrue(result.dtype == bool)




if __name__ == '__main__':
    unittest.main()
