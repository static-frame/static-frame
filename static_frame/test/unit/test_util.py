

import unittest

import numpy as np

from static_frame.core.util import _isna
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import array_set_ufunc_many

from static_frame.core.util import intersect2d
from static_frame.core.util import union2d
from static_frame.core.util import concat_resolved


from static_frame.core.util import _gen_skip_middle
from static_frame.core.util import _dtype_to_na
from static_frame.core.util import key_to_datetime_key

from static_frame.core.operator_delegate import _ufunc_logical_skipna

from static_frame.core.util import _read_url
from static_frame.core.util import _ufunc2d

from static_frame import Index

# TODO test
from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import iterable_to_array
from static_frame.core.util import _array_to_groups_and_locations
from static_frame.core.util import IndexCorrespondence

from static_frame.core.util import _slice_to_ascending_slice
from static_frame.core.util import array_shift
from static_frame.core.util import ufunc_unique

from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_gen_skip_middle_a(self):

        forward = lambda: [3, 2, 5]
        reverse = lambda: [3, 2, 5]

        post = _gen_skip_middle(
                forward_iter=forward,
                forward_count=3,
                reverse_iter=reverse,
                reverse_count=3,
                center_sentinel=-1)

        self.assertEqual(list(post), [3, 2, 5, -1, 5, 2, 3])

        post = _gen_skip_middle(
                forward_iter=forward,
                forward_count=2,
                reverse_iter=reverse,
                reverse_count=2,
                center_sentinel=0)

        self.assertEqual(list(post), [3, 2, 0, 2, 3])




    def test_ufunc_logical_skipna_a(self):

        # empty arrays
        a1 = np.array([], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False), True)

        a1 = np.array([], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=False), False)


        # float arrays 1d
        a1 = np.array([2.4, 5.4], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=True), True)

        a1 = np.array([2.4, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False), False)

        a1 = np.array([0, np.nan, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=True), False)

        a1 = np.array([0, np.nan, 0], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.any, skipna=False), True)


        # float arrays 2d
        a1 = np.array([[2.4, 5.4, 3.2], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, True])


        a1 = np.array([[2.4, 5.4, 3.2], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [True, True])

        a1 = np.array([[2.4, 5.4, 0], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, False])

        a1 = np.array([[2.4, 5.4, 0], [2.4, 5.4, 3.2]], dtype=float)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, True])


        # object arrays
        a1 = np.array([[2.4, 5.4, 0], [2.4, None, 3.2]], dtype=object)

        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False])
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, False, False])


        a1 = np.array([[2.4, 5.4, 0], [2.4, np.nan, 3.2]], dtype=object)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, True, False])


    def test_ufunc_logical_skipna_b(self):
        # object arrays

        a1 = np.array([['sdf', '', 'wer'], [True, False, True]], dtype=object)

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True, False, True]
                )
        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False]
                )


        # string arrays
        a1 = np.array(['sdf', ''], dtype=str)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=False, axis=0), False)
        self.assertEqual(_ufunc_logical_skipna(a1, np.all, skipna=True, axis=0), False)


        a1 = np.array([['sdf', '', 'wer'], ['sdf', '', 'wer']], dtype=str)
        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=0).tolist(),
                [True,  False,  True])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.all, skipna=False, axis=1).tolist(),
                [False, False])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.any, skipna=False, axis=0).tolist(),
                [True,  False,  True])

        self.assertEqual(
                _ufunc_logical_skipna(a1, np.any, skipna=False, axis=1).tolist(),
                [True, True])


    def test_ufunc_logical_skipna_c(self):

        a1 = np.array([], dtype=float)
        with self.assertRaises(NotImplementedError):
            _ufunc_logical_skipna(a1, np.sum, skipna=True)


    def test_ufunc_logical_skipna_d(self):

        a1 = np.array(['2018-01-01', '2018-02-01'], dtype=np.datetime64)
        post = _ufunc_logical_skipna(a1, np.all, skipna=True)
        self.assertTrue(post)


    def test_ufunc_logical_skipna_e(self):

        a1 = np.array([['2018-01-01', '2018-02-01'],
                ['2018-01-01', '2018-02-01']], dtype=np.datetime64)
        post = _ufunc_logical_skipna(a1, np.all, skipna=True)
        self.assertEqual(post.tolist(), [True, True])




    def test_resolve_dtype_a(self):

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(_resolve_dtype(a1.dtype, a1.dtype), a1.dtype)

        self.assertEqual(_resolve_dtype(a1.dtype, a2.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a2.dtype, a3.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a2.dtype, a4.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a3.dtype, a4.dtype), np.object_)
        self.assertEqual(_resolve_dtype(a3.dtype, a6.dtype), np.object_)

        self.assertEqual(_resolve_dtype(a1.dtype, a4.dtype), np.float64)
        self.assertEqual(_resolve_dtype(a1.dtype, a6.dtype), np.float64)
        self.assertEqual(_resolve_dtype(a4.dtype, a6.dtype), np.float64)

    def test_resolve_dtype_b(self):

        self.assertEqual(
                _resolve_dtype(np.array('a').dtype, np.array('aaa').dtype),
                np.dtype(('U', 3))
                )

    def test_resolve_dtype_iter_a(self):

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(resolve_dtype_iter((a1.dtype, a1.dtype)), a1.dtype)
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype)), a2.dtype)

        # boolean with mixed types
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype, a3.dtype)), np.object_)
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype, a5.dtype)), np.object_)
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype, a6.dtype)), np.object_)

        # numerical types go to float64
        self.assertEqual(resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype)), np.float64)

        # add in bool or str, goes to object
        self.assertEqual(resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a2.dtype)), np.object_)
        self.assertEqual(resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a5.dtype)), np.object_)

        # mixed strings go to the largest
        self.assertEqual(resolve_dtype_iter((a3.dtype, a5.dtype)), np.dtype('<U10'))



    def test_isna_array_a(self):

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3, 5.4], dtype='float32')

        self.assertEqual(_isna(a1).tolist(), [False, False, False])
        self.assertEqual(_isna(a2).tolist(), [False, False, False])
        self.assertEqual(_isna(a3).tolist(), [False, False, False])
        self.assertEqual(_isna(a4).tolist(), [False, False])
        self.assertEqual(_isna(a5).tolist(), [False, False])
        self.assertEqual(_isna(a6).tolist(), [False, False])

        a1 = np.array([1, 2, 3, None])
        a2 = np.array([False, True, False, None])
        a3 = np.array(['b', 'c', 'd', None])
        a4 = np.array([2.3, 3.2, None])
        a5 = np.array(['test', 'test again', None])
        a6 = np.array([2.3, 5.4, None])

        self.assertEqual(_isna(a1).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a2).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a3).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a4).tolist(), [False, False, True])
        self.assertEqual(_isna(a5).tolist(), [False, False, True])
        self.assertEqual(_isna(a6).tolist(), [False, False, True])

        a1 = np.array([1, 2, 3, np.nan])
        a2 = np.array([False, True, False, np.nan])
        a3 = np.array(['b', 'c', 'd', np.nan], dtype=object)
        a4 = np.array([2.3, 3.2, np.nan], dtype=object)
        a5 = np.array(['test', 'test again', np.nan], dtype=object)
        a6 = np.array([2.3, 5.4, np.nan], dtype='float32')

        self.assertEqual(_isna(a1).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a2).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a3).tolist(), [False, False, False, True])
        self.assertEqual(_isna(a4).tolist(), [False, False, True])
        self.assertEqual(_isna(a5).tolist(), [False, False, True])
        self.assertEqual(_isna(a6).tolist(), [False, False, True])


    def test_isna_array_b(self):

        a1 = np.array([[1, 2], [3, 4]])
        a2 = np.array([[False, True, False], [False, True, False]])
        a3 = np.array([['b', 'c', 'd'], ['b', 'c', 'd']])
        a4 = np.array([[2.3, 3.2, np.nan], [2.3, 3.2, np.nan]])
        a5 = np.array([['test', 'test again', np.nan],
                ['test', 'test again', np.nan]], dtype=object)
        a6 = np.array([[2.3, 5.4, np.nan], [2.3, 5.4, np.nan]], dtype='float32')

        self.assertEqual(_isna(a1).tolist(),
                [[False, False], [False, False]])

        self.assertEqual(_isna(a2).tolist(),
                [[False, False, False], [False, False, False]])

        self.assertEqual(_isna(a3).tolist(),
                [[False, False, False], [False, False, False]])

        self.assertEqual(_isna(a4).tolist(),
                [[False, False, True], [False, False, True]])

        self.assertEqual(_isna(a5).tolist(),
                [[False, False, True], [False, False, True]])

        self.assertEqual(_isna(a6).tolist(),
                [[False, False, True], [False, False, True]])


    def test_array_to_duplicated_a(self):
        a = _array_to_duplicated(
                np.array([0,1,2,2,1,4,5,3,4,5,5,6]),
                exclude_first=False,
                exclude_last=False
                )
        self.assertEqual(a.tolist(),
                [False, True, True, True, True, True, True, False, True, True, True, False])

        a = _array_to_duplicated(
                np.array([0,1,2,2,1,4,5,3,4,5,5,6]),
                exclude_first=True,
                exclude_last=False
                )
        self.assertEqual(a.tolist(),
                [False, False, False, True, True, False, False, False, True, True, True, False])


    def test_array_to_duplicated_b(self):
        a = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        # find duplicate rows
        post = _array_to_duplicated(a, axis=0)
        self.assertEqual(post.tolist(),
                [False, False])

        post = _array_to_duplicated(a, axis=1)
        self.assertEqual(post.tolist(),
                [True, True, False, True, True])

        post = _array_to_duplicated(a, axis=1, exclude_first=True)
        self.assertEqual(post.tolist(),
                [False, True, False, False, True])


    def test_array_to_duplicated_c(self):
        a = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        with self.assertRaises(Exception):
            _array_to_duplicated(a, axis=None)



    def test_array_set_ufunc_many_a(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2, 1])
        a3 = np.array([3, 2, 1])
        a4 = np.array([3, 2, 1])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=False)
        self.assertEqual(post.tolist(), [3, 2, 1])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=True)
        self.assertEqual(post.tolist(), [3, 2, 1])


    def test_array_set_ufunc_many_b(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2])
        a3 = np.array([5, 3, 2, 1])
        a4 = np.array([2])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=False)
        self.assertEqual(post.tolist(), [2])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=True)
        self.assertEqual(post.tolist(), [1, 2, 3, 5])


    def test_array_set_ufunc_many_c(self):
        a1 = np.array([[3, 2, 1], [1, 2, 3]])
        a2 = np.array([[5, 2, 1], [1, 2, 3]])
        a3 = np.array([[10, 20, 30], [1, 2, 3]])

        post = array_set_ufunc_many((a1, a2, a3), union=False)
        self.assertEqual(post.tolist(), [[1, 2, 3]])

        post = array_set_ufunc_many((a1, a2, a3), union=True)
        self.assertEqual(post.tolist(),
                [[1, 2, 3], [3, 2, 1], [5, 2, 1], [10, 20, 30]])


    def test_array_set_ufunc_many_d(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([[5, 2, 1], [1, 2, 3]])

        with self.assertRaises(Exception):
            post = array_set_ufunc_many((a1, a2), union=False)


    def test_array_set_ufunc_many_e(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([30, 20])

        post = array_set_ufunc_many((a1, a2), union=False)
        self.assertEqual(post.tolist(), [])


    def test_intersect2d_a(self):
        a = np.array([('a', 'b'), ('c', 'd'), ('e', 'f')])
        b = np.array([('a', 'g'), ('c', 'd'), ('e', 'f')])

        post = intersect2d(a, b)
        self.assertEqual(post.tolist(),
                [['c', 'd'], ['e', 'f']])

        post = intersect2d(a.astype(object), b.astype(object))
        self.assertEqual(post.tolist(),
                [['c', 'd'], ['e', 'f']])

        post = union2d(a, b)
        self.assertEqual(post.tolist(),
                [['a', 'b'], ['a', 'g'], ['c', 'd'], ['e', 'f']]
                )
        post = union2d(a.astype(object), b.astype(object))
        self.assertEqual(post.tolist(),
                [['a', 'b'], ['a', 'g'], ['c', 'd'], ['e', 'f']]
                )

    @unittest.skip('requires network')
    def test_read_url(self):
        url = 'https://jsonplaceholder.typicode.com/todos'
        post = _read_url(url)


    def test_slice_to_ascending_slice_a(self):

        a1 = np.arange(10)

        def compare(slc):
            slc_asc = _slice_to_ascending_slice(slc, len(a1))
            self.assertEqual(sorted(a1[slc]), list(a1[slc_asc]))
        #     print(slc, a1[slc])
        #     print(slc_asc, a1[slc_asc])

        compare(slice(4,))
        compare(slice(6, 1, -1))
        compare(slice(6, 1, -2))
        compare(slice(6, None, -3))
        compare(slice(6, 2, -2))
        compare(slice(None, 1, -1))


    def test_array_shift_a(self):
        a1 = np.arange(6)


        # import ipdb; ipdb.set_trace()

        self.assertEqual(array_shift(a1, 2, axis=0, wrap=True).tolist(),
                [4, 5, 0, 1, 2, 3])
        self.assertEqual(array_shift(a1, -2, axis=0, wrap=True).tolist(),
                [2, 3, 4, 5, 0, 1])
        self.assertEqual(array_shift(a1, 5, axis=0, wrap=True).tolist(),
                [1, 2, 3, 4, 5, 0])

        self.assertEqual(
                array_shift(a1, 2, axis=0, wrap=False, fill_value=-1).tolist(),
                [-1, -1, 0, 1, 2, 3])

        self.assertEqual(
                array_shift(a1, 2, axis=0, wrap=False, fill_value=1.5).tolist(),
                [1.5, 1.5, 0, 1, 2, 3])

        self.assertEqual(
                array_shift(a1, -2, axis=0, wrap=False, fill_value=1.5).tolist(),
                [2, 3, 4, 5, 1.5, 1.5])


    def test_array_shift_b(self):
        a1 = np.array([('a', 'b', 'e', 'd'),
                ('c', 'd', 'f', 'w'),
                ('e', 'f', 's', 'q')])

        self.assertEqual(array_shift(a1, 2, axis=0, wrap=True).tolist(),
                [['c', 'd', 'f', 'w'], ['e', 'f', 's', 'q'], ['a', 'b', 'e', 'd']])

        self.assertEqual(array_shift(a1, -2, axis=0, wrap=True).tolist(),
                [['e', 'f', 's', 'q'], ['a', 'b', 'e', 'd'], ['c', 'd', 'f', 'w']])


        self.assertEqual(
                array_shift(a1, -2, axis=0, wrap=False, fill_value='XX').dtype,
                np.dtype('<U2')
                )

        self.assertEqual(
                array_shift(a1, -2, axis=0, wrap=False, fill_value='XX').tolist(),
                [['e', 'f', 's', 'q'],
                ['XX', 'XX', 'XX', 'XX'],
                ['XX', 'XX', 'XX', 'XX']])

        self.assertEqual(
                array_shift(a1, 2, axis=1, wrap=False, fill_value='XX').tolist(),
                [['XX', 'XX', 'a', 'b'],
                ['XX', 'XX', 'c', 'd'],
                ['XX', 'XX', 'e', 'f']])

        self.assertEqual(
                array_shift(a1, -2, axis=1, wrap=False, fill_value='XX').tolist(),
                [['e', 'd', 'XX', 'XX'],
                ['f', 'w', 'XX', 'XX'],
                ['s', 'q', 'XX', 'XX']])


    def test_array_shift_c(self):
        a1 = np.arange(6)
        post = array_shift(a1, 0, axis=0, wrap=False)
        self.assertEqual(a1.tolist(), post.tolist())


    def test_ufunc_unique_a(self):

        a1 = np.array([1, 1, 1, 2, 2])
        post = ufunc_unique(a1)
        self.assertEqual(post.tolist(), [1, 2])

        a2 = np.array([1, 1, 1, 2, 2], dtype=object)
        post = ufunc_unique(a2)
        self.assertEqual(post.tolist(), [1, 2])

        a3 = np.array([1, 'x', 1, None, 2], dtype=object)
        post = ufunc_unique(a3)
        self.assertEqual(post, {None, 1, 2, 'x'})


    def test_ufunc_unique_b(self):

        a1 = np.array([[1, 1], [1, 2], [1, 2]])
        post = ufunc_unique(a1)
        self.assertEqual(post.tolist(), [1, 2])

        post = ufunc_unique(a1, axis=0)
        self.assertEqual(post.tolist(), [[1, 1], [1, 2]])

        post = ufunc_unique(a1, axis=1)
        self.assertEqual(post.tolist(), [[1, 1], [1, 2], [1, 2]])


    def test_ufunc_unique_c(self):

        a1 = np.array([[1, 'x', 1], [1, None, 1], [1, 'x', 1]], dtype=object)

        post = ufunc_unique(a1)
        self.assertEqual(post, {'x', 1, None})

        post = ufunc_unique(a1, axis=0)
        self.assertEqual(post, {(1, 'x', 1), (1, None, 1)})

        post = ufunc_unique(a1, axis=1)
        self.assertEqual(post, {(1, 1, 1), ('x', None, 'x')})


    def test_concat_resolved_a(self):
        a1 = np.array([[3,4,5],[0,0,0]])
        a2 = np.array([1,2,3]).reshape((1,3))
        a3 = np.array([('3', '4', '5'),('1','1','1')])
        a4 = np.array(['3', '5'])
        a5 = np.array([1, 1, 1])

        post = concat_resolved((a1, a3))
        self.assertEqual(
                post.tolist(),
                [[3, 4, 5], [0, 0, 0], ['3', '4', '5'], ['1', '1', '1']]
                )

        post = concat_resolved((a3, a1, a2))
        self.assertEqual(post.tolist(),
                [['3', '4', '5'], ['1', '1', '1'], [3, 4, 5], [0, 0, 0], [1, 2, 3]])

        self.assertEqual(concat_resolved((a1, a3), axis=1).tolist(),
                [[3, 4, 5, '3', '4', '5'], [0, 0, 0, '1', '1', '1']]
                )

        self.assertEqual(concat_resolved((a4, a5)).tolist(),
                ['3', '5', 1, 1, 1])


    def test_concat_resolved_b(self):
        a1 = np.array([[3,4,5],[0,0,0]])
        a2 = np.array([1,2,3]).reshape((1,3))

        with self.assertRaises(Exception):
            concat_resolved((a1, a2), axis=None)


    def test_dtype_to_na_a(self):

        self.assertEqual(_dtype_to_na(np.dtype(int)), 0)
        self.assertTrue(np.isnan(_dtype_to_na(np.dtype(float))))
        self.assertEqual(_dtype_to_na(np.dtype(bool)), False)
        self.assertEqual(_dtype_to_na(np.dtype(object)), None)
        self.assertEqual(_dtype_to_na(np.dtype(str)), '')


    def test_key_to_datetime_key_a(self):

        post = key_to_datetime_key(slice('2018-01-01', '2019-01-01'))
        self.assertEqual(post,
                slice(np.datetime64('2018-01-01'),
                np.datetime64('2019-01-01'), None))

        post = key_to_datetime_key(np.datetime64('2019-01-01'))
        self.assertEqual(post, np.datetime64('2019-01-01'))

        post = key_to_datetime_key('2019-01-01')
        self.assertEqual(post, np.datetime64('2019-01-01'))

        a1 = np.array(('2019-01-01'), dtype='M')
        post = key_to_datetime_key(a1)
        self.assertEqual(post, a1)

        post = key_to_datetime_key(np.array(['2018-01-01', '2019-01-01']))
        a2 = np.array(['2018-01-01', '2019-01-01'], dtype='datetime64[D]')
        self.assertEqual(post.tolist(), a2.tolist())

        post = key_to_datetime_key(['2018-01-01', '2019-01-01'])
        a3 = np.array(['2018-01-01', '2019-01-01'], dtype='datetime64[D]')
        self.assertEqual(post.tolist(), a3.tolist())

        post = key_to_datetime_key(['2018-01', '2019-01'])
        a4 = np.array(['2018-01', '2019-01'], dtype='datetime64[M]')
        self.assertEqual(post.tolist(), a4.tolist())


        post = key_to_datetime_key(str(x) for x in range(2012, 2015))
        a5 = np.array(['2012', '2013', '2014'], dtype='datetime64[Y]')
        self.assertEqual(post.tolist(), a5.tolist())

        post = key_to_datetime_key(None)
        self.assertEqual(post, None)


    def test_ufunc2d_a(self):

        a1 = np.array([1, 1, 1])
        with self.assertRaises(Exception):
            _ufunc2d(np.sum, a1, a1)


    def test_ufunc2d_b(self):

        a1 = np.array([['a', 'b'], ['b', 'c']])
        a2 = np.array([['b', 'cc'], ['dd', 'ee']])

        post = _ufunc2d(np.union1d, a1, a2)
        self.assertEqual(len(post), 4)
        self.assertEqual(str(post.dtype), '<U2')

        post = _ufunc2d(np.union1d, a2, a1)
        self.assertEqual(len(post), 4)
        self.assertEqual(str(post.dtype), '<U2')




if __name__ == '__main__':
    unittest.main()

