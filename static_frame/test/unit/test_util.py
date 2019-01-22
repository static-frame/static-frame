

import unittest

import numpy as np

from static_frame.core.util import _isna
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _resolve_dtype_iter
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import _array_set_ufunc_many

from static_frame.core.util import _intersect2d
from static_frame.core.util import _union2d


from static_frame.core.util import _gen_skip_middle
from static_frame.core.operator_delegate import _ufunc_logical_skipna

from static_frame.core.util import _read_url

from static_frame import Index

# TODO test
from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import _iterable_to_array
from static_frame.core.util import _array_to_groups_and_locations
from static_frame.core.util import IndexCorrespondence

from static_frame.core.util import _slice_to_ascending_slice
from static_frame.core.util import array_shift


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

        self.assertEqual(_resolve_dtype_iter((a1.dtype, a1.dtype)), a1.dtype)
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype)), a2.dtype)

        # boolean with mixed types
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype, a3.dtype)), np.object_)
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype, a5.dtype)), np.object_)
        self.assertEqual(_resolve_dtype_iter((a2.dtype, a2.dtype, a6.dtype)), np.object_)

        # numerical types go to float64
        self.assertEqual(_resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype)), np.float64)

        # add in bool or str, goes to object
        self.assertEqual(_resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a2.dtype)), np.object_)
        self.assertEqual(_resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a5.dtype)), np.object_)

        # mixed strings go to the largest
        self.assertEqual(_resolve_dtype_iter((a3.dtype, a5.dtype)), np.dtype('<U10'))



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


    def test_array_set_ufunc_many_a(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2, 1])
        a3 = np.array([3, 2, 1])
        a4 = np.array([3, 2, 1])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.intersect1d)
        self.assertEqual(post.tolist(), [3, 2, 1])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.union1d)
        self.assertEqual(post.tolist(), [3, 2, 1])

    def test_array_set_ufunc_many_b(self):
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2])
        a3 = np.array([5, 3, 2, 1])
        a4 = np.array([2])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.intersect1d)
        self.assertEqual(post.tolist(), [2])

        post = _array_set_ufunc_many((a1, a2, a3, a4), ufunc=np.union1d)
        self.assertEqual(post.tolist(), [1, 2, 3, 5])



    def test_intersect2d_a(self):
        a = np.array([('a', 'b'), ('c', 'd'), ('e', 'f')])
        b = np.array([('a', 'g'), ('c', 'd'), ('e', 'f')])

        post = _intersect2d(a, b)
        self.assertEqual(post.tolist(),
                [['c', 'd'], ['e', 'f']])

        post = _intersect2d(a.astype(object), b.astype(object))
        self.assertEqual(post.tolist(),
                [['c', 'd'], ['e', 'f']])

        post = _union2d(a, b)
        self.assertEqual(post.tolist(),
                [['a', 'b'], ['a', 'g'], ['c', 'd'], ['e', 'f']]
                )
        post = _union2d(a.astype(object), b.astype(object))
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


if __name__ == '__main__':
    unittest.main()

