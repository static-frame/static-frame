

import unittest

import numpy as np

from static_frame.core.util import _isna
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _resolve_dtype_iter
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import _array_set_ufunc_many


from static_frame.core.util import _gen_skip_middle
from static_frame.core.operator_delegate import _ufunc_logical_skipna

# TODO test
from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import _iterable_to_array
from static_frame.core.util import _array_to_groups_and_locations
from static_frame.core.util import IndexCorrespondence


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







if __name__ == '__main__':
    unittest.main()

