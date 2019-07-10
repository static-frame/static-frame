

import unittest
import datetime
import typing as tp

import numpy as np  # type: ignore

from static_frame.core.util import isna_array
from static_frame.core.util import resolve_dtype
from static_frame.core.util import resolve_dtype_iter
from static_frame.core.util import array_to_duplicated
from static_frame.core.util import array_set_ufunc_many

from static_frame.core.util import intersect2d
from static_frame.core.util import union2d
from static_frame.core.util import concat_resolved


from static_frame.core.util import _gen_skip_middle
from static_frame.core.util import dtype_to_na
from static_frame.core.util import key_to_datetime_key

from static_frame.core.operator_delegate import _ufunc_logical_skipna

from static_frame.core.util import _read_url
from static_frame.core.util import set_ufunc2d

from static_frame import Index

# from static_frame.core.util import _dict_to_sorted_items
from static_frame.core.util import iterable_to_array
# from static_frame.core.util import collection_and_dtype_to_array


from static_frame.core.util import array_to_groups_and_locations
from static_frame.core.util import IndexCorrespondence

from static_frame.core.util import slice_to_ascending_slice
from static_frame.core.util import array_shift
from static_frame.core.util import ufunc_unique
from static_frame.core.util import to_timedelta64
from static_frame.core.util import binary_transition

from static_frame.core.util import roll_1d
from static_frame.core.util import roll_2d

from static_frame.core.util import union1d
from static_frame.core.util import intersect1d

from static_frame.core.util import to_datetime64

from static_frame.core.util import resolve_type
from static_frame.core.util import resolve_type_iter


from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    def test_gen_skip_middle_a(self) -> None:

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




    def test_ufunc_logical_skipna_a(self) -> None:

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


    def test_ufunc_logical_skipna_b(self) -> None:
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


    def test_ufunc_logical_skipna_c(self) -> None:

        a1 = np.array([], dtype=float)
        with self.assertRaises(NotImplementedError):
            _ufunc_logical_skipna(a1, np.sum, skipna=True)


    def test_ufunc_logical_skipna_d(self) -> None:

        a1 = np.array(['2018-01-01', '2018-02-01'], dtype=np.datetime64)
        post = _ufunc_logical_skipna(a1, np.all, skipna=True)
        self.assertTrue(post)


    def test_ufunc_logical_skipna_e(self) -> None:

        a1 = np.array([['2018-01-01', '2018-02-01'],
                ['2018-01-01', '2018-02-01']], dtype=np.datetime64)
        post = _ufunc_logical_skipna(a1, np.all, skipna=True)
        self.assertEqual(post.tolist(), [True, True])




    def test_resolve_dtype_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(resolve_dtype(a1.dtype, a1.dtype), a1.dtype)

        self.assertEqual(resolve_dtype(a1.dtype, a2.dtype), np.object_)
        self.assertEqual(resolve_dtype(a2.dtype, a3.dtype), np.object_)
        self.assertEqual(resolve_dtype(a2.dtype, a4.dtype), np.object_)
        self.assertEqual(resolve_dtype(a3.dtype, a4.dtype), np.object_)
        self.assertEqual(resolve_dtype(a3.dtype, a6.dtype), np.object_)

        self.assertEqual(resolve_dtype(a1.dtype, a4.dtype), np.float64)
        self.assertEqual(resolve_dtype(a1.dtype, a6.dtype), np.float64)
        self.assertEqual(resolve_dtype(a4.dtype, a6.dtype), np.float64)

    def test_resolve_dtype_b(self) -> None:

        self.assertEqual(
                resolve_dtype(np.array('a').dtype, np.array('aaa').dtype),
                np.dtype(('U', 3))
                )



    def test_resolve_dtype_c(self) -> None:


        a1 = np.array(['2019-01', '2019-02'], dtype=np.datetime64)
        a2 = np.array(['2019-01-01', '2019-02-01'], dtype=np.datetime64)
        a3 = np.array([0, 1], dtype='datetime64[ns]')
        a4 = np.array([0, 1])

        self.assertEqual(str(resolve_dtype(a1.dtype, a2.dtype)),
                'datetime64[D]')
        self.assertEqual(resolve_dtype(a1.dtype, a3.dtype),
                np.dtype('<M8[ns]'))

        self.assertEqual(resolve_dtype(a1.dtype, a4.dtype),
                np.dtype('O'))





    def test_resolve_dtype_iter_a(self) -> None:

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



    def test_isna_array_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3, 5.4], dtype='float32')

        self.assertEqual(isna_array(a1).tolist(), [False, False, False])
        self.assertEqual(isna_array(a2).tolist(), [False, False, False])
        self.assertEqual(isna_array(a3).tolist(), [False, False, False])
        self.assertEqual(isna_array(a4).tolist(), [False, False])
        self.assertEqual(isna_array(a5).tolist(), [False, False])
        self.assertEqual(isna_array(a6).tolist(), [False, False])

        a1 = np.array([1, 2, 3, None])
        a2 = np.array([False, True, False, None])
        a3 = np.array(['b', 'c', 'd', None])
        a4 = np.array([2.3, 3.2, None])
        a5 = np.array(['test', 'test again', None])
        a6 = np.array([2.3, 5.4, None])

        self.assertEqual(isna_array(a1).tolist(), [False, False, False, True])
        self.assertEqual(isna_array(a2).tolist(), [False, False, False, True])
        self.assertEqual(isna_array(a3).tolist(), [False, False, False, True])
        self.assertEqual(isna_array(a4).tolist(), [False, False, True])
        self.assertEqual(isna_array(a5).tolist(), [False, False, True])
        self.assertEqual(isna_array(a6).tolist(), [False, False, True])

        a1 = np.array([1, 2, 3, np.nan])
        a2 = np.array([False, True, False, np.nan])
        a3 = np.array(['b', 'c', 'd', np.nan], dtype=object)
        a4 = np.array([2.3, 3.2, np.nan], dtype=object)
        a5 = np.array(['test', 'test again', np.nan], dtype=object)
        a6 = np.array([2.3, 5.4, np.nan], dtype='float32')

        self.assertEqual(isna_array(a1).tolist(), [False, False, False, True])
        self.assertEqual(isna_array(a2).tolist(), [False, False, False, True])
        self.assertEqual(isna_array(a3).tolist(), [False, False, False, True])
        self.assertEqual(isna_array(a4).tolist(), [False, False, True])
        self.assertEqual(isna_array(a5).tolist(), [False, False, True])
        self.assertEqual(isna_array(a6).tolist(), [False, False, True])


    def test_isna_array_b(self) -> None:

        a1 = np.array([[1, 2], [3, 4]])
        a2 = np.array([[False, True, False], [False, True, False]])
        a3 = np.array([['b', 'c', 'd'], ['b', 'c', 'd']])
        a4 = np.array([[2.3, 3.2, np.nan], [2.3, 3.2, np.nan]])
        a5 = np.array([['test', 'test again', np.nan],
                ['test', 'test again', np.nan]], dtype=object)
        a6 = np.array([[2.3, 5.4, np.nan], [2.3, 5.4, np.nan]], dtype='float32')

        self.assertEqual(isna_array(a1).tolist(),
                [[False, False], [False, False]])

        self.assertEqual(isna_array(a2).tolist(),
                [[False, False, False], [False, False, False]])

        self.assertEqual(isna_array(a3).tolist(),
                [[False, False, False], [False, False, False]])

        self.assertEqual(isna_array(a4).tolist(),
                [[False, False, True], [False, False, True]])

        self.assertEqual(isna_array(a5).tolist(),
                [[False, False, True], [False, False, True]])

        self.assertEqual(isna_array(a6).tolist(),
                [[False, False, True], [False, False, True]])


    def test_array_to_duplicated_a(self) -> None:
        a = array_to_duplicated(
                np.array([0,1,2,2,1,4,5,3,4,5,5,6]),
                exclude_first=False,
                exclude_last=False
                )
        self.assertEqual(a.tolist(),
                [False, True, True, True, True, True, True, False, True, True, True, False])

        a = array_to_duplicated(
                np.array([0,1,2,2,1,4,5,3,4,5,5,6]),
                exclude_first=True,
                exclude_last=False
                )
        self.assertEqual(a.tolist(),
                [False, False, False, True, True, False, False, False, True, True, True, False])


    def test_array_to_duplicated_b(self) -> None:
        a = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        # find duplicate rows
        post = array_to_duplicated(a, axis=0)
        self.assertEqual(post.tolist(),
                [False, False])

        post = array_to_duplicated(a, axis=1)
        self.assertEqual(post.tolist(),
                [True, True, False, True, True])

        post = array_to_duplicated(a, axis=1, exclude_first=True)
        self.assertEqual(post.tolist(),
                [False, True, False, False, True])


    def test_array_to_duplicated_c(self) -> None:
        a = np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]])
        with self.assertRaises(NotImplementedError):
            # axis cannot be None
            array_to_duplicated(a, axis=None)  # type: ignore

    def test_array_to_duplicated_d(self) -> None:
        c = array_to_duplicated(
                np.array(['q','q','q', 'a', 'w', 'w'], dtype=object),
                exclude_first=False,
                exclude_last=False
                )
        self.assertEqual(c.tolist(), [True, True, True, False, True, True])

    # def test_array_to_duplicated_e(self) -> None:
        # NOTE: these cases fail with hetergenous types as we cannot sort
        # a = array_to_duplicated(
        #         np.array([0,0,1,0,None,None,0,1,None], dtype=object),
        #         exclude_first=False,
        #         exclude_last=False
        #         )

        # b = array_to_duplicated(
        #         np.array([0,0,1,0,'q','q',0,1,'q'], dtype=object),
        #         exclude_first=False,
        #         exclude_last=False
        #         )



    def test_array_set_ufunc_many_a(self) -> None:
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2, 1])
        a3 = np.array([3, 2, 1])
        a4 = np.array([3, 2, 1])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=False)
        self.assertEqual(post.tolist(), [3, 2, 1])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=True)
        self.assertEqual(post.tolist(), [3, 2, 1])


    def test_array_set_ufunc_many_b(self) -> None:
        a1 = np.array([3, 2, 1])
        a2 = np.array([3, 2])
        a3 = np.array([5, 3, 2, 1])
        a4 = np.array([2])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=False)
        self.assertEqual(post.tolist(), [2])

        post = array_set_ufunc_many((a1, a2, a3, a4), union=True)
        self.assertEqual(post.tolist(), [1, 2, 3, 5])


    def test_array_set_ufunc_many_c(self) -> None:
        a1 = np.array([[3, 2, 1], [1, 2, 3]])
        a2 = np.array([[5, 2, 1], [1, 2, 3]])
        a3 = np.array([[10, 20, 30], [1, 2, 3]])

        post = array_set_ufunc_many((a1, a2, a3), union=False)
        self.assertEqual(post.tolist(), [[1, 2, 3]])

        post = array_set_ufunc_many((a1, a2, a3), union=True)
        self.assertEqual(post.tolist(),
                [[1, 2, 3], [3, 2, 1], [5, 2, 1], [10, 20, 30]])


    def test_array_set_ufunc_many_d(self) -> None:
        a1 = np.array([3, 2, 1])
        a2 = np.array([[5, 2, 1], [1, 2, 3]])

        with self.assertRaises(Exception):
            post = array_set_ufunc_many((a1, a2), union=False)


    def test_array_set_ufunc_many_e(self) -> None:
        a1 = np.array([3, 2, 1])
        a2 = np.array([30, 20])

        post = array_set_ufunc_many((a1, a2), union=False)
        self.assertEqual(post.tolist(), [])


    def test_union1d_a(self) -> None:
        a1 = np.array([3, 2, 1])
        a2 = np.array(['3', '2', '1'])

        # need to avoid this
        # ipdb> np.union1d(a1, a2)                                                             # array(['1', '2', '3'], dtype='<U21')
        self.assertEqual(set(union1d(a1, a2)),
                {1, 2, 3, '2', '1', '3'}
                )

        self.assertEqual(
                union1d(np.array(['a', 'b', 'c']), np.array(['aaa', 'bbb', 'ccc'])).tolist(),
                ['a', 'aaa', 'b', 'bbb', 'c', 'ccc']
                )

        self.assertEqual(
                set(union1d(np.array([1, 2, 3]), np.array([None, False]))),
                {False, 2, 3, None, 1}
                )

        self.assertEqual(
                set(union1d(np.array([False, True]), np.array([None, 'a']))),
                {False, True, None, 'a'}
                )

        self.assertEqual(set(union1d(np.array([None, 1, 'd']), np.array([None, 3, 'ff']))),
                {'d', 1, 3, None, 'ff'}
                )

    def test_intersect1d_a(self) -> None:

        a1 = np.array([3, 2, 1])
        a2 = np.array(['3', '2', '1'])

        self.assertEqual(len(intersect1d(a1, a2)), 0)

        self.assertEqual(
                len(intersect1d(np.array([1, 2, 3]), np.array([None, False]))), 0)

        self.assertEqual(
                set(intersect1d(np.array(['a', 'b', 'c']), np.array(['aa', 'bbb', 'c']))),
                {'c'}
                )

    def test_intersect1d_b(self) -> None:
        # long way of
        a1 = np.empty(4, dtype=object)
        a1[:] = [(0, 0), (0, 1), (0, 2), (0, 3)]

        a2 = np.empty(3, dtype=object)
        a2[:] = [(0, 1), (0, 3), (4, 5)]

        # must get an array of tuples back
        post = intersect1d(a1, a2)
        self.assertEqual(post.tolist(),
                [(0, 1), (0, 3)])

    def test_intersect2d_a(self) -> None:
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
    def test_read_url(self) -> None:
        url = 'https://jsonplaceholder.typicode.com/todos'
        post = _read_url(url)


    def test_slice_to_ascending_slice_a(self) -> None:

        a1 = np.arange(10)

        def compare(slc: slice) -> None:
            slc_asc = slice_to_ascending_slice(slc, len(a1))
            self.assertEqual(sorted(a1[slc]), list(a1[slc_asc]))

        compare(slice(4,))
        compare(slice(6, 1, -1))
        compare(slice(6, 1, -2))
        compare(slice(6, None, -3))
        compare(slice(6, 2, -2))
        compare(slice(None, 1, -1))


    def test_array_shift_a(self) -> None:
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


    def test_array_shift_b(self) -> None:
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


    def test_array_shift_c(self) -> None:
        a1 = np.arange(6)
        post = array_shift(a1, 0, axis=0, wrap=False)
        self.assertEqual(a1.tolist(), post.tolist())


    def test_ufunc_unique_a(self) -> None:

        a1 = np.array([1, 1, 1, 2, 2])
        post = ufunc_unique(a1)
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1, 2])

        a2 = np.array([1, 1, 1, 2, 2], dtype=object)
        post = ufunc_unique(a2)
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1, 2])

        a3 = np.array([1, 'x', 1, None, 2], dtype=object)
        post = ufunc_unique(a3)
        self.assertEqual(post, {None, 1, 2, 'x'})


    def test_ufunc_unique_b(self) -> None:

        a1 = np.array([[1, 1], [1, 2], [1, 2]])
        post = ufunc_unique(a1)
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [1, 2])

        post = ufunc_unique(a1, axis=0)
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [[1, 1], [1, 2]])

        post = ufunc_unique(a1, axis=1)
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), [[1, 1], [1, 2], [1, 2]])


    def test_ufunc_unique_c(self) -> None:

        a1 = np.array([[1, 'x', 1], [1, None, 1], [1, 'x', 1]], dtype=object)

        post = ufunc_unique(a1)
        self.assertEqual(post, {'x', 1, None})

        post = ufunc_unique(a1, axis=0)
        self.assertEqual(post, {(1, 'x', 1), (1, None, 1)})

        post = ufunc_unique(a1, axis=1)
        self.assertEqual(post, {(1, 1, 1), ('x', None, 'x')})


    def test_concat_resolved_a(self) -> None:
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


    def test_concat_resolved_b(self) -> None:
        a1 = np.array([[3,4,5],[0,0,0]])
        a2 = np.array([1,2,3]).reshape((1,3))

        with self.assertRaises(Exception):
            concat_resolved((a1, a2), axis=None)


    def test_dtype_to_na_a(self) -> None:

        self.assertEqual(dtype_to_na(np.dtype(int)), 0)
        self.assertTrue(np.isnan(dtype_to_na(np.dtype(float))))
        self.assertEqual(dtype_to_na(np.dtype(bool)), False)
        self.assertEqual(dtype_to_na(np.dtype(object)), None)
        self.assertEqual(dtype_to_na(np.dtype(str)), '')


    def test_key_to_datetime_key_a(self) -> None:

        post = key_to_datetime_key(slice('2018-01-01', '2019-01-01'))  # type: ignore  # https://github.com/python/typeshed/pull/3024
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
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), a2.tolist())

        post = key_to_datetime_key(['2018-01-01', '2019-01-01'])
        a3 = np.array(['2018-01-01', '2019-01-01'], dtype='datetime64[D]')
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), a3.tolist())

        post = key_to_datetime_key(['2018-01', '2019-01'])
        a4 = np.array(['2018-01', '2019-01'], dtype='datetime64[M]')
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), a4.tolist())


        post = key_to_datetime_key(str(x) for x in range(2012, 2015))
        a5 = np.array(['2012', '2013', '2014'], dtype='datetime64[Y]')
        assert isinstance(post, np.ndarray)
        self.assertEqual(post.tolist(), a5.tolist())

        post = key_to_datetime_key(None)
        self.assertEqual(post, None)


    def test_ufunc2d_a(self) -> None:
        # fails due to wrong dimensionality, not wrong function
        a1 = np.array([1, 1, 1])
        with self.assertRaises(IndexError):
            set_ufunc2d(np.sum, a1, a1)


    def test_ufunc2d_b(self) -> None:

        a1 = np.array([['a', 'b'], ['b', 'c']])
        a2 = np.array([['b', 'cc'], ['dd', 'ee']])

        post = set_ufunc2d(np.union1d, a1, a2)
        self.assertEqual(len(post), 4)
        self.assertEqual(str(post.dtype), '<U2')

        post = set_ufunc2d(np.union1d, a2, a1)
        self.assertEqual(len(post), 4)
        self.assertEqual(str(post.dtype), '<U2')

    def test_to_timedelta64_a(self) -> None:
        timedelta = datetime.timedelta

        self.assertEqual(
                to_timedelta64(timedelta(days=4)),
                np.timedelta64(4, 'D'))

        self.assertEqual(
                to_timedelta64(timedelta(seconds=4)),
                np.timedelta64(4, 's'))

        self.assertEqual(
                to_timedelta64(timedelta(minutes=4)),
                np.timedelta64(240, 's'))

    def test_transition_indices_a(self) -> None:
        a1 = np.array([False, True, True, False, False, True, True, False])
        self.assertEqual(binary_transition(a1).tolist(), [0, 3, 4, 7])

        a1 = np.array([False, False, True, False, True, True, True, True])
        self.assertEqual(binary_transition(a1).tolist(), [1, 3])


        a1 = np.array([True, False, True])
        self.assertEqual(binary_transition(a1).tolist(), [1])

        a1 = np.array([False, True, False])
        self.assertEqual(binary_transition(a1).tolist(), [0, 2])

        a1 = np.array([True])
        self.assertEqual(binary_transition(a1).tolist(), [])

        a1 = np.array([False])
        self.assertEqual(binary_transition(a1).tolist(), [])

        a1 = np.array([False, True])
        self.assertEqual(binary_transition(a1).tolist(), [0])

        a1 = np.array([True, False])
        self.assertEqual(binary_transition(a1).tolist(), [1])


    def test_roll_1d_a(self) -> None:

        a1 = np.arange(12)

        for i in range(len(a1) + 1):
            post = roll_1d(a1, i)
            self.assertEqual(post.tolist(), np.roll(a1, i).tolist())

            post = roll_1d(a1, -i)
            self.assertEqual(post.tolist(), np.roll(a1, -i).tolist())

    def test_roll_1d_b(self) -> None:
        post = roll_1d(np.array([]), -4)
        self.assertEqual(len(post), 0)

    def test_roll_2d_a(self) -> None:

        a1 = np.arange(12).reshape((3,4))

        for i in range(a1.shape[0] + 1):
            post = roll_2d(a1, i, axis=0)
            self.assertEqual(post.tolist(), np.roll(a1, i, axis=0).tolist())

            post = roll_2d(a1, -i, axis=0)
            self.assertEqual(post.tolist(), np.roll(a1, -i, axis=0).tolist())

        for i in range(a1.shape[1] + 1):
            post = roll_2d(a1, i, axis=1)
            self.assertEqual(post.tolist(), np.roll(a1, i, axis=1).tolist())

            post = roll_2d(a1, -i, axis=1)
            self.assertEqual(post.tolist(), np.roll(a1, -i, axis=1).tolist())

    def test_roll_2d_b(self) -> None:
        post = roll_2d(np.array([[]]), -4, axis=1)
        self.assertEqual(post.shape, (1, 0))



    def test_index_correspondence_a(self) -> None:
        idx0 = Index([0, 1, 2, 3, 4], loc_is_iloc=True)
        idx1 = Index([0, 1, 2, 3, 4, '100185', '100828', '101376', '100312', '101092'], dtype=object)
        ic = IndexCorrespondence.from_correspondence(idx0, idx1)
        self.assertFalse(ic.is_subset)
        self.assertTrue(ic.has_common)
        # this is an array, due to loc_is_iloc being True
        assert isinstance(ic.iloc_src, np.ndarray)
        self.assertEqual(ic.iloc_src.tolist(),
                [0, 1, 2, 3, 4]
                )
        self.assertEqual(ic.iloc_dst,
                [0, 1, 2, 3, 4]
                )

    def test_to_datetime64_a(self) -> None:

        dt = to_datetime64('2019')
        self.assertEqual(dt, np.datetime64('2019'))

        dt = to_datetime64('2019', dtype=np.dtype('datetime64[D]'))
        self.assertEqual(dt, np.datetime64('2019-01-01'))

        dt = to_datetime64(np.datetime64('2019'), dtype=np.dtype('datetime64[Y]'))
        self.assertEqual(dt, np.datetime64('2019'))

        with self.assertRaises(RuntimeError):
            dt = to_datetime64(np.datetime64('2019'), dtype=np.dtype('datetime64[D]'))


    def test_resolve_type_a(self) -> None:

        self.assertEqual(resolve_type('a', str), (str, False))
        self.assertEqual(resolve_type('a', int), (object, False))
        self.assertEqual(resolve_type(3, str), (object, False))
        self.assertEqual(resolve_type((3,4), str), (object, True))
        self.assertEqual(resolve_type((3,4), tuple), (object, True))


        self.assertEqual(resolve_type(3, float), (float, False))
        self.assertEqual(resolve_type(False, str), (object, False))
        self.assertEqual(resolve_type(1.2, int), (float, False))


    def test_resolve_type_iter_a(self) -> None:

        v1 = ('a', 'b', 'c')
        resolved, has_tuple, values = resolve_type_iter(v1)
        self.assertEqual(resolved, str)

        v22 = ('a', 'b', 3)
        resolved, has_tuple, values = resolve_type_iter(v22)
        self.assertEqual(resolved, object)

        v3 = ('a', 'b', (1, 2))
        resolved, has_tuple, values = resolve_type_iter(v3)
        self.assertEqual(resolved, object)
        self.assertTrue(has_tuple)

        v4 = (1, 2, 4.3, 2)
        resolved, has_tuple, values = resolve_type_iter(v4)
        self.assertEqual(resolved, float)


        v5 = (1, 2, 4.3, 2, None)
        resolved, has_tuple, values = resolve_type_iter(v5)
        self.assertEqual(resolved, object)


        v6 = (1, 2, 4.3, 2, 'g')
        resolved, has_tuple, values = resolve_type_iter(v6)
        self.assertEqual(resolved, object)

        v7 = ()
        resolved, has_tuple, values = resolve_type_iter(v7)
        self.assertEqual(resolved, None)


    def test_resolve_type_iter_b(self) -> None:

        v1 = iter(('a', 'b', 'c'))
        resolved, has_tuple, values = resolve_type_iter(v1)
        self.assertEqual(resolved, str)

        v2 = iter(('a', 'b', 3))
        resolved, has_tuple, values = resolve_type_iter(v2)
        self.assertEqual(resolved, object)

        v3 = iter(('a', 'b', (1, 2)))
        resolved, has_tuple, values = resolve_type_iter(v3)
        self.assertEqual(resolved, object)
        self.assertTrue(has_tuple)

        v4 = range(4)
        resolved, has_tuple, values = resolve_type_iter(v4)
        self.assertEqual(resolved, int)


    def test_resolve_type_iter_c(self) -> None:

        a = [True, False, True]
        resolved, has_tuple, values = resolve_type_iter(a)
        self.assertEqual(id(a), id(values))

        resolved, has_tuple, values = resolve_type_iter(iter(a))
        self.assertNotEqual(id(a), id(values))

        self.assertEqual(resolved, bool)
        self.assertEqual(has_tuple, False)


    def test_resolve_type_iter_d(self) -> None:

        a = [3, 2, (3,4)]
        resolved, has_tuple, values = resolve_type_iter(a)
        self.assertEqual(id(a), id(values))

        resolved, has_tuple, values = resolve_type_iter(iter(a))
        self.assertNotEqual(id(a), id(values))

        self.assertEqual(resolved, object)
        self.assertEqual(has_tuple, True)


    def test_resolve_type_iter_e(self) -> None:

        a = [300000000000000002, 5000000000000000001]
        resolved, has_tuple, values = resolve_type_iter(a)
        self.assertEqual(id(a), id(values))

        resolved, has_tuple, values = resolve_type_iter(iter(a))
        self.assertNotEqual(id(a), id(values))

        self.assertEqual(resolved, int)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_f(self) -> None:

        def a() -> tp.Iterator[tp.Any]:
            for i in range(3):
                yield i
            yield None

        resolved, has_tuple, values = resolve_type_iter(a())
        self.assertEqual(values, [0, 1, 2, None])
        self.assertEqual(resolved, object)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_g(self) -> None:

        def a() -> tp.Iterator[tp.Any]:
            yield None
            for i in range(3):
                yield i

        resolved, has_tuple, values = resolve_type_iter(a())
        self.assertEqual(values, [None, 0, 1, 2])
        self.assertEqual(resolved, object)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_h(self) -> None:

        def a() -> tp.Iterator[tp.Any]:
            yield 10
            yield None
            for i in range(3):
                yield i
            yield (3,4)

        resolved, has_tuple, values = resolve_type_iter(a())
        self.assertEqual(values, [10, None, 0, 1, 2, (3,4)])
        self.assertEqual(resolved, object)
        # we stop evaluation after finding object
        self.assertEqual(has_tuple, True)

        post = iterable_to_array(a())
        self.assertEqual(post[0].tolist(),
                [10, None, 0, 1, 2, (3, 4)]
                )


    def test_iterable_to_array_a(self) -> None:
        a1, is_unique = iterable_to_array({3,4,5})
        self.assertTrue(is_unique)
        self.assertEqual(set(a1.tolist()), {3,4,5})

        a2, is_unique = iterable_to_array({None: 3, 'f': 4, 39: 0})
        self.assertTrue(is_unique)
        self.assertEqual(set(a2.tolist()), {None, 'f', 39})

        a3, is_unique = iterable_to_array((x*10 for x in range(1,4)))
        self.assertFalse(is_unique)
        self.assertEqual(a3.tolist(), [10, 20, 30])

        a1, is_unique = iterable_to_array({3,4,5}, dtype=np.dtype(int))
        self.assertEqual(set(a1.tolist()), {3,4,5})

        a1, is_unique = iterable_to_array((3,4,5), dtype=np.dtype(object))
        self.assertTrue(a1.dtype == object)
        self.assertEqual(a1.tolist(), [3,4,5])

        x = [(0, 0), (0, 1), (0, 2), (0, 3)]
        a1, is_unique = iterable_to_array(x, np.dtype(object))
        self.assertEqual(a1.tolist(), [(0, 0), (0, 1), (0, 2), (0, 3)])
        # must get an array of tuples back

        x = [(0, 0), (0, 1), (0, 2), (0, 3)]
        a1, is_unique = iterable_to_array(iter(x))
        self.assertEqual(a1.tolist(), [(0, 0), (0, 1), (0, 2), (0, 3)])


        self.assertEqual(iterable_to_array((1, 1.1))[0].dtype,
                np.dtype('float64'))

        self.assertEqual(iterable_to_array((1.1, 0, -29))[0].dtype,
                np.dtype('float64'))


    def test_iterable_to_array_b(self) -> None:

        iterable: tp.Iterable[tp.Any]

        for iterable in (  # type: ignore
                [1, 2, 3],
                dict(a=1, b=2, c=3).values(),
                dict(a=1, b=2, c=3).keys(),
                {1, 2, 3},
                frozenset((1, 2, 3)),
                ('a', 3, None),
                (1, 2, 'e', 1.1)
                ):

            a1, _ = iterable_to_array(iterable)
            self.assertEqual(set(a1), set(iterable))

            a2, _ = iterable_to_array(iter(iterable))
            self.assertEqual(set(a2), set(iterable))


    def test_iterable_to_array_c(self) -> None:

        iterable: tp.Iterable[tp.Any]

        for iterable, dtype in (  # type: ignore
                ([1, 2, 3], int),
                (dict(a=1, b=2, c=3).values(), int),
                (dict(a=1, b=2, c=3).keys(), str),
                ({1, 2, 3}, int),
                (frozenset((1, 2, 3)), int),
                (('a', 3, None), object),
                ((1, 2, 'e', 1.1), object),
                ):
            a1, _ = iterable_to_array(iterable, dtype=dtype)
            self.assertEqual(set(a1), set(iterable))

            a2, _ = iterable_to_array(iter(iterable), dtype=dtype)
            self.assertEqual(set(a2), set(iterable))



    def test_iterable_to_array_d(self) -> None:

        self.assertEqual(
                iterable_to_array((True, False, True))[0].dtype,
                np.dtype('bool')
        )

        self.assertEqual(
                iterable_to_array((0, 1, 0), dtype=bool)[0].dtype,
                np.dtype('bool')
        )

        self.assertEqual(
                iterable_to_array((1, 2, 'w'))[0].dtype,
                np.dtype('O')
        )

        self.assertEqual(iterable_to_array(((2,3), (3,2)))[0].tolist(),
                [(2, 3), (3, 2)]
        )


if __name__ == '__main__':
    unittest.main()

