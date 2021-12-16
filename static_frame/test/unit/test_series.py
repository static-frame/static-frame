
import unittest
from collections import OrderedDict
from io import StringIO
import string
import pickle
import datetime
import typing as tp
from enum import Enum
import copy
import re


import numpy as np

from static_frame.test.test_case import TestCase
from static_frame.test.test_case import temp_file

import static_frame as sf
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
# from static_frame import SeriesHE
from static_frame import Frame
from static_frame import FrameGO
from static_frame import mloc
from static_frame import DisplayConfig
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO
from static_frame import IndexDate
from static_frame import IndexSecond
from static_frame import IndexYearMonth
from static_frame import IndexAutoFactory
from static_frame import IndexDefaultFactory
from static_frame.core.util import DTYPE_INT_DEFAULT
from static_frame.core.util import isna_array

from static_frame import HLoc
from static_frame import ILoc

from static_frame.core.exception import AxisInvalid
from static_frame.core.exception import ErrorInitSeries

nan = np.nan

LONG_SAMPLE_STR = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'


class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    # test series

    def test_series_slotted_a(self) -> None:
        s1 = Series.from_element(10, index=('a', 'b', 'c', 'd'))

        with self.assertRaises(AttributeError):
            s1.g = 30 #type: ignore #pylint: disable=E0237
        with self.assertRaises(AttributeError):
            s1.__dict__ #pylint: disable=W0104

    def test_series_init_a(self) -> None:
        s1 = Series.from_element(np.nan, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s1.dtype == float)
        self.assertTrue(len(s1) == 4)

        s2 = Series.from_element(False, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s2.dtype == bool)
        self.assertTrue(len(s2) == 4)

        s3 = Series.from_element(None, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s3.dtype == object)
        self.assertTrue(len(s3) == 4)


    def test_series_init_b(self) -> None:
        s1 = Series(['a', 'b', 'c', 'd'], index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')))

        # testing direct specification of string type
        s2 = Series(['a', 'b', 'c', 'd'], index=('a', 'b', 'c', 'd'), dtype=str)
        self.assertEqual(s2.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')))

    def test_series_init_c(self) -> None:

        s1 = Series.from_dict(OrderedDict([('b', 4), ('a', 1)]), dtype=np.int64)
        self.assertEqual(s1.to_pairs(),
                (('b', 4), ('a', 1)))

    def test_series_init_d(self) -> None:
        # single element, when the element is a string
        s1 = Series.from_element('abc', index=range(4))
        self.assertEqual(s1.to_pairs(),
                ((0, 'abc'), (1, 'abc'), (2, 'abc'), (3, 'abc')))

        # this is an array with shape == (), or a single element
        s2 = Series(np.array('abc'), index=range(4))
        self.assertEqual(s2.to_pairs(),
                ((0, 'abc'), (1, 'abc'), (2, 'abc'), (3, 'abc')))

        # single element, generator index
        s3 = Series.from_element(None, index=(x * 10 for x in (1,2,3)))
        self.assertEqual(s3.to_pairs(),
                ((10, None), (20, None), (30, None))
                )

    def test_series_init_e(self) -> None:
        s1 = Series.from_dict(dict(a=1, b=2, c=np.nan, d=None), dtype=object)
        self.assertEqual(s1.to_pairs(),
                (('a', 1), ('b', 2), ('c', nan), ('d', None))
                )
        with self.assertRaises(ValueError):
            s1.values[1] = 23

    def test_series_init_f(self) -> None:
        s1 = Series.from_dict({'a': 'x', 'b': 'y', 'c': 'z'})
        self.assertEqual(s1.to_pairs(), (('a', 'x'), ('b', 'y'), ('c', 'z')))

    def test_series_init_g(self) -> None:
        with self.assertRaises(RuntimeError):
            s1 = Series(range(4), own_index=True, index=None)

    def test_series_init_h(self) -> None:
        s1 = Series(range(4), index_constructor=IndexSecond)
        self.assertEqual(s1.to_pairs(),
            ((np.datetime64('1970-01-01T00:00:00'), 0),
            (np.datetime64('1970-01-01T00:00:01'), 1),
            (np.datetime64('1970-01-01T00:00:02'), 2),
            (np.datetime64('1970-01-01T00:00:03'), 3)))

    def test_series_init_i(self) -> None:
        s1 = Series((3, 4, 'a'))
        self.assertEqual(s1.values.tolist(),
                [3, 4, 'a']
                )

    def test_series_init_j(self) -> None:
        s1 = Series((3, 4, 'a'), index=IndexAutoFactory)
        self.assertEqual(s1.to_pairs(),
                ((0, 3), (1, 4), (2, 'a')))

    def test_series_init_k(self) -> None:
        s1 = Series.from_element('cat', index=(1, 2, 3))
        self.assertEqual(s1.to_pairs(),
                ((1, 'cat'), (2, 'cat'), (3, 'cat'))
                )

    def test_series_init_l(self) -> None:
        s1 = Series(([None], [1, 2], ['a', 'b']), index=(1, 2, 3))
        self.assertEqual(s1[2:].to_pairs(),
                ((2, [1, 2]), (3, ['a', 'b'])))
        self.assertEqual((s1 * 2).to_pairs(),
                ((1, [None, None]), (2, [1, 2, 1, 2]), (3, ['a', 'b', 'a', 'b']))
                )

    def test_series_init_m(self) -> None:

        # if index is None or IndexAutoFactory, we supply an index of 0
        s1 = Series.from_element('a', index=(0,))
        self.assertEqual(s1.to_pairs(),
                ((0, 'a'),))

        # an element with an explicitl empty index results in an empty series
        s2 = Series.from_element('a', index=())
        self.assertEqual(s2.to_pairs(), ())

    def test_series_init_n(self) -> None:
        with self.assertRaises(RuntimeError):
            s1 = Series(np.array([['a', 'b']]))

        s2 = Series([['a', 'b']], dtype=object)
        self.assertEqual(s2.to_pairs(),
            ((0, ['a', 'b']),)
            )

    def test_series_init_o(self) -> None:
        with self.assertRaises(ErrorInitSeries):
            s1 = Series('T', index=range(3))

        s1 = Series.from_element('T', index=())
        self.assertEqual(s1.to_pairs(), ())


    def test_series_init_p(self) -> None:
        # 3d array raises exception
        a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        with self.assertRaises(RuntimeError):
            s1 = Series(a1)


    def test_series_init_q(self) -> None:
        with self.assertRaises(RuntimeError):
            s1 = Series(dict(a=3, b=4))


    def test_series_init_r(self) -> None:
        with self.assertRaises(RuntimeError):
            s1 = Series(np.array((3, 4, 5)), dtype=object)


    def test_series_init_s(self) -> None:
        s1 = Series(np.array('a'))
        self.assertEqual(s1.to_pairs(), ((0, 'a'),))


    def test_series_init_t(self) -> None:
        s1 = Series(('a', 'b', 'c'), index=(10, 20, 30))
        s2 = Series(s1)
        s3 = Series(values=s1)

        # experimented with, but did not enable, for object aliasing when immutable
        self.assertTrue(id(s1) != id(s2))
        self.assertTrue(id(s1) != id(s3))

        # same array is used
        self.assertTrue(id(s1.values) == id(s2.values))
        self.assertTrue(id(s1.values) == id(s3.values))

        # same index is used
        self.assertTrue(id(s1.index) == id(s2.index))
        self.assertTrue(id(s1.index) == id(s3.index))

        # can swap in a different index
        s4 = Series(s1, index=('x', 'y', 'z'))
        self.assertEqual(s4.to_pairs(),
                (('x', 'a'), ('y', 'b'), ('z', 'c'))
                )
        self.assertTrue(id(s1.values) == id(s4.values))

        with self.assertRaises(ErrorInitSeries):
            Series(s1, dtype=float)



    def test_series_init_u(self) -> None:
        with self.assertRaises(ErrorInitSeries):
            s1 = Series(('a', 'b', 'c'), index=(10, 30))

        with self.assertRaises(ErrorInitSeries):
            s1 = Series(('a', 'b', 'c'), index=())

        with self.assertRaises(ErrorInitSeries):
            s1 = Series(('a', 'b', 'c'), index=(10, 20, 30, 40))

        with self.assertRaises(ErrorInitSeries):
            s1 = Series(range(3), index=range(2))

        with self.assertRaises(ErrorInitSeries):
            s1 = Series(range(3), index=Index(range(2)), own_index=True)


        s1 = Series(np.array(3), index=(10, 20, 30, 40))
        self.assertEqual(s1.to_pairs(),
                ((10, 3), (20, 3), (30, 3), (40, 3))
                )
        s2 = Series(np.array(3))
        self.assertEqual(s2.to_pairs(), ((0, 3),))


    def test_series_init_v(self) -> None:
        f1 = Frame(np.arange(4).reshape(2,2))
        f2 = Frame(np.arange(4).reshape(2,2))

        s = Series((f1, f2))
        self.assertEqual(len(s), 2)
        self.assertTrue(s[0].equals(f1))
        self.assertTrue(s[1].equals(f1))


    def test_series_init_w(self) -> None:
        s1 = Series.from_element(0, index=IndexAutoFactory(4))
        self.assertEqual(s1.shape, (4,))
        self.assertEqual(s1.to_pairs(),
                ((0, 0), (1, 0), (2, 0), (3, 0)))
        self.assertTrue(s1._index._map is None) #type: ignore

    #---------------------------------------------------------------------------
    def test_series_from_dict_a(self) -> None:

        s1 = Series.from_dict(OrderedDict([('b', 4), ('a', 1)]),
                index_constructor=IndexDefaultFactory('foo'), #type: ignore
                )
        self.assertEqual(s1.to_pairs(),
                (('b', 4), ('a', 1)))
        self.assertEqual(s1.index.name, 'foo')

    #---------------------------------------------------------------------------

    def test_series_slice_a(self) -> None:
        # create a series from a single value

        # generator based construction of values and index
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1['a':'c']   # type: ignore  # https://github.com/python/typeshed/pull/3024  # with Pandas this is inclusive
        self.assertEqual(s2.values.tolist(), [0, 1, 2])
        self.assertTrue(s2['b'] == s1['b'])

        s3 = s1['c':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(s3.values.tolist(), [2, 3])
        self.assertTrue(s3['d'] == s1['d'])

        self.assertEqual(s1['b':'d'].values.tolist(), [1, 2, 3])  # type: ignore  # https://github.com/python/typeshed/pull/3024

        self.assertEqual(s1[['a', 'c']].values.tolist(), [0, 2])

    def test_series_slice_b(self) -> None:

        # using step sizes mixed with locs
        s1 = sf.Series([1, 2, 3], index=['a', 'b', 'c'])['b'::-1] #type: ignore

        self.assertEqual(s1.to_pairs(),
                (('b', 2), ('a', 1)))

    def test_series_slice_c(self) -> None:

        # using step sizes mixed with locs
        s1 = sf.Series(range(10), index=IndexDate.from_date_range('2019-12-30', '2020-01-08'))

        s2 = s1.loc[np.datetime64('2020-01-07'): np.datetime64('2020-01-02'): -1]

        self.assertEqual(s2.to_pairs(),
                ((np.datetime64('2020-01-07'), 8), (np.datetime64('2020-01-06'), 7), (np.datetime64('2020-01-05'), 6), (np.datetime64('2020-01-04'), 5))
                )


    #---------------------------------------------------------------------------

    def test_series_keys_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1.keys()), ['a', 'b', 'c', 'd'])

    def test_series_iter_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1), ['a', 'b', 'c', 'd'])

    def test_series_items_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1.items()), [('a', 0), ('b', 1), ('c', 2), ('d', 3)])


    def test_series_intersection_a(self) -> None:
        # create a series from a single value
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s3 = s1['c':]  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(s1.index.intersection(s3.index).values.tolist(),
            ['c', 'd'])


    def test_series_intersection_b(self) -> None:
        # create a series from a single value
        idxa = IndexGO(('a', 'b', 'c'))
        idxb = IndexGO(('b', 'c', 'd'))

        self.assertEqual(idxa.intersection(idxb).values.tolist(),
            ['b', 'c'])

        self.assertEqual(idxa.union(idxb).values.tolist(),
            ['a', 'b', 'c', 'd'])

    #---------------------------------------------------------------------------


    def test_series_binary_operator_a(self) -> None:
        '''Test binary operators where one operand is a numeric.
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')

        self.assertEqual(list((s1 * 3).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])
        self.assertEqual((s1 * 3).name, 'foo')

        self.assertEqual(list((s1 / .5).items()),
                [('a', 0.0), ('b', 2.0), ('c', 4.0), ('d', 6.0)])

        self.assertEqual(list((s1 ** 3).items()),
                [('a', 0), ('b', 1), ('c', 8), ('d', 27)])
        self.assertEqual((s1 ** 3).name, 'foo')


    def test_series_binary_operator_b(self) -> None:
        '''Test binary operators with Series of same index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        s2 = Series((x * 2 for x in range(4)), index=('a', 'b', 'c', 'd'), name='bar')

        self.assertEqual(list((s1 + s2).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual((s1 + s2).name, None)

        self.assertEqual(list((s1 * s2).items()),
                [('a', 0), ('b', 2), ('c', 8), ('d', 18)])


    def test_series_binary_operator_c(self) -> None:
        '''Test binary operators with Series of different index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('c', 'd', 'e', 'f'))

        self.assertAlmostEqualItems(list((s1 * s2).items()),
                [('a', nan), ('b', nan), ('c', 0), ('d', 6), ('e', nan), ('f', nan)]
                )


    def test_series_binary_operator_d(self) -> None:
        s1 = Series(range(4), index=list('abcd'))
        s2 = Series(range(3), index=list('abc'))
        s3 = s1 + s2

        self.assertEqual(s3.fillna(None).to_pairs(),
                (('a', 0), ('b', 2), ('c', 4), ('d', None))
                )

        s1 = Series((False, True, False, True), index=list('abcd'))
        s2 = Series([True] * 3, index=list('abc'))

        # NOTE: for now, we cannot resolve this case, as after reindexing we get an object array that is not compatible with Boolean array for the NaN4
        with self.assertRaises(TypeError):
            s3 = s1 | s2


    def test_series_binary_operator_e(self) -> None:

        s1 = Series((False, True, False, True), index=list('abcd'), name='foo')
        s2 = Series([True] * 3, index=list('abc'))

        self.assertEqual((s1 == -1).to_pairs(),
                (('a', False), ('b', False), ('c', False), ('d', False)))

        self.assertEqual((s1 == s2).to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', False)))

        self.assertEqual((s1 == True).to_pairs(), #pylint: disable=C0121
                (('a', False), ('b', True), ('c', False), ('d', True)))
        self.assertEqual((s1 == True).name, 'foo') #pylint: disable=C0121

        self.assertEqual((s1 == (True,)).to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', True)))
        # as this is samed sized, NP does element wise comparison
        self.assertEqual((s1 == (False, True, False, True)).to_pairs(),
                (('a', True), ('b', True), ('c', True), ('d', True)))

        # NOTE: these are unexpected results that derive from NP Boolean operator behaviors
        with self.assertRaises(ValueError):
            _ = s1 == (True, False)
        with self.assertRaises(ValueError):
            _ = s1 == (False, True, False, True, False)

    def test_series_binary_operator_f(self) -> None:
        r = Series(['100312', '101376', '100828', '101214', '100185'])
        c = Series(['100312', '101376', '101092', '100828', '100185'],
                index=['100312', '101376', '101092', '100828', '100185'])
        post = r == c

        self.assertEqual(set(post.to_pairs()),
                set(((0, False), (1, False), (2, False), (3, False), (4, False), ('101376', False), ('101092', False), ('100828', False), ('100312', False), ('100185', False)))
                )


    def test_series_binary_operator_g(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(
                (s1 - 1).to_pairs(),
                (('a', -1), ('b', 0), ('c', 1), ('d', 2))
                )

        self.assertEqual((1 - s1).to_pairs(),
                (('a', 1), ('b', 0), ('c', -1), ('d', -2))
                )


    def test_series_binary_operator_h(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(
                s1 @ sf.Series([3, 4, 1, 2], index=('a', 'b', 'c', 'd')),
                12
                )
        self.assertEqual(
                s1 @ sf.Series([3, 4, 1, 2], index=('a', 'c', 'b', 'd')),
                15
                )

    def test_series_binary_operator_i(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        post = [3, 4, 1, 2] @ s1
        self.assertEqual(post, 12)



    def test_series_binary_operator_j(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        with self.assertRaises(NotImplementedError):
            _ = s1 + np.arange(4).reshape((2, 2))


    def test_series_binary_operator_k(self) -> None:

        s3 = sf.Series.from_element('b', index=range(3), name='foo')
        s4 = 3 * s3

        self.assertEqual(s4.name, 'foo')

        self.assertEqual(s4.to_pairs(),
                ((0, 'bbb'), (1, 'bbb'), (2, 'bbb')))

        self.assertEqual((s3 * 3).to_pairs(),
                ((0, 'bbb'), (1, 'bbb'), (2, 'bbb')))


        s5 = s3 + '_'
        self.assertEqual(s5.to_pairs(),
                ((0, 'b_'), (1, 'b_'), (2, 'b_'))
                )

        self.assertEqual(('_' + s3).to_pairs(),
                ((0, '_b'), (1, '_b'), (2, '_b'))
                )

    def test_series_binary_operator_l(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        s2 = s1 * np.arange(10, 14)
        self.assertEqual(s2.to_pairs(),
                (('a', 0), ('b', 11), ('c', 24), ('d', 39)))
        self.assertEqual(s2.name, None)


    def test_series_binary_operator_m(self) -> None:

        s = Series((np.datetime64('2000-01-01'), np.datetime64('2001-01-01')))
        d = np.datetime64('2000-12-31')

        self.assertEqual((s > d).to_pairs(),
                ((0, False), (1, True)))

        with self.assertRaises(TypeError):
            # TypeError: invalid type promotion
            _ = d < s # why does this fail?

        s2 = s.iloc[:1]

        self.assertEqual((s2 < d).to_pairs(),
                ((0, True),))

        with self.assertRaises(TypeError):
            # TypeError: int() argument must be a string, a bytes-like object or a number, not 'datetime.date'
            _ = d < s2


    def test_series_binary_operator_n(self) -> None:


        s1 = Series([0, 1])
        s2 = Series([1, 2, 3])
        # when comparing reindexable containers, we get a result
        s3 = s1 == s2
        self.assertEqual(s3.to_pairs(),
                ((0, False), (1, False), (2, False)))

        with self.assertRaises(ValueError):
            # an index is not a reindexable container, so this raises
            _ = s1 == s2.index


    def test_series_binary_operator_o(self) -> None:

        s1 = Series([10, 20, 30])
        s2 = s1 == ''
        self.assertEqual(s2.values.tolist(),
                [False, False, False])

        # NOTE: numpy compares each value
        s3 = s1 == (10, 20, 30)
        self.assertEqual(s3.values.tolist(),
                [True, True, True])

        # we treat this as a single tuple
        with self.assertRaises(ValueError):
            _ = s1 == (10, 20)

        with self.assertRaises(ValueError):
            _ = s1 == [10, 20]


    def test_series_binary_operator_p(self) -> None:

        s1 = Series([10, 20, 30]) << Series([1, 2, 1])
        self.assertEqual(s1.to_pairs(),
            ((0, 20), (1, 80), (2, 60)))

        s2 = Series([10, 20, 30]) >> Series([1, 2, 1])
        self.assertEqual(s2.to_pairs(),
            ((0, 5), (1, 5), (2, 15)))

        s3 = [10, 20, 30] / Series([1, 2, 1])
        self.assertEqual(s3.to_pairs(),
            ((0, 10), (1, 10), (2, 30)))

        s4 = [10, 20, 30] // Series([2, 3, 4])
        self.assertEqual(s4.to_pairs(),
            ((0, 5), (1, 6), (2, 7)))

    #---------------------------------------------------------------------------
    def test_series_rename_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(s1.name, 'foo')
        s2 = s1.rename(None)
        self.assertEqual(s2.name, None)

    def test_series_rename_b(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(s1.name, 'foo')
        s2 = Series(s1)
        self.assertEqual(s2.name, 'foo')

    def test_series_rename_c(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        s2 = s1.rename(None, index='bar')
        self.assertEqual(s2.name, None)
        self.assertEqual(s2.index.name, 'bar')

    def test_series_rename_d(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        s2 = s1.rename(index='a')
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.index.name, 'a')

    #---------------------------------------------------------------------------


    def test_series_reindex_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.reindex(('c', 'd', 'a'))
        self.assertEqual(list(s2.items()), [('c', 2), ('d', 3), ('a', 0)])

        s3 = s1.reindex(['a','b'])
        self.assertEqual(list(s3.items()), [('a', 0), ('b', 1)])


        # an int-valued array is hard to provide missing values for

        s4 = s1.reindex(['b', 'q', 'g', 'a'], fill_value=None)
        self.assertEqual(list(s4.items()),
                [('b', 1), ('q', None), ('g', None), ('a', 0)])

        # by default this gets float because filltype is nan by default
        s5 = s1.reindex(['b', 'q', 'g', 'a'])
        self.assertAlmostEqualItems(list(s5.items()),
                [('b', 1), ('q', nan), ('g', nan), ('a', 0)])


    def test_series_reindex_b(self) -> None:
        s1 = Series(range(4), index=IndexHierarchy.from_product(('a', 'b'), ('x', 'y')))
        s2 = Series(range(4), index=IndexHierarchy.from_product(('b', 'c'), ('x', 'y')))

        s3 = s1.reindex(s2.index, fill_value=None)

        self.assertEqual(s3.to_pairs(),
                ((('b', 'x'), 2), (('b', 'y'), 3), (('c', 'x'), None), (('c', 'y'), None)))

        # can reindex with a different dimensionality if no matches
        self.assertEqual(
                s1.reindex((3,4,5,6), fill_value=None).to_pairs(),
                ((3, None), (4, None), (5, None), (6, None)))

        self.assertEqual(
                s1.reindex((('b', 'x'),4,5,('a', 'y')), fill_value=None).to_pairs(),
                ((('b', 'x'), 2), (4, None), (5, None), (('a', 'y'), 1)))



    def test_series_reindex_c(self) -> None:
        s1 = Series(('a', 'b', 'c', 'd'), index=((0, x) for x in range(4)))
        self.assertEqual(s1.loc[(0, 2)], 'c')

        s1.reindex(((0, 1), (0, 3), (4,5)))

        self.assertEqual(
                s1.reindex(((0, 1), (0, 3), (4,5)), fill_value=None).to_pairs(),
                (((0, 1), 'b'), ((0, 3), 'd'), ((4, 5), None)))


        s2 = s1.reindex(('c', 'd', 'a'))
        self.assertEqual(sorted(s2.index.values.tolist()), ['a', 'c', 'd'])


    def test_series_reindex_d(self) -> None:

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        s2 = s1.reindex(('c', 'd', 'a'))
        self.assertEqual(s2.index.values.tolist(), ['c', 'd', 'a'])
        self.assertEqual(s2.name, 'foo')

    def test_series_reindex_e(self) -> None:

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        idx = Index(('c', 'd', 'a'))
        s2 = s1.reindex(idx, own_index=True)
        self.assertEqual(s2.index.values.tolist(), ['c', 'd', 'a'])
        self.assertEqual(s2.name, 'foo')
        # we owned the index, so have the same instance
        self.assertEqual(id(s2.index), id(idx))

    def test_series_reindex_f(self) -> None:

        index = IndexDate.from_date_range('2020-03-05', '2020-03-10')

        s1 = Series(range(6), index=index.values, index_constructor=IndexDate) # create an Index
        s2 = s1.reindex(index) # same values, different class
        self.assertTrue(s2.index.__class__, index.__class__)

    def test_series_reindex_g(self) -> None:

        s1 = sf.Series((3, 0, 1), index=(
                datetime.date(2020,12,31),
                datetime.date(2021,1,15),
                datetime.date(2021,1,31)))

        s2 = s1.reindex(IndexDate([np.datetime64(d) for d in s1.index[:2]]), fill_value=None) #type: ignore
        self.assertEqual(s2.to_pairs(),
                ((np.datetime64('2020-12-31'), 3),
                (np.datetime64('2021-01-15'), 0))
                )

        s3 = s1.reindex(IndexDate([np.datetime64(d) for d in s1.index]), fill_value=None)

        self.assertEqual(s3.to_pairs(),
                ((np.datetime64('2020-12-31'), 3),
                (np.datetime64('2021-01-15'), 0),
                (np.datetime64('2021-01-31'), 1)),
                )

    def test_series_reindex_h(self) -> None:

        dt = datetime.date
        dt64 = np.datetime64

        s1 = sf.Series((3, 0, 1), index=IndexDate((
                dt(2020,12,31),
                dt(2021,1,15),
                dt(2021,1,31),
                )))

        s2 = s1.reindex(IndexDate(reversed(s1.index))) #type: ignore
        self.assertEqual(s2.to_pairs(),
                ((dt(2021, 1, 31), 1),
                (dt(2021, 1, 15), 0),
                (dt(2020, 12, 31), 3)))

        s3 = s1.reindex(IndexDate([dt64(d) for d in reversed(s1.index)])) #type: ignore
        self.assertEqual(s3.to_pairs(),
                ((dt64('2021-01-31'), 1),
                (dt64('2021-01-15'), 0),
                (dt64('2020-12-31'), 3)))


    def test_series_reindex_i(self) -> None:

        dt = datetime.date
        dt64 = np.datetime64

        s1 = sf.Series((3, 0, 1), index=IndexDate((
                dt(2020,12,31),
                dt(2021,1,15),
                dt(2021,1,31),
                )))

        s2 = s1.reindex(IndexDate(reversed(s1.index))) #type: ignore
        self.assertEqual(s2.to_pairs(),
                ((dt(2021, 1, 31), 1),
                (dt(2021, 1, 15), 0),
                (dt(2020, 12, 31), 3)))

        s3 = s1.reindex(IndexDate([dt64(d) for d in reversed(s1.index)])) #type: ignore
        self.assertEqual(s3.to_pairs(),
                ((dt64('2021-01-31'), 1),
                (dt64('2021-01-15'), 0),
                (dt64('2020-12-31'), 3)))

    def test_series_reindex_j(self) -> None:

        ih1 = IndexHierarchy.from_labels(((1, '2020-01-01'), (1, '2020-01-02'), (1, '2020-01-03')),
                index_constructors=(Index, IndexDate))

        ih2 = IndexHierarchy.from_labels(((1, '2020-01-01'), (1, '2020-01-02'), (1, '2020-01-05')),
                index_constructors=(Index, IndexDate))

        s1 = Series((1, 2, 3), index=ih1)
        self.assertEqual(s1.reindex(ih2, fill_value=None).to_pairs(),
                (((1, datetime.date(2020, 1, 1)), 1), ((1, datetime.date(2020, 1, 2)), 2), ((1, datetime.date(2020, 1, 5)), None)))


    def test_series_reindex_k(self) -> None:
        dt = datetime.date

        s1 = sf.Frame.from_dict({'a': [1,1,1], 'b':[dt(2020, 1, 1), dt(2020, 1, 2), dt(2020, 1, 3)], 'd':['a', 'b', 'c']}, dtypes={'b': 'datetime64[D]'}).set_index_hierarchy(('a', 'b'), drop=True, index_constructors=(Index, IndexDate))['d']

        ih2 = IndexHierarchy.from_labels(((1, '2020-01-01'), (1, '2020-01-02'), (1, '2020-01-05')),
                index_constructors=(Index, IndexDate))

        self.assertEqual(s1.reindex(ih2, fill_value=None).to_pairs(),
                (((1, datetime.date(2020, 1, 1)), 'a'), ((1, datetime.date(2020, 1, 2)), 'b'), ((1, datetime.date(2020, 1, 5)), None)))


    #---------------------------------------------------------------------------
    def test_series_isna_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', True)]
                )
        self.assertEqual(list(s2.isna().items()),
                [('a', False), ('b', True), ('c', False), ('d', True)])

        self.assertEqual(list(s3.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])

        self.assertEqual(list(s4.isna().items()),
                [('a', False), ('b', True), ('c', True), ('d', True)])

        # those that are always false
        self.assertEqual(list(s5.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])

        self.assertEqual(list(s6.isna().items()),
                [('a', False), ('b', False), ('c', False), ('d', False)])



    def test_series_isna_b(self) -> None:

        # NOTE: this is a problematic case as it as a string with numerics and None
        s1 = Series((234.3, 'a', None, 6.4, np.nan), index=('a', 'b', 'c', 'd', 'e'))

        self.assertEqual(list(s1.isna().items()),
                [('a', False), ('b', False), ('c', True), ('d', False), ('e', True)]
                )

    def test_series_notnull(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', False)]
                )
        self.assertEqual(list(s2.notna().items()),
                [('a', True), ('b', False), ('c', True), ('d', False)])

        self.assertEqual(list(s3.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])

        self.assertEqual(list(s4.notna().items()),
                [('a', True), ('b', False), ('c', False), ('d', False)])

        # those that are always false
        self.assertEqual(list(s5.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])

        self.assertEqual(list(s6.notna().items()),
                [('a', True), ('b', True), ('c', True), ('d', True)])


    def test_series_dropna_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.dropna().to_pairs(),
                (('a', 234.3), ('b', 3.2), ('c', 6.4)))
        self.assertEqual(list(s2.dropna().items()),
                [('a', 234.3), ('c', 6.4)])
        self.assertEqual(s4.dropna().to_pairs(),
                (('a', 234.3),))
        self.assertEqual(s5.dropna().to_pairs(),
                (('a', 'p'), ('b', 'q'), ('c', 'e'), ('d', 'g')))
        self.assertEqual(s6.dropna().to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', True)))

    def test_series_dropna_b(self) -> None:
        s1 = sf.Series.from_element(np.nan, index=sf.IndexHierarchy.from_product(['A', 'B'], [1, 2]))
        s2 = s1.dropna()
        self.assertEqual(len(s2), 0)
        self.assertEqual(s1.__class__, s2.__class__)

    def test_series_dropna_c(self) -> None:
        s1 = sf.Series([1, np.nan, 2, np.nan],
                index=sf.IndexHierarchy.from_product(['A', 'B'], [1, 2]))
        s2 = s1.dropna()
        self.assertEqual(s2.to_pairs(), ((('A', 1), 1.0), (('B', 1), 2.0)))

    #---------------------------------------------------------------------------

    def test_series_fillna_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))
        s7 = Series((10, 20, 30, 40), index=('a', 'b', 'c', 'd'))
        s8 = Series((234.3, None, 6.4, np.nan, 'q'), index=('a', 'b', 'c', 'd', 'e'))


        self.assertEqual(s1.fillna(0.0).values.tolist(),
                [234.3, 3.2, 6.4, 0.0])

        self.assertEqual(s1.fillna(-1).values.tolist(),
                [234.3, 3.2, 6.4, -1.0])

        # given a float array, inserting None, None is casted to nan
        self.assertEqual(s1.fillna(None).values.tolist(),
                [234.3, 3.2, 6.4, None])

        post = s1.fillna('wer')
        self.assertEqual(post.dtype, object)
        self.assertEqual(post.values.tolist(),
                [234.3, 3.2, 6.4, 'wer'])


        post = s7.fillna(None)
        self.assertEqual(post.dtype, int)


    def test_series_fillna_b(self) -> None:

        s1 = Series(())
        s2 = s1.fillna(0)
        self.assertTrue(len(s2) == 0)


    def test_series_fillna_c(self) -> None:

        s1 = Series((np.nan, 3, np.nan))
        with self.assertRaises(RuntimeError):
            _ = s1.fillna(np.arange(3))


    def test_series_fillna_d(self) -> None:

        s1 = Series((np.nan, 3, np.nan, 4), index=tuple('abcd'))
        s2 = Series((100, 200), index=tuple('ca'))
        s3 = s1.fillna(s2)
        self.assertEqual(s3.dtype, float)
        self.assertEqual(s3.to_pairs(),
                (('a', 200.0), ('b', 3.0), ('c', 100.0), ('d', 4.0))
                )

    def test_series_fillna_e(self) -> None:

        s1 = Series((None, None, 'foo', 'bar'), index=tuple('abcd'))
        s2 = Series((100, 200), index=tuple('ca'))
        s3 = s1.fillna(s2)
        self.assertEqual(s3.dtype, object)
        self.assertEqual(type(s3['a']), int)
        self.assertEqual(s3.to_pairs(),
                (('a', 200), ('b', None), ('c', 'foo'), ('d', 'bar'))
                )


    def test_series_fillna_f(self) -> None:

        s1 = Series((None, None, 'foo', 'bar'), index=tuple('abcd'))
        s2 = Series((100, 200))
        s3 = s1.fillna(s2)
        # no alignment, return the same Series
        self.assertEqual(id(s3), id(s1))



    def test_series_fillna_g(self) -> None:

        s1 = Series((np.nan, 3, np.nan, 4), index=tuple('abcd'))
        s2 = Series((False, True), index=tuple('ba'))
        s3 = s1.fillna(s2)
        self.assertEqual(s3.dtype, object)
        self.assertEqual(s3.fillna(-1).to_pairs(),
                (('a', True), ('b', 3.0), ('c', -1), ('d', 4.0))
                )


    #---------------------------------------------------------------------------

    def test_series_fillna_directional_a(self) -> None:

        a1 = np.array((3, 4))
        a2 = Series._fill_missing_directional(
                array=a1,
                directional_forward=True,
                func_target=isna_array,
                limit=2)

        self.assertEqual(id(a1), id(a2))


    def test_series_fillna_sided_a(self) -> None:

        a1 = np.array((np.nan, 3, np.nan))

        with self.assertRaises(RuntimeError):
            _ = Series._fill_missing_sided(
                    array=a1,
                    value=a1,
                    func_target=isna_array,
                    sided_leading=True)



    #---------------------------------------------------------------------------

    def test_series_fillna_leading_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((np.nan, None, 6, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((np.nan, np.nan, np.nan, 4), index=('a', 'b', 'c', 'd'))
        s4 = Series((None, None, None, None), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.fillna_leading(-1).fillna(0).to_pairs(),
                (('a', 234.3), ('b', 3.2), ('c', 6.4), ('d', 0.0)))

        self.assertEqual(s2.fillna_leading(0).fillna(-1).to_pairs(),
                (('a', 0), ('b', 0), ('c', 6), ('d', -1)))

        self.assertEqual(s3.fillna_leading('a').to_pairs(),
                (('a', 'a'), ('b', 'a'), ('c', 'a'), ('d', 4.0)))

        self.assertEqual(s4.fillna_leading('b').to_pairs(),
                (('a', 'b'), ('b', 'b'), ('c', 'b'), ('d', 'b')))


    def test_series_fillna_leading_b(self) -> None:

        s1 = Series((3.2, 6.4), index=('a', 'b',))
        s2 = s1.fillna_leading(0)
        self.assertTrue(s1.to_pairs() == s2.to_pairs())

    def test_series_fillfalsy_leading_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, 0), index=('a', 'b', 'c', 'd'))
        s2 = Series((0, 0, 6, 0), index=('a', 'b', 'c', 'd'))
        s3 = Series(('', '', '', 4), index=('a', 'b', 'c', 'd'))
        s4 = Series(('', '', '', ''), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.fillfalsy_leading(-1).to_pairs(),
                (('a', 234.3), ('b', 3.2), ('c', 6.4), ('d', 0.0)))

        self.assertEqual(s2.fillfalsy_leading(-1).to_pairs(),
                (('a', -1), ('b', -1), ('c', 6), ('d', 0)))

        self.assertEqual(s3.fillfalsy_leading('a').to_pairs(),
                (('a', 'a'), ('b', 'a'), ('c', 'a'), ('d', 4.0)))

        self.assertEqual(s4.fillfalsy_leading('b').to_pairs(),
                (('a', 'b'), ('b', 'b'), ('c', 'b'), ('d', 'b')))

    def test_series_fillna_trailing_a(self) -> None:

        s1 = Series((234.3, 3.2, np.nan, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((np.nan, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((np.nan, 2.3, 6.4, 4), index=('a', 'b', 'c', 'd'))
        s4 = Series((None, None, None, None), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.fillna_trailing(0).to_pairs(),
                (('a', 234.3), ('b', 3.2), ('c', 0.0), ('d', 0.0)))

        self.assertEqual(s2.fillna_trailing(0).fillna(-1).to_pairs(),
                (('a', -1), ('b', -1), ('c', 6.4), ('d', 0)))

        self.assertEqual(s3.fillna_trailing(2).fillna(-1).to_pairs(),
                (('a', -1.0), ('b', 2.3), ('c', 6.4), ('d', 4.0)))

        self.assertEqual(s4.fillna_trailing('c').to_pairs(),
                (('a', 'c'), ('b', 'c'), ('c', 'c'), ('d', 'c')))


    def test_series_fillfalsy_trailing_a(self) -> None:

        s1 = Series((234.3, 3.2, 0, 0), index=('a', 'b', 'c', 'd'))
        s2 = Series(('', None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((np.nan, 2.3, 6.4, 4), index=('a', 'b', 'c', 'd'))
        s4 = Series(('', '', '', ''), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.fillfalsy_trailing(-1).to_pairs(),
                (('a', 234.3), ('b', 3.2), ('c', -1.0), ('d', -1.0)))

        self.assertEqual(s2.fillfalsy_trailing(100).to_pairs(),
                (('a', ''), ('b', None), ('c', 6.4), ('d', 100)))

        self.assertEqual(s3.fillfalsy_trailing(2).fillna(-1).to_pairs(),
                (('a', -1.0), ('b', 2.3), ('c', 6.4), ('d', 4.0)))

        self.assertEqual(s4.fillfalsy_trailing('c').to_pairs(),
                (('a', 'c'), ('b', 'c'), ('c', 'c'), ('d', 'c')))


    def test_series_fillna_forward_a(self) -> None:

        index = tuple(string.ascii_lowercase[:8])

        # target_index [0 3 6]
        s1 = Series((3, None, None, 4, None, None, 5, 6), index=index)
        self.assertEqual(s1.fillna_forward().to_pairs(),
                (('a', 3), ('b', 3), ('c', 3), ('d', 4), ('e', 4), ('f', 4), ('g', 5), ('h', 6)))

        # target_index [3]
        s2 = Series((None, None, None, 4, None, None, None, None), index=index)
        self.assertEqual(s2.fillna_forward().to_pairs(),
                (('a', None), ('b', None), ('c', None), ('d', 4), ('e', 4), ('f', 4), ('g', 4), ('h', 4)))

        # target_index [0 6]
        s3 = Series((1, None, None, None, None, None, 4, None), index=index)
        self.assertEqual(s3.fillna_forward().to_pairs(),
                (('a', 1), ('b', 1), ('c', 1), ('d', 1), ('e', 1), ('f', 1), ('g', 4), ('h', 4)))

        # target_index [0 7]
        s4 = Series((1, None, None, None, None, None, None, 4), index=index)
        self.assertEqual(s4.fillna_forward().to_pairs(),
                (('a', 1), ('b', 1), ('c', 1), ('d', 1), ('e', 1), ('f', 1), ('g', 1), ('h', 4)))

        # target_index [7]
        s5 = Series((None, None, None, None, None, None, None, 4), index=index)
        self.assertEqual(s5.fillna_forward().to_pairs(),
                (('a', None), ('b', None), ('c', None), ('d', None), ('e', None), ('f', None), ('g', None), ('h', 4)))

        # target index = array([0, 3, 6])
        s6 = Series((2, None, None, 3, 4, 5, 6, None), index=index)
        self.assertEqual(s6.fillna_forward().to_pairs(),
                (('a', 2), ('b', 2), ('c', 2), ('d', 3), ('e', 4), ('f', 5), ('g', 6), ('h', 6))
                )
        # target_index [6]
        s7 = Series((2, 1, 0, 3, 4, 5, 6, None), index=index)
        self.assertEqual(s7.fillna_forward().to_pairs(),
                (('a', 2), ('b', 1), ('c', 0), ('d', 3), ('e', 4), ('f', 5), ('g', 6), ('h', 6)))

        s8 = Series((2, None, None, None, 4, None, 6, None), index=index)
        self.assertEqual(s8.fillna_forward().to_pairs(),
                (('a', 2), ('b', 2), ('c', 2), ('d', 2), ('e', 4), ('f', 4), ('g', 6), ('h', 6)))

        s9 = Series((None, 2, 3, None, 4, None, 6, 7), index=index)
        self.assertEqual(s9.fillna_forward().to_pairs(),
                (('a', None), ('b', 2), ('c', 3), ('d', 3), ('e', 4), ('f', 4), ('g', 6), ('h', 7)))


    def test_series_fillna_forward_b(self) -> None:

        index = tuple(string.ascii_lowercase[:8])

        # target_index [0 3 6]
        s1 = Series((3, None, None, None, 4, None, None, None), index=index)
        s2 = s1.fillna_forward(limit=2)

        self.assertEqual(s2.to_pairs(),
                (('a', 3), ('b', 3), ('c', 3), ('d', None), ('e', 4), ('f', 4), ('g', 4), ('h', None))
                )

        self.assertEqual(s1.fillna_forward(limit=1).to_pairs(),
                (('a', 3), ('b', 3), ('c', None), ('d', None), ('e', 4), ('f', 4), ('g', None), ('h', None)))

        self.assertEqual(s1.fillna_forward(limit=10).to_pairs(),
                (('a', 3), ('b', 3), ('c', 3), ('d', 3), ('e', 4), ('f', 4), ('g', 4), ('h', 4)))

    def test_series_fillna_forward_c(self) -> None:

        # this case shown to justify the slice_condition oassed to slices_from_targets
        index = tuple(string.ascii_lowercase[:8])
        s1 = Series((3, 2, None, 4, None, None, 5, 6), index=index)

        self.assertEqual(s1.fillna_forward().to_pairs(),
                (('a', 3), ('b', 2), ('c', 2), ('d', 4), ('e', 4), ('f', 4), ('g', 5), ('h', 6)))

        self.assertEqual(s1.fillna_backward().to_pairs(),
                (('a', 3), ('b', 2), ('c', 4), ('d', 4), ('e', 5), ('f', 5), ('g', 5), ('h', 6)))


    def test_series_fillfalsy_forward_a(self) -> None:

        index = tuple(string.ascii_lowercase[:8])

        # target_index [0 3 6]
        s1 = Series((3, None, 0, '', 4, None, '', ''), index=index)

        self.assertEqual(s1.fillfalsy_forward(limit=2).to_pairs(),
                (('a', 3), ('b', 3), ('c', 3), ('d', ''), ('e', 4), ('f', 4), ('g', 4), ('h', ''))
                )

        self.assertEqual(s1.fillfalsy_forward(limit=1).to_pairs(),
                (('a', 3), ('b', 3), ('c', 0), ('d', ''), ('e', 4), ('f', 4), ('g', ''), ('h', '')))

        self.assertEqual(s1.fillfalsy_forward(limit=10).to_pairs(),
                (('a', 3), ('b', 3), ('c', 3), ('d', 3), ('e', 4), ('f', 4), ('g', 4), ('h', 4)))


    def test_series_fillna_backward_a(self) -> None:

        index = tuple(string.ascii_lowercase[:8])

        # target_index [0 3 6]
        s1 = Series((3, None, None, 4, None, None, 5, 6), index=index)
        self.assertEqual(s1.fillna_backward().to_pairs(),
                (('a', 3), ('b', 4), ('c', 4), ('d', 4), ('e', 5), ('f', 5), ('g', 5), ('h', 6)))

        s2 = Series((None, None, None, 4, None, None, None, None), index=index)
        self.assertEqual(s2.fillna_backward().to_pairs(),
                (('a', 4), ('b', 4), ('c', 4), ('d', 4), ('e', None), ('f', None), ('g', None), ('h', None)))

        s3 = Series((1, None, None, None, None, None, 4, None), index=index)
        self.assertEqual(s3.fillna_backward().to_pairs(),
                (('a', 1), ('b', 4), ('c', 4), ('d', 4), ('e', 4), ('f', 4), ('g', 4), ('h', None)))

        s4 = Series((1, None, None, None, None, None, None, 4), index=index)
        self.assertEqual(s4.fillna_backward().to_pairs(),
                (('a', 1), ('b', 4), ('c', 4), ('d', 4), ('e', 4), ('f', 4), ('g', 4), ('h', 4)))

        s5 = Series((None, None, None, None, None, None, None, 4), index=index)
        self.assertEqual(s5.fillna_backward().to_pairs(),
                (('a', 4), ('b', 4), ('c', 4), ('d', 4), ('e', 4), ('f', 4), ('g', 4), ('h', 4)))

        s6 = Series((2, None, None, 3, 4, 5, 6, None), index=index)
        self.assertEqual(s6.fillna_backward().to_pairs(),
                (('a', 2), ('b', 3), ('c', 3), ('d', 3), ('e', 4), ('f', 5), ('g', 6), ('h', None)))

        s7 = Series((None, 1, 0, 3, 4, 5, 6, 7), index=index)
        self.assertEqual(s7.fillna_backward().to_pairs(),
            (('a', 1), ('b', 1), ('c', 0), ('d', 3), ('e', 4), ('f', 5), ('g', 6), ('h', 7)))

        s8 = Series((2, None, None, None, 4, None, 6, None), index=index)
        self.assertEqual(s8.fillna_backward().to_pairs(),
            (('a', 2), ('b', 4), ('c', 4), ('d', 4), ('e', 4), ('f', 6), ('g', 6), ('h', None)))

        s9 = Series((None, 2, 3, None, 4, None, 6, 7), index=index)
        self.assertEqual(s9.fillna_backward().to_pairs(),
                (('a', 2), ('b', 2), ('c', 3), ('d', 4), ('e', 4), ('f', 6), ('g', 6), ('h', 7)))


    def test_series_fillna_backward_b(self) -> None:

        index = tuple(string.ascii_lowercase[:8])

        # target_index [0 3 6]
        s1 = Series((3, None, None, 4, None, None, 5, 6), index=index)
        self.assertEqual(s1.fillna_backward(1).to_pairs(),
                (('a', 3), ('b', None), ('c', 4), ('d', 4), ('e', None), ('f', 5), ('g', 5), ('h', 6)))

        s2 = Series((3, None, None, None, 4, None, None, None), index=index)
        self.assertEqual(s2.fillna_backward(2).to_pairs(),
                (('a', 3), ('b', None), ('c', 4), ('d', 4), ('e', 4), ('f', None), ('g', None), ('h', None)))

        s3 = Series((None, 1, None, None, None, None, None, 5), index=index)
        self.assertEqual(s3.fillna_backward(4).to_pairs(),
                (('a', 1), ('b', 1), ('c', None), ('d', 5), ('e', 5), ('f', 5), ('g', 5), ('h', 5)))


    def test_series_fillfalsy_backward_a(self) -> None:

        index = tuple(string.ascii_lowercase[:8])

        # target_index [0 3 6]
        s1 = Series((3, '', '', 4, '', '', 5, 6), index=index)
        self.assertEqual(s1.fillfalsy_backward(1).to_pairs(),
                (('a', 3), ('b', ''), ('c', 4), ('d', 4), ('e', ''), ('f', 5), ('g', 5), ('h', 6)))

        s2 = Series((3, 0, 0, 0, 4, 0, 0, 0), index=index)
        self.assertEqual(s2.fillfalsy_backward(2).to_pairs(),
                (('a', 3), ('b', 0), ('c', 4), ('d', 4), ('e', 4), ('f', 0), ('g', 0), ('h', 0)))

        s3 = Series(('', 1, '', '', '', '', '', 5), index=index)
        self.assertEqual(s3.fillfalsy_backward(4).to_pairs(),
                (('a', 1), ('b', 1), ('c', ''), ('d', 5), ('e', 5), ('f', 5), ('g', 5), ('h', 5)))

    #---------------------------------------------------------------------------
    def test_series_from_element_a(self) -> None:
        s1 = Series.from_element('a', index=range(3))
        self.assertEqual(s1.to_pairs(),
                ((0, 'a'), (1, 'a'), (2, 'a'))
                )

    def test_series_from_element_b(self) -> None:
        s1 = Series.from_element('foo', index=Index((3, 4, 5)), own_index=True)
        self.assertEqual(s1.to_pairs(),
                ((3, 'foo'), (4, 'foo'), (5, 'foo'))
                )

    def test_series_from_element_c(self) -> None:
        s1 = Series.from_element(('a', 'b'), index=Index((3, 4, 5)), own_index=True)
        self.assertEqual(s1.to_pairs(),
                ((3, ('a', 'b')), (4, ('a', 'b')), (5, ('a', 'b')))
                )

    def test_series_from_element_d(self) -> None:
        s1 = Series.from_element(('a', 'b'), index=Index((3, 4, 5)), own_index=True)
        self.assertEqual(s1.to_pairs(),
                ((3, ('a', 'b')), (4, ('a', 'b')), (5, ('a', 'b')))
                )

    def test_series_from_element_e(self) -> None:
        s1 = Series.from_element([0], index=Index((3, 4, 5)), own_index=True)
        self.assertEqual(s1.values.tolist(), [[0], [0], [0]])

        s2 = Series.from_element(range(3), index=Index((3, 4, 5)), own_index=True)
        self.assertEqual(s2.to_pairs(),
                ((3, range(0, 3)), (4, range(0, 3)), (5, range(0, 3))))


    #---------------------------------------------------------------------------
    def test_series_from_items_a(self) -> None:

        def gen() -> tp.Iterator[tp.Tuple[int, int]]:
            r1 = range(10)
            r2 = iter(range(10, 20))
            for x in r1:
                yield x, next(r2)

        s1 = Series.from_items(gen())
        self.assertEqual(s1.loc[7:9].values.tolist(), [17, 18, 19])

        s2 = Series.from_items(dict(a=30, b=40, c=50).items())
        self.assertEqual(s2['c'], 50)
        self.assertEqual(s2['b'], 40)
        self.assertEqual(s2['a'], 30)


    def test_series_from_items_b(self) -> None:

        s1 = Series.from_items(zip(list('abc'), (1,2,3)),
                dtype=str,
                name='foo',
                index_constructor=IndexDefaultFactory('bar'), #type: ignore
                )
        self.assertEqual(s1.name, 'foo')
        self.assertEqual(s1.values.tolist(), ['1', '2', '3'])
        self.assertEqual(s1.index.name, 'bar')

    def test_series_from_items_c(self) -> None:

        s1 = Series.from_items(zip(
                ((1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')), range(4)),
                index_constructor=IndexHierarchy.from_labels)
        self.assertEqual(s1[HLoc[:, 'b']].to_pairs(),
                (((1, 'b'), 1), ((2, 'b'), 3))
                )

    def test_series_from_items_d(self) -> None:

        with self.assertRaises(RuntimeError):
            s1 = Series.from_items(zip(
                    ((1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')), range(4)),
                    index_constructor=IndexHierarchyGO.from_labels)

    def test_series_from_items_e(self) -> None:
        s1 = Series.from_items(zip(('2017-11', '2017-12', '2018-01', '2018-02'),
                range(4)),
                index_constructor=IndexYearMonth)

        self.assertEqual(s1['2017'].to_pairs(),
                ((np.datetime64('2017-11'), 0),
                (np.datetime64('2017-12'), 1))
                )

        self.assertEqual(s1['2018'].to_pairs(),
                ((np.datetime64('2018-01'), 2),
                (np.datetime64('2018-02'), 3))
                )


    #---------------------------------------------------------------------------

    def test_series_contains_a(self) -> None:

        s1 = Series.from_items(zip(('a', 'b', 'c'), (10, 20, 30)))
        self.assertTrue('b' in s1)
        self.assertTrue('c' in s1)
        self.assertTrue('a' in s1)

        self.assertFalse('d' in s1)
        self.assertFalse('' in s1)

    #---------------------------------------------------------------------------


    def test_series_sum_a(self) -> None:

        s1 = Series.from_items(zip(('a', 'b', 'c'), (10, 20, 30)))
        self.assertEqual(s1.sum(), 60)

        s1 = Series.from_items(zip(('a', 'b', 'c', 'd'), (10, 20, 30, np.nan)))
        self.assertEqual(s1.sum(), 60)

        s1 = Series.from_items(zip(('a', 'b', 'c', 'd'), (10, 20, 30, None)))
        self.assertEqual(s1.sum(), 60)


    def test_series_sum_b(self) -> None:
        s1 = Series(list('abc'), dtype=object)
        self.assertEqual(s1.sum(), 'abc')
        # get the same result from character arrays
        s2 = sf.Series(list('abc'))
        self.assertEqual(s2.sum(), 'abc')


    def test_series_cumsum_a(self) -> None:

        s1 = Series.from_items(zip('abc', (10, 20, 30)))

        self.assertEqual(s1.cumsum().to_pairs(),
                (('a', 10), ('b', 30), ('c', 60))
                )

        s2 = Series.from_items(zip('abc', (10, np.nan, 30))).cumsum(skipna=False).fillna(None)
        self.assertEqual(s2.to_pairs(),
                (('a', 10.0), ('b', None), ('c', None))
                )


    def test_series_cumprod_a(self) -> None:

        s1 = Series.from_items(zip('abc', (10, 20, 30)))
        self.assertEqual(
                s1.cumprod().to_pairs(),
                (('a', 10), ('b', 200), ('c', 6000))
                )


    def test_series_median_a(self) -> None:

        s1 = Series.from_items(zip('abcde', (10, 20, 0, 15, 30)))
        self.assertEqual(s1.median(), 15)
        self.assertEqual(s1.median(skipna=False), 15)

        s2 = Series.from_items(zip('abcde', (10, 20, np.nan, 15, 30)))
        self.assertEqual(s2.median(), 17.5)
        self.assertTrue(np.isnan(s2.median(skipna=False)))

        with self.assertRaises(TypeError):
            # should raise with bad keyword argumenty
            s2.median(skip_na=False)

    #---------------------------------------------------------------------------

    def test_series_mask_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(
                s1.mask.loc[['b', 'd']].values.tolist(),
                [False, True, False, True])
        self.assertEqual(s1.mask.iloc[1:].values.tolist(),
                [False, True, True, True])

        self.assertEqual(s1.masked_array.loc[['b', 'd']].sum(), 2)
        self.assertEqual(s1.masked_array.loc[['a', 'b']].sum(), 5)



    def test_series_assign_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))


        self.assertEqual(
                s1.assign.loc[['b', 'd']](3000).values.tolist(), #type: ignore
                [0, 3000, 2, 3000])

        self.assertEqual(
                s1.assign['b':](300).values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                [0, 300, 300, 300])


    def test_series_assign_b(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.isin([2]).items()),
                [('a', False), ('b', False), ('c', True), ('d', False)])

        self.assertEqual(list(s1.isin({2, 3}).items()),
                [('a', False), ('b', False), ('c', True), ('d', True)])

        self.assertEqual(list(s1.isin(range(2, 4)).items()),
                [('a', False), ('b', False), ('c', True), ('d', True)])


    def test_series_assign_c(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.assign.loc['c':](0).to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 0), ('b', 1), ('c', 0), ('d', 0))
                )
        self.assertEqual(s1.assign.loc['c':]((20, 30)).to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 0), ('b', 1), ('c', 20), ('d', 30)))

        self.assertEqual(s1.assign['c':](s1['c':] * 10).to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 0), ('b', 1), ('c', 20), ('d', 30)))

        self.assertEqual(s1.assign['c':](Series.from_dict({'d':40, 'c':60})).to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 0), ('b', 1), ('c', 60), ('d', 40)))


    def test_series_assign_d(self) -> None:
        s1 = Series(tuple('pqrs'), index=('a', 'b', 'c', 'd'))
        s2 = s1.assign['b'](None)
        self.assertEqual(s2.to_pairs(),
                (('a', 'p'), ('b', None), ('c', 'r'), ('d', 's')))
        self.assertEqual(s1.assign['b':](None).to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 'p'), ('b', None), ('c', None), ('d', None)))


    def test_series_assign_e(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series(range(2), index=('c', 'd'))
        self.assertEqual(
                s1.assign[s2.index](s2).to_pairs(),
                (('a', 0), ('b', 1), ('c', 0), ('d', 1))
                )
    def test_series_assign_f(self) -> None:
        s1 = Series(range(5), index=('a', 'b', 'c', 'd', 'e'))

        with self.assertRaises(Exception):
            # cannot have an assignment target that is not in the Series
            s1.assign[['f', 'd']](10)

        self.assertEqual(
                s1.assign[['d', 'c']](Series((10, 20), index=('d', 'c'))).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 10), ('e', 4)))

        self.assertEqual(
                s1.assign[['c', 'd']](Series((10, 20), index=('d', 'c'))).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 10), ('e', 4)))

        self.assertEqual(
                s1.assign[['c', 'd']](Series((10, 20, 30), index=('d', 'c', 'f'))).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 10), ('e', 4)))


        self.assertEqual(
                s1.assign[['c', 'd', 'b']](Series((10, 20), index=('d', 'c')), fill_value=-1).to_pairs(),
                (('a', 0), ('b', -1), ('c', 20), ('d', 10), ('e', 4))
                )

    def test_series_assign_g(self) -> None:
        s1 = Series(range(5), index=('a', 'b', 'c', 'd', 'e'), name='x')

        s2 = Series(list('abc'), index=list('abc'), name='y')

        post = s1.assign[s2.index](s2)
        self.assertEqual(post.name, 'x')
        self.assertEqual(post.values.tolist(), ['a', 'b', 'c', 3, 4])


    def test_series_assign_h(self) -> None:
        s1 = Series(range(5), index=('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(s1.assign['c':].apply(lambda s: -s).to_pairs(), #type: ignore
                (('a', 0), ('b', 1), ('c', -2), ('d', -3), ('e', -4)))


    #---------------------------------------------------------------------------
    def test_series_iloc_extract_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.iloc[0], 0)

        self.assertEqual(s1.iloc[2:].to_pairs(), (('c', 2), ('d', 3)))

    #---------------------------------------------------------------------------


    def test_series_loc_extract_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        with self.assertRaises(KeyError):
            s1.loc[['c', 'd', 'e']] #pylint: disable=W0104

    def test_series_loc_extract_b(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'), name='foo')
        s2 = s1.loc[['b', 'd']]

        self.assertEqual(s2.to_pairs(), (('b', 1), ('d', 3)))
        self.assertEqual(s2.name, 'foo')

    def test_series_loc_extract_c(self) -> None:
        s = sf.Series(range(5),
                index=sf.IndexHierarchy.from_labels(
                (('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b'), ('b', 'c'))))

        # this selection returns just a single value
        # import ipdb; ipdb.set_trace()
        s2 = s.loc[sf.HLoc[:, 'c']]
        self.assertEqual(s2.__class__, s.__class__)
        self.assertEqual(s2.to_pairs(), ((('b', 'c'), 4),))

        # this selection yields a series
        self.assertEqual(s.loc[sf.HLoc[:, 'a']].to_pairs(),
                ((('a', 'a'), 0), (('b', 'a'), 2)))


    def test_series_loc_extract_d(self) -> None:
        s = sf.Series(range(5),
                index=sf.IndexHierarchy.from_labels(
                (('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b'), ('b', 'c'))))
        # leaf loc selection must be terminal; using a slice or list is an exception
        with self.assertRaises(RuntimeError):
            s.loc['a', :] #pylint: disable=W0104

        with self.assertRaises(RuntimeError):
            s.loc[['a', 'b'], 'b'] #pylint: disable=W0104


    def test_series_loc_extract_e(self) -> None:
        s1 = sf.Series(range(4), index=sf.IndexHierarchy.from_product(['A', 'B'], [1, 2]))

        self.assertEqual(s1.loc[('B', 1)], 2)
        self.assertEqual(s1.loc[sf.HLoc['B', 1]], 2)
        self.assertEqual(s1.iloc[2], 2)


    def test_series_loc_extract_f(self) -> None:
        s1 = sf.Series(range(4), index=sf.IndexHierarchy.from_product(['A', 'B'], [1, 2]))

        post1 = s1[HLoc['A', [2]]]
        self.assertEqual(post1.to_pairs(), ((('A', 2), 1),))

        post2 = s1[HLoc['A', 2]]
        self.assertEqual(post2, 1)


    def test_series_loc_extract_g(self) -> None:

        s1 = Series(('a', 'b', 'c', 'd'))
        post = s1.loc[0:2].to_pairs()
        self.assertEqual(post,
                ((0, 'a'), (1, 'b'), (2, 'c'))
                )

    def test_series_loc_extract_h(self) -> None:
        a1 = np.array((None, None, None))
        a1[2] = [3, 4]
        a1[0] = [9]
        s1 = Series(a1, index=('a', 'b', 'c'))
        self.assertEqual(s1['a'], [9])
        self.assertEqual(s1['c'], [3, 4])


    def test_series_loc_extract_i(self) -> None:
        a1 = np.array((None, None, None))
        a1[2] = np.array([3, 4])
        a1[0] = np.array([9])
        s1 = Series(a1, index=('a', 'b', 'c'))
        self.assertEqual(s1['a'].tolist(), [9])
        self.assertEqual(s1['c'].tolist(), [3, 4])


    #---------------------------------------------------------------------------

    def test_series_group_a(self) -> None:

        s1 = Series((0, 1, 0, 1), index=('a', 'b', 'c', 'd'))

        groups = tuple(s1.iter_group_items())

        self.assertEqual([g[0] for g in groups], [0, 1])

        self.assertEqual([g[1].to_pairs() for g in groups],
                [(('a', 0), ('c', 0)), (('b', 1), ('d', 1))])

    def test_series_group_b(self) -> None:

        s1 = Series(('foo', 'bar', 'foo', 20, 20),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        groups = tuple(s1.iter_group_items())


        self.assertEqual([g[0] for g in groups],
                [20, 'bar', 'foo'])
        self.assertEqual([g[1].to_pairs() for g in groups],
                [(('d', 20), ('e', 20)), (('b', 'bar'),), (('a', 'foo'), ('c', 'foo'))])


    def test_series_group_c(self) -> None:

        s1 = Series((10, 10, 10, 20, 20),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        groups = tuple(s1.iter_group())
        self.assertEqual([g.sum() for g in groups], [30, 40])

        self.assertEqual(
                s1.iter_group().apply(np.sum).to_pairs(),
                ((10, 30), (20, 40)))

        self.assertEqual(
                s1.iter_group_items().apply(lambda g, s: (g * s).values.tolist()).to_pairs(),
                ((10, [100, 100, 100]), (20, [400, 400])))


    #---------------------------------------------------------------------------

    def test_series_iter_element_a(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        self.assertEqual([x for x in s1.iter_element()], [10, 3, 15, 21, 28])

        self.assertEqual([x for x in s1.iter_element_items()],
                        [('a', 10), ('b', 3), ('c', 15), ('d', 21), ('e', 28)])

        self.assertEqual(s1.iter_element().apply(lambda x: x * 20).to_pairs(),
                (('a', 200), ('b', 60), ('c', 300), ('d', 420), ('e', 560)))

        self.assertEqual(
                s1.iter_element_items().apply(lambda k, v: v * 20 if k == 'b' else 0).to_pairs(),
                (('a', 0), ('b', 60), ('c', 0), ('d', 0), ('e', 0)))


    def test_series_iter_element_b(self) -> None:

        s1 = Series((10, 3, 15, 21, 28, 50),
                index=IndexHierarchy.from_product(tuple('ab'), tuple('xyz')),
                dtype=object)
        s2 = s1.iter_element().apply(str, name='foo')
        self.assertEqual(s2.index.__class__, IndexHierarchy)
        self.assertEqual(s2.name, 'foo')

        self.assertEqual(s2.to_pairs(),
                ((('a', 'x'), '10'), (('a', 'y'), '3'), (('a', 'z'), '15'), (('b', 'x'), '21'), (('b', 'y'), '28'), (('b', 'z'), '50')))


    def test_series_iter_element_c(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                )
        with self.assertRaises(RuntimeError):
            s1.iter_element().apply({15:150})

        post1 = tuple(s1.iter_element_items().map_any_iter_items({('c', 15): 150}))
        self.assertEqual(post1,
                (('a', 10), ('b', 3), ('c', 150), ('d', 21), ('e', 28)))

        post2 = tuple(s1.iter_element_items().map_fill_iter_items(
                {('c', 15): 150}, fill_value=0))
        self.assertEqual(post2,
                (('a', 0), ('b', 0), ('c', 150), ('d', 0), ('e', 0)))

        post3 = tuple(s1.iter_element_items().map_all_iter_items(
                {(k, v): v * 10 for k, v in s1.items()}))
        self.assertEqual(post3,
                (('a', 100), ('b', 30), ('c', 150), ('d', 210), ('e', 280)))


    #---------------------------------------------------------------------------

    def test_series_iter_element_map_any_a(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        post = s1.iter_element().map_any({3: 100, 21: 101})

        self.assertEqual(post.to_pairs(),
                (('a', 10), ('b', 100), ('c', 15), ('d', 101), ('e', 28))
                )


    def test_series_iter_element_map_any_b(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        s2 = Series((100, 101), index=(3, 21))

        post = s1.iter_element().map_any(s2)

        self.assertEqual(post.to_pairs(),
                (('a', 10), ('b', 100), ('c', 15), ('d', 101), ('e', 28))
                )


    def test_series_iter_element_map_any_c(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        s2 = Series((100, 101), index=(3, 21))

        self.assertEqual(tuple(s2.iter_element().map_any_iter(s2)),
            (100, 101))
        self.assertEqual(tuple(s2.iter_element().map_any_iter_items(s2)),
            ((3, 100), (21, 101)))


    def test_series_iter_element_map_all_a(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        with self.assertRaises(KeyError):
            post = s1.iter_element().map_all({3: 100, 21: 101})

        post = s1.iter_element().map_all({v: k for k, v in s1.items()})

        self.assertEqual(post.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd'), ('e', 'e'))
                )

    def test_series_iter_element_map_all_b(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        s2 = Series((100, 101), index=(3, 21))

        with self.assertRaises(KeyError):
            post = s1.iter_element().map_all(s2)

        s3 = Series.from_items((v, i) for i, v in enumerate(s1.values))

        self.assertEqual(s3.to_pairs(),
                ((10, 0), (3, 1), (15, 2), (21, 3), (28, 4))
                )

    def test_series_iter_element_map_all_c(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        s2 = Series.from_items((v, i) for i, v in enumerate(s1.values))

        self.assertEqual(tuple(s1.iter_element().map_all_iter(s2)),
                (0, 1, 2, 3, 4))

        self.assertEqual(tuple(s1.iter_element().map_all_iter_items(s2)),
                (('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4))
                )

    def test_series_iter_element_map_fill_a(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        post = s1.iter_element().map_fill({21: 100, 28: 101}, fill_value=-1)
        self.assertEqual(post.to_pairs(),
                (('a', -1), ('b', -1), ('c', -1), ('d', 100), ('e', 101))
                )

    def test_series_iter_element_map_fill_b(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        s2 = Series((100, 101), index=(21, 28))

        post = s1.iter_element().map_fill(s2, fill_value=-1)
        self.assertEqual(post.to_pairs(),
                (('a', -1), ('b', -1), ('c', -1), ('d', 100), ('e', 101))
                )


    def test_series_iter_element_map_fill_c(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        s2 = Series((100, 101), index=(21, 28))

        self.assertEqual(tuple(s1.iter_element().map_fill_iter(s2, fill_value=0)),
                (0, 0, 0, 100, 101))

        self.assertEqual(tuple(s1.iter_element().map_fill_iter_items(s2, fill_value=0)),
                (('a', 0), ('b', 0), ('c', 0), ('d', 100), ('e', 101))
                )



    #---------------------------------------------------------------------------
    def test_series_sort_index_a(self) -> None:

        s1 = Series((10, 3, 28, 21, 15),
                index=('a', 'c', 'b', 'e', 'd'),
                dtype=object,
                name='foo')

        s2 = s1.sort_index()
        self.assertEqual(s2.to_pairs(),
                (('a', 10), ('b', 28), ('c', 3), ('d', 15), ('e', 21)))
        self.assertEqual(s2.name, s1.name)

        s3 = s1.sort_values()
        self.assertEqual(s3.to_pairs(),
                (('c', 3), ('a', 10), ('d', 15), ('e', 21), ('b', 28)))
        self.assertEqual(s3.name, s1.name)


    def test_series_sort_index_b(self) -> None:

        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15')
        s = Series(list('abcd'), index=index)

        post = s.sort_index(ascending=False)

        self.assertEqual(
                post.to_pairs(),
                ((np.datetime64('2018-03'), 'd'), (np.datetime64('2018-02'), 'c'), (np.datetime64('2018-01'), 'b'), (np.datetime64('2017-12'), 'a'))
                )

        self.assertEqual(post.index.__class__, IndexYearMonth)


    def test_series_sort_index_c(self) -> None:

        index = IndexHierarchy.from_product((0, 1), (10, 20))
        s = Series(list('abcd'), index=index)

        post = s.sort_index(ascending=False)

        self.assertEqual(post.to_pairs(),
            (((1, 20), 'd'), ((1, 10), 'c'), ((0, 20), 'b'), ((0, 10), 'a'))
            )
        self.assertEqual(post.index.__class__, IndexHierarchy)


    def test_series_sort_index_d(self) -> None:

        index = IndexHierarchy.from_product((0, 1), (10, 20), name='foo')
        s1 = Series(list('abcd'), index=index)
        s2 = s1.sort_index()
        self.assertEqual(s2.index.name, s1.index.name)



    def test_series_sort_index_e(self) -> None:

        index = IndexHierarchy.from_product(('c', 'b', 'a'), (20, 10), name='foo')
        s1 = Series(range(6), index=index)
        s2 = s1.sort_index()
        self.assertEqual(s2.values.tolist(),
                [5, 4, 3, 2, 1, 0])

        # this is a stable sort, so we retain inner order
        s3 = s1.sort_index(key=lambda i: i.values_at_depth(0))
        self.assertEqual(s3.values.tolist(),
                [4, 5, 2, 3, 0, 1])

        s4 = s1.sort_index(key=lambda i: i.rehierarch([1, 0]))
        self.assertEqual(s4.values.tolist(),
                [5, 4, 3, 2, 1, 0])

        with self.assertRaises(RuntimeError):
            _ = s1.sort_index(key=lambda i: i.values_at_depth(0)[:2])



    def test_series_sort_index_f(self) -> None:

        ih1 = IndexHierarchy.from_product(('a', 'b'), (1, 5, 3, -4))
        s1 = Series(range(len(ih1)), index=ih1)

        self.assertEqual(s1.sort_index(ascending=(False, True)).to_pairs(),
                ((('b', -4), 7), (('b', 1), 4), (('b', 3), 6), (('b', 5), 5), (('a', -4), 3), (('a', 1), 0), (('a', 3), 2), (('a', 5), 1))
                )

        self.assertEqual(s1.sort_index(ascending=(True, False)).to_pairs(),
                ((('a', 5), 1), (('a', 3), 2), (('a', 1), 0), (('a', -4), 3), (('b', 5), 5), (('b', 3), 6), (('b', 1), 4), (('b', -4), 7))
                )



    #---------------------------------------------------------------------------
    def test_series_sort_values_a(self) -> None:

        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15', name='foo')
        s = Series(list('abcd'), index=index)

        post = s.sort_values(ascending=False)

        self.assertEqual(
                post.to_pairs(),
                ((np.datetime64('2018-03'), 'd'), (np.datetime64('2018-02'), 'c'), (np.datetime64('2018-01'), 'b'), (np.datetime64('2017-12'), 'a'))
                )

        self.assertEqual(post.index.__class__, IndexYearMonth)
        self.assertEqual(post.index.name, 'foo')

    def test_series_sort_values_b(self) -> None:

        index = IndexHierarchy.from_product((0, 1), (10, 20))
        s = Series(list('abcd'), index=index)

        post = s.sort_values(ascending=False)

        self.assertEqual(post.to_pairs(),
                (((1, 20), 'd'), ((1, 10), 'c'), ((0, 20), 'b'), ((0, 10), 'a'))
                )

        self.assertEqual(post.index.__class__, IndexHierarchy)

    def test_series_sort_values_c(self) -> None:

        index = IndexDate(('2017-12-03', '2020-03-15', '2016-01-31'), name='foo')
        s = Series(list('abc'), index=index)

        self.assertEqual(s.sort_values(
                key=lambda s: s.index.via_dt.year).values.tolist(),
                ['c', 'a', 'b'])

        self.assertEqual(s.sort_values(
                key=lambda s: s.index.via_dt.month).values.tolist(),
                ['c', 'b', 'a'])

        self.assertEqual(s.sort_values(
                key=lambda s: s.index.via_dt.day).values.tolist(),
                ['a', 'b', 'c'])


        self.assertEqual(s.sort_values(
                key=lambda s:s.via_str.find('b')).values.tolist(),
                ['a', 'c', 'b'])


    def test_series_sort_values_d(self) -> None:

        index = IndexHierarchy.from_product((0, 1), (10, 20))
        s = Series(list('abcd'), index=index)
        with self.assertRaises(RuntimeError):
            s.sort_values(ascending=(False, True))

    #---------------------------------------------------------------------------
    def test_series_reversed(self) -> None:

        idx = tuple('abcd')
        s = Series(range(4), index=idx)
        self.assertTrue(tuple(reversed(s)) == tuple(reversed(idx)))

    #---------------------------------------------------------------------------

    def test_series_relabel_a(self) -> None:

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.relabel({'b': 'bbb'})
        self.assertEqual(s2.to_pairs(),
                (('a', 0), ('bbb', 1), ('c', 2), ('d', 3)))

        self.assertEqual(mloc(s2.values), mloc(s1.values))


    def test_series_relabel_b(self) -> None:

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = s1.relabel({'a':'x', 'b':'y', 'c':'z', 'd':'q'})

        self.assertEqual(list(s2.items()),
            [('x', 0), ('y', 1), ('z', 2), ('q', 3)])


    def test_series_relabel_c(self) -> None:

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = s1.relabel(IndexAutoFactory)
        self.assertEqual(
                s2.to_pairs(),
                ((0, 0), (1, 1), (2, 2), (3, 3))
                )

    def test_series_relabel_d(self) -> None:

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        idx = IndexHierarchy.from_product(('a', 'b'), (1, 2))
        s2 = s1.relabel(idx)
        self.assertEqual(s2.to_pairs(),
            ((('a', 1), 0), (('a', 2), 1), (('b', 1), 2), (('b', 2), 3))
            )

    def test_series_relabel_e(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.relabel(IndexAutoFactory)
        self.assertEqual(s2.to_pairs(),
                ((0, 0), (1, 1), (2, 2), (3, 3))
                )


    def test_series_relabel_f(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        # reuse the same instance
        s2 = s1.relabel(None)
        self.assertEqual(id(s1.index), id(s2.index))


    def test_series_relabel_g(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        # reuse the same instance
        with self.assertRaises(RuntimeError):
            s1.relabel({'d', 'c', 'b', 'a'})



    #---------------------------------------------------------------------------

    def test_series_relabel_flat_a(self) -> None:

        s1 = Series(range(4), index=IndexHierarchy.from_product((10, 20), ('a', 'b')))

        s2 = s1.relabel_flat()
        self.assertEqual(s2.to_pairs(),
                (((10, 'a'), 0), ((10, 'b'), 1), ((20, 'a'), 2), ((20, 'b'), 3)))

        with self.assertRaises(RuntimeError):
            _ = s2.relabel_flat()



    def test_series_relabel_drop_level_a(self) -> None:

        s1 = Series(range(2), index=IndexHierarchy.from_labels(((10, 20), ('a', 'b'))))

        s2 = s1.relabel_level_drop()
        self.assertEqual(s2.to_pairs(), ((20, 0), ('b', 1)))

        with self.assertRaises(RuntimeError):
            _ = s2.relabel_level_drop()


    #---------------------------------------------------------------------------

    def test_series_rehierarch_a(self) -> None:

        colors = ('red', 'green')
        shapes = ('square', 'circle', 'triangle')
        textures = ('smooth', 'rough')

        s1 = sf.Series(range(12), index=sf.IndexHierarchy.from_product(shapes, colors, textures))

        s2 = s1.rehierarch((2,1,0))

        self.assertEqual(s2.to_pairs(),
                ((('smooth', 'red', 'square'), 0), (('smooth', 'red', 'circle'), 4), (('smooth', 'red', 'triangle'), 8), (('smooth', 'green', 'square'), 2), (('smooth', 'green', 'circle'), 6), (('smooth', 'green', 'triangle'), 10), (('rough', 'red', 'square'), 1), (('rough', 'red', 'circle'), 5), (('rough', 'red', 'triangle'), 9), (('rough', 'green', 'square'), 3), (('rough', 'green', 'circle'), 7), (('rough', 'green', 'triangle'), 11))
                )


    def test_series_rehierarch_b(self) -> None:
        s1 = sf.Series(range(8), index=sf.IndexHierarchy.from_product(('B', 'A'), (100, 2), ('iv', 'ii')))

        self.assertEqual(s1.rehierarch((2,1,0)).to_pairs(),
                ((('iv', 100, 'B'), 0), (('iv', 100, 'A'), 4), (('iv', 2, 'B'), 2), (('iv', 2, 'A'), 6), (('ii', 100, 'B'), 1), (('ii', 100, 'A'), 5), (('ii', 2, 'B'), 3), (('ii', 2, 'A'), 7))
                )

        self.assertEqual(s1.rehierarch((1,2,0)).to_pairs(),
                (((100, 'iv', 'B'), 0), ((100, 'iv', 'A'), 4), ((100, 'ii', 'B'), 1), ((100, 'ii', 'A'), 5), ((2, 'iv', 'B'), 2), ((2, 'iv', 'A'), 6), ((2, 'ii', 'B'), 3), ((2, 'ii', 'A'), 7))
                )


    def test_series_rehierarch_c(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        with self.assertRaises(RuntimeError):
            s1.rehierarch(())


    #---------------------------------------------------------------------------


    def test_series_get_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.get('q'), None)
        self.assertEqual(s1.get('a'), 0)
        self.assertEqual(s1.get('f', -1), -1)

    #---------------------------------------------------------------------------

    def test_series_all_a(self) -> None:
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.all(), False)
        self.assertEqual(s1.any(), True)

    def test_series_all_b(self) -> None:
        s1 = Series([True, True, np.nan, True], index=('a', 'b', 'c', 'd'), dtype=object)

        self.assertEqual(s1.all(skipna=True), True)
        self.assertEqual(s1.any(), True)
        self.assertTrue(s1.all(skipna=False))

    def test_series_all_c(self) -> None:
        s1 = Series([1, np.nan, 1], index=('a', 'b', 'c'))
        self.assertEqual(s1.all(), True)
        self.assertEqual(s1.any(), True)

    def test_series_all_d(self) -> None:
        s1 = Series([True, np.nan, True], index=('a', 'b', 'c'))
        self.assertEqual(s1.all(), True)
        self.assertEqual(s1.any(), True)

    def test_series_all_e(self) -> None:
        s1 = Series([True, None, True], index=('a', 'b', 'c'))
        self.assertEqual(s1.all(), True)
        self.assertEqual(s1.any(), True)


    def test_series_all_f(self) -> None:
        s1 = Series([True, None, 1], index=('a', 'b', 'c'))
        self.assertFalse(s1.all(skipna=False))
        self.assertTrue(s1.any(skipna=False))

    def test_series_all_g(self) -> None:
        s1 = Series(['', 'sdf', np.nan], index=('a', 'b', 'c'))
        self.assertFalse(s1.all())
        self.assertFalse(s1.all(skipna=False))
        self.assertTrue(s1.any(skipna=False))

    def test_series_all_h(self) -> None:
        s1 = Series(['', 'sdf', 'wer'], index=('a', 'b', 'c'))
        self.assertEqual(s1.all(), False)
        self.assertEqual(s1.any(), True)

    def test_series_all_i(self) -> None:
        s1 = Series(['sdf', 'wer'], index=('a', 'b'))
        self.assertEqual(s1.all(), True)
        self.assertEqual(s1.any(), True)

    def test_series_all_j(self) -> None:
        s1 = Series(['', 'sdf', 'wer', 30], index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.all(), False)
        self.assertEqual(s1.any(), True)

    def test_series_all_k(self) -> None:
        s1 = Series(['sdf', 'wer', 30], index=('a', 'b', 'c'))
        self.assertEqual(s1.all(), True)
        self.assertEqual(s1.any(), True)

    def test_series_all_m(self) -> None:
        s1 = Series(['', 0, False], index=('a', 'b', 'c'))
        self.assertEqual(s1.all(), False)
        self.assertEqual(s1.any(), False)


    def test_series_all_n(self) -> None:
        s1 = Series(['foo', None, 'bar'])
        self.assertEqual(s1.all(skipna=False), False)
        self.assertEqual(s1.any(), True)



    #---------------------------------------------------------------------------

    def test_series_unique_a(self) -> None:
        s1 = Series([10, 10, 2, 2], index=('a', 'b', 'c', 'd'), dtype=np.int64)

        self.assertEqual(s1.unique().tolist(), [2, 10])

        s2 = Series(['b', 'b', 'c', 'c'], index=('a', 'b', 'c', 'd'), dtype=object)
        self.assertEqual(s2.unique().tolist(), ['b', 'c'])


    def test_series_unique_b(self) -> None:
        s1 = Series([10, 10, 2, 2], index=('a', 'b', 'c', 'd'), dtype=np.int64)

        self.assertEqual(s1.unique().tolist(), [2, 10])

        s2 = Series(['b', 'b', 'c', 'c'], index=('a', 'b', 'c', 'd'), dtype=object)
        self.assertEqual(s2.unique().tolist(), ['b', 'c'])



    def test_series_duplicated_a(self) -> None:
        s1 = Series([1, 10, 10, 5, 2, 2],
                index=('a', 'b', 'c', 'd', 'e', 'f'), dtype=np.int64)

        # this is showing all duplicates, not just the first-found
        self.assertEqual(s1.duplicated().to_pairs(),
                (('a', False), ('b', True), ('c', True), ('d', False), ('e', True), ('f', True)))

        self.assertEqual(s1.duplicated(exclude_first=True).to_pairs(),
                (('a', False), ('b', False), ('c', True), ('d', False), ('e', False), ('f', True)))

        self.assertEqual(s1.duplicated(exclude_last=True).to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', False), ('e', True), ('f', False)))


    def test_series_duplicated_b(self) -> None:
        s1 = Series([5, 3, 3, 3, 7, 2, 2, 2, 1],
                index=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), dtype=np.int64)

        # this is showing all duplicates, not just the first-found
        self.assertEqual(s1.duplicated().to_pairs(),
                (('a', False), ('b', True), ('c', True),
                ('d', True), ('e', False), ('f', True),
                ('g', True), ('h', True), ('i', False),
                ))

        self.assertEqual(s1.duplicated(exclude_first=True).to_pairs(),
                (('a', False), ('b', False), ('c', True),
                ('d', True), ('e', False), ('f', False),
                ('g', True), ('h', True), ('i', False),
                ))

        self.assertEqual(s1.duplicated(exclude_last=True).to_pairs(),
                (('a', False), ('b', True), ('c', True),
                ('d', False), ('e', False), ('f', True),
                ('g', True), ('h', False), ('i', False),
                ))


    def test_series_drop_duplicated_a(self) -> None:
        s1 = Series([5, 3, 3, 3, 7, 2, 2, 2, 1],
                index=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), dtype=int)

        self.assertEqual(s1.drop_duplicated().to_pairs(),
                (('a', 5), ('e', 7), ('i', 1)))

        self.assertEqual(s1.drop_duplicated(exclude_first=True).to_pairs(),
                (('a', 5), ('b', 3), ('e', 7), ('f', 2), ('i', 1))
                )


    def test_series_reindex_add_level(self) -> None:
        s1 = Series(['a', 'b', 'c'])

        s2 = s1.relabel_level_add('I')
        self.assertEqual(s2.index.depth, 2)
        self.assertEqual(s2.to_pairs(),
                ((('I', 0), 'a'), (('I', 1), 'b'), (('I', 2), 'c')))

        s3 = s2.relabel_flat()
        self.assertEqual(s3.index.depth, 1)
        self.assertEqual(s3.to_pairs(),
                ((('I', 0), 'a'), (('I', 1), 'b'), (('I', 2), 'c')))


    def test_series_drop_level_a(self) -> None:
        s1 = Series(['a', 'b', 'c'],
                index=IndexHierarchy.from_labels([('A', 1), ('B', 1), ('C', 1)]))
        s2 = s1.relabel_level_drop(-1)
        self.assertEqual(s2.to_pairs(),
                (('A', 'a'), ('B', 'b'), ('C', 'c'))
                )

    #---------------------------------------------------------------------------
    def test_series_from_pandas_a(self) -> None:
        import pandas as pd

        pds = pd.Series([3,4,5], index=list('abc'))
        sfs = Series.from_pandas(pds)
        self.assertEqual(list(pds.items()), list(sfs.items()))

        # mutate Pandas
        pds['c'] = 50
        self.assertNotEqual(pds['c'], sfs['c'])

        # owning data
        pds = pd.Series([3,4,5], index=list('abc'))
        sfs = Series.from_pandas(pds, own_data=True)
        self.assertEqual(list(pds.items()), list(sfs.items()))

    def test_series_from_pandas_b(self) -> None:
        import pandas as pd

        pds = pd.Series([3,4,5], index=list('abc'))
        if hasattr(pds, 'convert_dtypes'):
            pds = pds.convert_dtypes()
        sfs = Series.from_pandas(pds)
        self.assertEqual(list(pds.items()), list(sfs.items()))

        # mutate Pandas
        pds['c'] = 50
        self.assertNotEqual(pds['c'], sfs['c'])

        # owning data
        pds = pd.Series([3,4,5], index=list('abc'))
        sfs = Series.from_pandas(pds, own_data=True)
        self.assertEqual(list(pds.items()), list(sfs.items()))


    def test_series_from_pandas_c(self) -> None:
        import pandas as pd

        pds1 = pd.Series(['a', 'b', np.nan], index=list('abc'))
        if hasattr(pds1, 'convert_dtypes'):
            pds1 = pds1.convert_dtypes()
            sfs1 = Series.from_pandas(pds1)
            self.assertEqual(sfs1.dtype, np.dtype('O'))

        pds2 = pd.Series(['a', 'b', 'c'], index=list('abc'))
        if hasattr(pds2,  'convert_dtypes'):
            pds2 = pds2.convert_dtypes()
            sfs2 = Series.from_pandas(pds2)
            self.assertEqual(sfs2.dtype, np.dtype('<U1'))

        pds3 = pd.Series([False, True, np.nan], index=list('abc'))
        if hasattr(pds3,  'convert_dtypes'):
            pds3 = pds3.convert_dtypes()
            sfs3 = Series.from_pandas(pds3)
            self.assertEqual(sfs3.dtype, np.dtype('O'))

        pds4 = pd.Series([False, True, np.nan], index=list('abc'))
        if hasattr(pds4,  'convert_dtypes'):
            pds4 = pds4.convert_dtypes()
            sfs4 = Series.from_pandas(pds4)
            self.assertEqual(sfs4.dtype, np.dtype('O'))

        pds5 = pd.Series([False, True, False], index=list('abc'))
        if hasattr(pds5,  'convert_dtypes'):
            pds5 = pds5.convert_dtypes()
            sfs5 = Series.from_pandas(pds5)
            self.assertEqual(sfs5.dtype, np.dtype('bool'))


    def test_series_from_pandas_d(self) -> None:
        import pandas as pd

        pds1 = pd.Series(['a', 'b', np.nan], index=list('abc'))
        if hasattr(pds1, 'convert_dtypes'):
            pds1 = pds1.convert_dtypes()
            sfs1 = Series.from_pandas(pds1, own_data=True)
            self.assertEqual(sfs1.dtype, np.dtype('O'))

        pds2 = pd.Series(['a', 'b', 'c'], index=list('abc'))
        if hasattr(pds2,  'convert_dtypes'):
            pds2 = pds2.convert_dtypes()
            sfs2 = Series.from_pandas(pds2, own_data=True)
            self.assertEqual(sfs2.dtype, np.dtype('<U1'))

        pds3 = pd.Series([False, True, np.nan], index=list('abc'))
        if hasattr(pds3,  'convert_dtypes'):
            pds3 = pds3.convert_dtypes()
            sfs3 = Series.from_pandas(pds3, own_data=True)
            self.assertEqual(sfs3.dtype, np.dtype('O'))

        pds4 = pd.Series([False, True, np.nan], index=list('abc'))
        if hasattr(pds4,  'convert_dtypes'):
            pds4 = pds4.convert_dtypes()
            sfs4 = Series.from_pandas(pds4, own_data=True)
            self.assertEqual(sfs4.dtype, np.dtype('O'))

        pds5 = pd.Series([False, True, False], index=list('abc'))
        if hasattr(pds5,  'convert_dtypes'):
            pds5 = pds5.convert_dtypes()
            sfs5 = Series.from_pandas(pds5, own_data=True)
            self.assertEqual(sfs5.dtype, np.dtype('bool'))



    def test_series_from_pandas_e(self) -> None:
        import pandas as pd

        pds1 = pd.Series(['a', 'b', None], index=list('abc'))
        self.assertEqual(sf.Series.from_pandas(pds1,
                index_constructor=sf.IndexAutoFactory).to_pairs(),
                ((0, 'a'), (1, 'b'), (2, None))
                )


    def test_series_from_pandas_f(self) -> None:
        import pandas as pd

        pds1 = pd.Series(['a', 'b', None], index=('2012', '2013', '2014'))
        self.assertEqual(sf.Series.from_pandas(pds1,
                index_constructor=sf.IndexYear).to_pairs(),
                ((np.datetime64('2012'), 'a'),
                (np.datetime64('2013'), 'b'),
                (np.datetime64('2014'), None))
                )

    def test_series_from_pandas_g(self) -> None:
        with self.assertRaises(ErrorInitSeries):
            Series.from_pandas(Series(['a', 'b', None], index=list('abc')))

    def test_series_from_pandas_h(self) -> None:
        import pandas as pd

        pds1 = pd.Series(['a', 'b', None], index=list('abc'), name='foo')

        self.assertEqual(sf.Series.from_pandas(pds1).name, 'foo')
        self.assertEqual(sf.Series.from_pandas(pds1, name=None).name, None)
        self.assertEqual(sf.Series.from_pandas(pds1, name='bar').name, 'bar')


    #---------------------------------------------------------------------------
    def test_series_to_pandas_a(self) -> None:

        s1 = Series(range(4),
            index=IndexHierarchy.from_product(('a', 'b'), ('x', 'y')))
        df = s1.to_pandas()

        self.assertEqual(df.index.values.tolist(),
                [('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y')]
                )
        self.assertEqual(df.values.tolist(),
                [0, 1, 2, 3]
                )

    def test_series_to_pandas_b(self) -> None:

        from pandas import Timestamp

        s1 = Series(range(4),
            index=IndexDate(('2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05')))
        df = s1.to_pandas()

        self.assertEqual(df.index.tolist(),
            [Timestamp('2018-01-02 00:00:00'), Timestamp('2018-01-03 00:00:00'), Timestamp('2018-01-04 00:00:00'), Timestamp('2018-01-05 00:00:00')]
            )
        self.assertEqual(df.values.tolist(),
            [0, 1, 2, 3]
            )



    def test_series_astype_a(self) -> None:

        s1 = Series(['a', 'b', 'c'])

        s2 = s1.astype(object)
        self.assertEqual(s2.to_pairs(),
                ((0, 'a'), (1, 'b'), (2, 'c')))
        self.assertTrue(s2.dtype == object)

        # we cannot convert to float
        with self.assertRaises(ValueError):
            s1.astype(float)

    def test_series_astype_b(self) -> None:

        s1 = Series([1, 3, 4, 0])

        s2 = s1.astype(bool)
        self.assertEqual(
                s2.to_pairs(),
                ((0, True), (1, True), (2, True), (3, False)))
        self.assertTrue(s2.dtype == bool)


    def test_series_min_max_a(self) -> None:

        s1 = Series([1, 3, 4, 0])
        self.assertEqual(s1.min(), 0)
        self.assertEqual(s1.max(), 4)


        s2 = sf.Series([-1, 4, None, np.nan])
        self.assertEqual(s2.min(), -1)
        with self.assertRaises(TypeError):
            s2.min(skipna=False)

        self.assertEqual(s2.max(), 4)
        with self.assertRaises(TypeError):
            s2.max(skipna=False)

        s3 = sf.Series([-1, 4, None])
        self.assertEqual(s3.min(), -1)
        with self.assertRaises(TypeError):
            s2.max(skipna=False)



    def test_series_min_max_b(self) -> None:
        # string objects work as expected; when fixed length strings, however, the do not

        s1 = Series(list('abc'), dtype=object)
        self.assertEqual(s1.min(), 'a')
        self.assertEqual(s1.max(), 'c')

        # get the same result from character arrays
        s2 = sf.Series(list('abc'))
        self.assertEqual(s2.min(), 'a')
        self.assertEqual(s2.max(), 'c')

    #---------------------------------------------------------------------------

    def test_series_clip_a(self) -> None:

        s1 = Series(range(6), index=list('abcdef'))

        self.assertEqual(s1.clip(lower=3).to_pairs(),
                (('a', 3), ('b', 3), ('c', 3), ('d', 3), ('e', 4), ('f', 5))
                )

        self.assertEqual(s1.clip(lower=-1).to_pairs(),
                (('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 4), ('f', 5))
                )

        self.assertEqual(s1.clip(upper=-1).to_pairs(),
                (('a', -1), ('b', -1), ('c', -1), ('d', -1), ('e', -1), ('f', -1))
                )

        self.assertEqual(s1.clip(upper=3).to_pairs(),
                (('a', 0), ('b', 1), ('c', 2), ('d', 3), ('e', 3), ('f', 3))
                )


    def test_series_clip_b(self) -> None:
        s1 = Series(range(6), index=list('abcdef'))

        s2 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        self.assertEqual(s1.clip(lower=s2).to_pairs(),
                (('a', 2), ('b', 3), ('c', 2), ('d', 3), ('e', 8), ('f', 6))
                )

        self.assertEqual(s1.clip(upper=s2).to_pairs(),
                (('a', 0), ('b', 1), ('c', 0), ('d', -1), ('e', 4), ('f', 5))
                )

        s3 = Series((2, 3, 0), index=list('abc'))

        self.assertEqual(s1.clip(lower=s3).to_pairs(),
                (('a', 2), ('b', 3), ('c', 2), ('d', 3), ('e', 4), ('f', 5))
                )

        self.assertEqual(s1.clip(upper=s3).to_pairs(),
                (('a', 0), ('b', 1), ('c', 0), ('d', 3), ('e', 4), ('f', 5))
                )


    def test_series_clip_c(self) -> None:
        s1 = Series(range(6), index=list('abcdef'))

        with self.assertRaises(RuntimeError):
            _ = s1.clip(lower=(2, 5))


    #---------------------------------------------------------------------------
    def test_series_pickle_a(self) -> None:
        s1 = Series(range(6), index=list('abcdef'))
        s2 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))
        s3 = s2.astype(bool)


        for series in (s1, s2, s3):
            pbytes = pickle.dumps(series)
            series_new = pickle.loads(pbytes)
            for v in series: # iter labels
                # this compares series objects
                self.assertFalse(series_new.values.flags.writeable)
                self.assertEqual(series_new.loc[v], series.loc[v])


    #---------------------------------------------------------------------------

    def test_series_drop_loc_a(self) -> None:
        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        self.assertEqual(s1.drop.loc['d'].to_pairs(),
                (('a', 2), ('b', 3), ('c', 0), ('e', 8), ('f', 6)))

        self.assertEqual(s1.drop.loc['d':].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 2), ('b', 3), ('c', 0)))

        self.assertEqual(s1.drop.loc['d':'e'].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                (('a', 2), ('b', 3), ('c', 0), ('f', 6)))

        self.assertEqual(s1.drop.loc[s1 > 0].to_pairs(),
                (('c', 0), ('d', -1)))


    def test_series_drop_loc_b(self) -> None:
        s1 = Series((2, 3, 0, -1), index=list('abcd'))
        s2 = s1._drop_iloc((s1 < 1).values)
        self.assertEqual(s2.to_pairs(), (('a', 2), ('b', 3)))



    def test_series_drop_iloc_a(self) -> None:
        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        self.assertEqual(s1.drop.iloc[-1].to_pairs(),
                (('a', 2), ('b', 3), ('c', 0), ('d', -1), ('e', 8))
                )
        self.assertEqual(s1.drop.iloc[2:].to_pairs(),
                (('a', 2), ('b', 3)))

        self.assertEqual(s1.drop.iloc[[0, 3]].to_pairs(),
                (('b', 3), ('c', 0), ('e', 8), ('f', 6)))



    #---------------------------------------------------------------------------

    def test_series_head_a(self) -> None:
        s1 = Series(range(100), index=reversed(range(100)))
        self.assertEqual(s1.head().to_pairs(),
                ((99, 0), (98, 1), (97, 2), (96, 3), (95, 4)))
        self.assertEqual(s1.head(2).to_pairs(),
                ((99, 0), (98, 1)))

    #---------------------------------------------------------------------------
    def test_series_tail_a(self) -> None:
        s1 = Series(range(100), index=reversed(range(100)))

        self.assertEqual(s1.tail().to_pairs(),
                ((4, 95), (3, 96), (2, 97), (1, 98), (0, 99)))

        self.assertEqual(s1.tail(2).to_pairs(),
                ((1, 98), (0, 99)))

    #---------------------------------------------------------------------------
    def test_series_count_a(self) -> None:
        s1 = Series((2, 3, 0, np.nan, 8, 6), index=list('abcdef'))
        self.assertEqual(s1.count(), 5)

        s2 = Series((2, None, 0, np.nan, 8, 6), index=list('abcdef'))
        self.assertEqual(s2.count(), 4)

    def test_series_count_b(self) -> None:
        s1 = Series((2, 3, 0, np.nan, 8, 6), index=list('abcdef'))
        self.assertEqual(s1.count(skipna=False), 6)

    def test_series_count_c(self) -> None:
        s1 = Series((2, 3, 0, np.nan, 8, 6), index=list('abcdef'))
        self.assertEqual(s1.count(skipfalsy=True), 4)

    def test_series_count_d(self) -> None:
        s1 = Series((2, 6, 0, np.nan, 0, 6), index=list('abcdef'))
        self.assertEqual(s1.count(skipfalsy=True, unique=True), 2)

    def test_series_count_e(self) -> None:
        s1 = Series((2, 6, 0, np.nan, 0, 6), index=list('abcdef'))
        with self.assertRaises(RuntimeError):
            s1.count(skipfalsy=True, skipna=False)

    def test_series_count_f(self) -> None:
        s1 = Series((2, 6, 0, np.nan, 0, 6), index=list('abcdef'))
        self.assertEqual(s1.count(skipfalsy=True, unique=False), 3)

    def test_series_count_g(self) -> None:
        s1 = Series((2, 6, 0, np.nan, 0, 6), index=list('abcdef'))
        self.assertEqual(s1.count(unique=True), 3)

    def test_series_count_h(self) -> None:
        s1 = Series((2, 6, 0, np.nan, 0, 6), index=list('abcdef'))
        self.assertEqual(s1.count(skipna=True, unique=False), 5)

    def test_series_count_i(self) -> None:
        s1 = Series((2, 3, 8, 8, 6, None), index=list('abcdef'))
        self.assertEqual(s1.count(skipna=False, skipfalsy=False, unique=True), 5)

    #---------------------------------------------------------------------------
    def test_series_roll_a(self) -> None:
        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        self.assertEqual(s1.roll(2).to_pairs(),
                (('a', 8), ('b', 6), ('c', 2), ('d', 3), ('e', 0), ('f', -1))
                )

        self.assertEqual(s1.roll(-2).to_pairs(),
                (('a', 0), ('b', -1), ('c', 8), ('d', 6), ('e', 2), ('f', 3))
                )

        # if the roll is a noop, we reuse the same array
        self.assertEqual(s1.mloc, s1.roll(len(s1)).mloc)


    def test_series_roll_b(self) -> None:
        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        self.assertEqual(s1.roll(2, include_index=True).to_pairs(),
            (('e', 8), ('f', 6), ('a', 2), ('b', 3), ('c', 0), ('d', -1))
            )

        self.assertEqual(s1.roll(-2, include_index=True).to_pairs(),
            (('c', 0), ('d', -1), ('e', 8), ('f', 6), ('a', 2), ('b', 3))
            )


    def test_series_shift_a(self) -> None:
        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        # if the shift is a noop, we reuse the same array
        self.assertEqual(s1.mloc, s1.shift(0).mloc)

        # default fill is NaN
        self.assertEqual(s1.shift(4).dtype,
                np.dtype('float64')
                )

        # import ipdb; ipdb.set_trace()
        self.assertEqual(s1.shift(4, fill_value=None).to_pairs(),
                (('a', None), ('b', None), ('c', None), ('d', None), ('e', 2), ('f', 3))
                )

        self.assertEqual(s1.shift(-4, fill_value=None).to_pairs(),
                (('a', 8), ('b', 6), ('c', None), ('d', None), ('e', None), ('f', None))
                )

        self.assertEqual(
                s1.shift(6, fill_value=None).to_pairs(),
                (('a', None), ('b', None), ('c', None), ('d', None), ('e', None), ('f', None))
                )

        self.assertEqual(
                s1.shift(-6, fill_value=None).to_pairs(),
                (('a', None), ('b', None), ('c', None), ('d', None), ('e', None), ('f', None))
                )

    def test_series_shift_b(self) -> None:
        s1 = sf.Series([]).shift(1)
        self.assertEqual(len(s1), 0)


    #---------------------------------------------------------------------------
    def test_series_isin_a(self) -> None:

        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        self.assertEqual(s1.isin([]).to_pairs(),
            (('a', False), ('b', False), ('c', False), ('d', False), ('e', False), ('f', False))
            )

        self.assertEqual(s1.isin((-1, 8)).to_pairs(),
            (('a', False), ('b', False), ('c', False), ('d', True), ('e', True), ('f', False))
            )

        self.assertEqual(s1.isin(s1.values).to_pairs(),
            (('a', True), ('b', True), ('c', True), ('d', True), ('e', True), ('f', True))
            )


    def test_series_isin_b(self) -> None:

        s1 = Series(['a', 'b', 'c', 'd'])
        self.assertEqual(s1.isin(('b', 'c')).to_pairs(),
                ((0, False), (1, True), (2, True), (3, False)))

        self.assertEqual(s1.isin(('b', 'c', None)).to_pairs(),
                ((0, False), (1, True), (2, True), (3, False)))

        self.assertEqual(s1.isin(s1[[1, 2]].values).to_pairs(),
                ((0, False), (1, True), (2, True), (3, False)))

        self.assertEqual(s1.isin({'b', 'c'}).to_pairs(),
                ((0, False), (1, True), (2, True), (3, False)))


    def test_series_isin_c(self) -> None:

        s1 = Series(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'])

        self.assertEqual(s1.isin(('a', 'd')).to_pairs(),
                ((0, True), (1, False), (2, False), (3, True), (4, True), (5, False), (6, False), (7, True)))


    def test_series_isin_d(self) -> None:
        s1 = Series((1, 1), index=list('ab'))
        lookup = {2,3,4,5,6,7,8,9,10,11,12,13}
        # Checks an edge case where if a numpy `assume_unique` flag is incorrectly passed, it returns the wrong result
        result = s1.isin(lookup)
        self.assertEqual(result.to_pairs(),
                (('a', False), ('b', False)))


    #---------------------------------------------------------------------------

    def test_series_to_html_a(self) -> None:

        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        post = s1.to_html(config=DisplayConfig(type_show=False, type_color=False), style_config=None)
        html = '<table><tbody><tr><th>a</th><td>2</td></tr><tr><th>b</th><td>3</td></tr><tr><th>c</th><td>0</td></tr><tr><th>d</th><td>-1</td></tr><tr><th>e</th><td>8</td></tr><tr><th>f</th><td>6</td></tr></tbody></table>'
        self.assertEqual(post.strip(), html.strip())

        post = s1.to_html(config=DisplayConfig(type_show=False, type_color=False))
        html = '<table style="border-collapse:collapse;border-width:1px;border-color:#898b8e;border-style:solid"><tbody><tr><th style="background-color:#d1d2d4;font-weight:bold;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">a</th><td style="background-color:#ffffff;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">2</td></tr><tr><th style="background-color:#d1d2d4;font-weight:bold;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">b</th><td style="background-color:#f2f2f2;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">3</td></tr><tr><th style="background-color:#d1d2d4;font-weight:bold;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">c</th><td style="background-color:#ffffff;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">0</td></tr><tr><th style="background-color:#d1d2d4;font-weight:bold;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">d</th><td style="background-color:#f2f2f2;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">-1</td></tr><tr><th style="background-color:#d1d2d4;font-weight:bold;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">e</th><td style="background-color:#ffffff;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">8</td></tr><tr><th style="background-color:#d1d2d4;font-weight:bold;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">f</th><td style="background-color:#f2f2f2;font-weight:normal;padding:2px;font-size:14px;border-width:1px;border-color:#898b8e;border-style:solid;color:#2b2a2a">6</td></tr></tbody></table>'

        self.assertEqual(post.strip(), html.strip())



    def test_series_to_html_datatables_a(self) -> None:

        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))
        sio = StringIO()
        post = s1.to_html_datatables(sio, show=False)
        self.assertEqual(post, None)
        self.assertTrue(len(sio.read()) >= 1385)


    def test_series_to_html_datatables_b(self) -> None:

        s1 = Series((2, 3, 0, -1, 8, 6), index=list('abcdef'))

        with temp_file('.html', path=True) as fp:
            s1.to_html_datatables(fp, show=False)
            with open(fp) as file:
                data = file.read()
                self.assertTrue('SFTable' in data)
                self.assertTrue(len(data) > 800)




    def test_series_disply_a(self) -> None:

        s1 = Series((2, 3), index=list('ab'), name='alt', dtype=np.int64)

        match = tuple(s1.display(DisplayConfig(type_color=False)))
        self.assertEqual(
            match,
            (['<Series: alt>'], ['<Index>', ''], ['a', '2'], ['b', '3'], ['<<U1>', '<int64>'])
            )

        s2 = Series(('a', 'b'), index=Index(('x', 'y'), name='bar'), name='foo')

        match = tuple(s2.display(DisplayConfig(type_color=False)))

        self.assertEqual(
            match,
            (['<Series: foo>'], ['<Index: bar>', ''], ['x', 'a'], ['y', 'b'], ['<<U1>', '<<U1>'])
            )


    def test_series_to_frame_a(self) -> None:

        s1 = Series((2, 3), index=list('ab'), name='alt')

        f1 = s1.to_frame()

        self.assertTrue(f1.__class__ is Frame)
        self.assertEqual(f1.columns.values.tolist(), ['alt'])
        self.assertEqual(f1.to_pairs(0),
            (('alt', (('a', 2), ('b', 3))),))

        self.assertTrue(s1.mloc == f1.mloc.tolist()[0])

    def test_series_to_frame_b(self) -> None:

        s1 = Series((2, 3), index=list('ab'), name='alt')

        f1 = s1.to_frame_go()

        self.assertTrue(f1.__class__ is FrameGO)
        self.assertEqual(f1.columns.values.tolist(), ['alt'])
        self.assertEqual(f1.to_pairs(0),
            (('alt', (('a', 2), ('b', 3))),))

        self.assertTrue(s1.mloc == f1.mloc.tolist()[0])

    def test_series_to_frame_c(self) -> None:

        s1 = Series((2, 3, 4), index=list('abc'), name='alt')

        f2 = s1.to_frame(axis=0)
        self.assertEqual(f2.to_pairs(0),
            (('a', (('alt', 2),)), ('b', (('alt', 3),)), ('c', (('alt', 4),))))

    def test_series_to_frame_d(self) -> None:

        s1 = Series((2, 3, 4), index=list('abc'), name='alt')
        with self.assertRaises(NotImplementedError):
            s1.to_frame(axis=None)  # type: ignore


    def test_series_to_frame_go_a(self) -> None:
        a = sf.Series((1, 2, 3), name='a')
        f = a.to_frame_go(axis=0)
        f['b'] = 'b'

        self.assertEqual(f.to_pairs(0),
                ((0, (('a', 1),)), (1, (('a', 2),)), (2, (('a', 3),)), ('b', (('a', 'b'),)))
                )


    def test_series_from_concat_a(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series((10, 20), index=list('de'))
        s3 = Series((8, 6), index=list('fg'))

        s = Series.from_concat((s1, s2, s3))

        self.assertEqual(s.to_pairs(),
                (('a', 2), ('b', 3), ('c', 0), ('d', 10), ('e', 20), ('f', 8), ('g', 6))
                )

    def test_series_from_concat_b(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series(('10', '20'), index=list('de'))
        s3 = Series((8, 6), index=list('fg'))

        s = Series.from_concat((s1, s2, s3))

        self.assertEqual(s.to_pairs(),
                (('a', 2), ('b', 3), ('c', 0), ('d', '10'), ('e', '20'), ('f', 8), ('g', 6))
                )


    def test_series_from_concat_c(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series(('10', '20'), index=list('de'))
        s3 = Series((8, 6), index=(1, 2))

        s = Series.from_concat((s1, s2, s3))

        self.assertEqual(s.to_pairs(),
                (('a', 2), ('b', 3), ('c', 0), ('d', '10'), ('e', '20'), (1, 8), (2, 6))
                )

    def test_series_from_concat_d(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc')).relabel_level_add('i')
        s2 = Series(('10', '20', '100'), index=list('abc')).relabel_level_add('ii')

        s3 = Series.from_concat((s1, s2))

        self.assertEqual(s3.to_pairs(),
                ((('i', 'a'), 2), (('i', 'b'), 3), (('i', 'c'), 0), (('ii', 'a'), '10'), (('ii', 'b'), '20'), (('ii', 'c'), '100'))
                )

    def test_series_from_concat_e(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series((10, 20), index=list('de'))
        s3 = Series((8, 6), index=list('fg'))


        s = Series.from_concat((s1, s2, s3), index=IndexAutoFactory)

        self.assertEqual(s.to_pairs(),
                ((0, 2), (1, 3), (2, 0), (3, 10), (4, 20), (5, 8), (6, 6))
                )

    def test_series_from_concat_f(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series((10, 20), index=list('de'))
        s3 = Series((8, 6), index=list('fg'))

        s = Series.from_concat((s1, s2, s3), index=list('pqrstuv'))

        self.assertEqual(s.to_pairs(),
                (('p', 2), ('q', 3), ('r', 0), ('s', 10), ('t', 20), ('u', 8), ('v', 6))
                )

    def test_series_from_concat_g(self) -> None:

        s1 = Series.from_concat([])
        self.assertEqual((0,), s1.shape)

        s2 = Series.from_concat([], index=[])
        self.assertEqual((0,), s2.shape)
        self.assertEqual((0,), s2.index.shape)

        s3 = Series.from_concat([], name='s3')
        self.assertEqual((0,), s3.shape)
        self.assertEqual('s3', s3.name)

        s4 = Series.from_concat([], index=[], name='s4')
        self.assertEqual((0,), s4.shape)
        self.assertEqual((0,), s4.index.shape)
        self.assertEqual('s4', s4.name)


    def test_series_from_concat_h(self) -> None:
        s1 = Series((2, 3, 0,), index=Index(list('abc'), name='foo'))
        s2 = Series((10, 20), index=Index(list('de'), name='foo'))

        s3 = Series.from_concat((s1, s2))
        self.assertEqual(s3.index.name, 'foo')
        self.assertEqual(s3.to_pairs(),
                (('a', 2), ('b', 3), ('c', 0), ('d', 10), ('e', 20))
                )

    def test_series_from_concat_i(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'), name='a')
        s2 = Series((10, 20), index=list('de'), name='a')
        s3 = Series((8, 6), index=list('fg'), name='b')

        s = Series.from_concat((s1, s2, s3))
        self.assertEqual(s.name, None)

    #---------------------------------------------------------------------------

    def test_series_iter_group_a(self) -> None:

        s1 = Series((10, 4, 10, 4, 10),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        group = tuple(s1.iter_group(axis=0))

        self.assertEqual(group[0].to_pairs(),
                (('a', 10), ('c', 10), ('e', 10)))

        self.assertEqual(group[1].to_pairs(),
                (('b', 4), ('d', 4)))

        with self.assertRaises(AxisInvalid):
            tuple(s1.iter_group(axis=1))

        with self.assertRaises(TypeError):
            tuple(s1.iter_group('sdf')) #type: ignore

        with self.assertRaises(TypeError):
            tuple(s1.iter_group(foo='sdf')) #type: ignore


    #---------------------------------------------------------------------------
    def test_series_iter_group_index_a(self) -> None:

        s1 = Series((10, 3, 15, 21, 28),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        post = tuple(s1.iter_group_labels_items())
        self.assertTrue(len(post), len(s1))
        self.assertTrue(all(isinstance(x[1], Series) for x in post))

    def test_series_iter_group_index_b(self) -> None:

        colors = ('red', 'green')
        shapes = ('square', 'circle', 'triangle')
        s1 = sf.Series(range(6), index=sf.IndexHierarchy.from_product(shapes, colors))

        post = tuple(s1.iter_group_labels(depth_level=0))
        self.assertTrue(len(post), 3)

        self.assertEqual(s1.iter_group_labels(depth_level=0).apply(np.sum).to_pairs(),
                (('circle', 5), ('square', 1), ('triangle', 9))
                )

        self.assertEqual(s1.iter_group_labels(depth_level=1).apply(np.sum).to_pairs(),
                (('green', 9), ('red', 6))
                )

    def test_series_iter_group_index_c(self) -> None:

        colors = ('red', 'green')
        shapes = ('square', 'circle', 'triangle')
        textures = ('smooth', 'rough')

        s1 = sf.Series(range(12),
                index=sf.IndexHierarchy.from_product(shapes, colors, textures)
                )

        post = tuple(s1.iter_group_labels(depth_level=[0, 2]))
        self.assertTrue(len(post), 6)

        self.assertEqual(s1.iter_group_labels(depth_level=[0, 2]).apply(np.sum).to_pairs(),
                ((('circle', 'rough'), 12), (('circle', 'smooth'), 10), (('square', 'rough'), 4), (('square', 'smooth'), 2), (('triangle', 'rough'), 20), (('triangle', 'smooth'), 18))
                )

    #---------------------------------------------------------------------------

    def test_series_locmin_a(self) -> None:
        s1 = Series((2, 3, 0,), index=list('abc'))
        self.assertEqual(s1.loc_min(), 'c')
        self.assertEqual(s1.iloc_min(), 2)
        self.assertEqual(s1.loc_max(), 'b')
        self.assertEqual(s1.iloc_max(), 1)

    def test_series_locmin_b(self) -> None:
        s1 = Series((2, np.nan, 0, -1), index=list('abcd'))
        self.assertEqual(s1.loc_min(), 'd')
        self.assertEqual(s1.iloc_min(), 3)
        self.assertEqual(s1.loc_max(), 'a')
        self.assertEqual(s1.iloc_max(), 0)


    def test_series_locmin_c(self) -> None:
        s1 = Series((2, np.nan, 0,), index=list('abc'))

        with self.assertRaises(RuntimeError):
            s1.loc_min(skipna=False)

        with self.assertRaises(RuntimeError):
            s1.loc_max(skipna=False)

    #---------------------------------------------------------------------------
    def test_series_cov_a(self) -> None:

        s1 = Series((3, 34, 87, 145, 234, 543, 8234), index=tuple('abcdefg'))
        s2 = Series((3, 34, 87, 145, 234, 543, 8234), index=tuple('abcdefg'))
        self.assertAlmostEqualArray(s1.cov(s2), 9312581.904761903)

        s3 = Series((8234, 3, 34, 87, 145, 234, 543), index=tuple('gabcdef'))
        self.assertAlmostEqualArray(s1.cov(s3), 9312581.904761903)

    def test_series_cov_b(self) -> None:

        s1 = Series((3, 34, 87, 145, 234, 543, 8234), index=tuple('abcdefg'))
        s2 = np.array([3, 34, 87, 145, 234, 543, 8234])
        self.assertAlmostEqualArray(s1.cov(s2), 9312581.904761903)




    #---------------------------------------------------------------------------
    def test_series_iloc_searchsorted(self) -> None:
        s1 = Series((3, 34, 87, 145, 234, 543, 8234), index=tuple('abcdefg'))
        self.assertEqual(s1.iloc_searchsorted(88), 3)
        self.assertEqual(s1.iloc_searchsorted(88, side_left=False), 3)

        self.assertEqual(s1.iloc_searchsorted(87), 2)
        self.assertEqual(s1.iloc_searchsorted(87, side_left=False), 3)

        # import ipdb; ipdb.set_trace()
        self.assertEqual(s1.iloc_searchsorted([0, 123]).tolist(), [0, 3])
        self.assertEqual(s1.iloc_searchsorted([0, 6]).tolist(), [0, 1])
        self.assertEqual(s1.iloc_searchsorted([3, 8234]).tolist(), [0, 6])
        self.assertEqual(s1.iloc_searchsorted([3, 8234], side_left=False).tolist(), [1, 7])

    #---------------------------------------------------------------------------
    def test_series_loc_searchsorted_a(self) -> None:
        s1 = Series((3, 34, 87, 145, 234, 543, 8234), index=tuple('abcdefg'))
        self.assertEqual(s1.loc_searchsorted(88), 'd')
        self.assertEqual(s1.loc_searchsorted(88, side_left=False), 'd')

        self.assertEqual(s1.loc_searchsorted(87), 'c')
        self.assertEqual(s1.loc_searchsorted(87, side_left=False), 'd')

        self.assertEqual(s1.loc_searchsorted([0, 123]).tolist(), ['a', 'd'])
        self.assertEqual(s1.loc_searchsorted([0, 6]).tolist(), ['a', 'b'])
        self.assertEqual(s1.loc_searchsorted([3, 8234]).tolist(), ['a', 'g'])
        self.assertEqual(s1.loc_searchsorted([3, 8234],
                side_left=False,
                fill_value=None).tolist(),
                ['b', None])

        self.assertEqual(
                s1.loc_searchsorted([3, 8235, 3, 8235], fill_value=None).tolist(),
                ['a', None, 'a', None])
        self.assertEqual(
                s1.loc_searchsorted(8235, fill_value=None),
                None)

        self.assertEqual(s1.loc_searchsorted(8234), 'g')
        self.assertTrue(np.isnan(s1.loc_searchsorted(8235)))

    def test_series_loc_searchsorted_b(self) -> None:

        s1 = Series(range(10), index=IndexDate.from_date_range('2020-01-01', '2020-01-10'))

        self.assertEqual(s1.astype(float).loc_searchsorted(2.5).tolist(),
                datetime.date(2020, 1, 4))

        self.assertEqual(
                s1.astype(float).loc_searchsorted((2.5, 5.5, 2000), fill_value=None).tolist(),
                [datetime.date(2020, 1, 4), datetime.date(2020, 1, 7), None]
                )


    #---------------------------------------------------------------------------

    def test_series_from_concat_items_a(self) -> None:

        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series((2, np.nan, 0, -1), index=list('abcd'))

        s3 = Series.from_concat_items((('x', s1), ('y', s2)), name='foo')

        self.assertAlmostEqualItems(s3.to_pairs(),
                ((('x', 'a'), 2.0), (('x', 'b'), 3.0), (('x', 'c'), 0.0), (('y', 'a'), 2.0), (('y', 'b'), np.nan), (('y', 'c'), 0.0), (('y', 'd'), -1.0))
                )

        self.assertAlmostEqualItems(s3[HLoc[:, 'b']].to_pairs(),
                ((('x', 'b'), 3.0), (('y', 'b'), np.nan)))

        self.assertEqual(s3.name, 'foo')


    def test_series_from_concat_items_b(self) -> None:
        s1 = Series.from_concat_items([])

        self.assertEqual((0,), s1.shape)


    def test_series_from_concat_items_c(self) -> None:

        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series((2, np.nan, 0, -1), index=list('abcd'))


        s3 = Series.from_concat_items((('x', s1), ('y', s2)),
            name='foo',
            index_constructor=Index,
            )
        self.assertEqual(s3.fillna('').to_pairs(),
                ((('x', 'a'), 2.0), (('x', 'b'), 3.0), (('x', 'c'), 0.0), (('y', 'a'), 2.0), (('y', 'b'), ''), (('y', 'c'), 0.0), (('y', 'd'), -1.0)))

    def test_series_from_concat_items_d(self) -> None:

        s1 = Series((2, 3, 0,), index=list('abc'))
        s2 = Series((2, np.nan, 0, -1), index=list('abcd'))

        s3 = Series.from_concat_items((('x', s1), ('y', s2)),
                index_constructor=IndexDefaultFactory('bar'), #type: ignore
                )
        self.assertEqual(s3.index.name, 'bar')

    #---------------------------------------------------------------------------

    def test_series_axis_window_items_a(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        post = tuple(s1._axis_window_items(as_array=True, size=2, step=1, label_shift=0))

        # first window has second label, and first two values
        self.assertEqual(post[0][1].tolist(), [1, 2]) #type: ignore
        self.assertEqual(post[0][0], 'b')

        self.assertEqual(post[-1][1].tolist(), [19, 20]) #type: ignore
        self.assertEqual(post[-1][0], 't')


    def test_series_axis_window_items_b(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        post = tuple(s1._axis_window_items(as_array=True, size=2, step=1, label_shift=-1))

        # first window has first label, and first two values
        self.assertEqual(post[0][1].tolist(), [1, 2]) #type: ignore
        self.assertEqual(post[0][0], 'a')

        self.assertEqual(post[-1][1].tolist(), [19, 20]) #type: ignore
        self.assertEqual(post[-1][0], 's')



    def test_series_axis_window_items_c(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        # this is an expanding window anchored at the first index
        post = tuple(s1._axis_window_items(as_array=True, size=1, step=0, size_increment=1))

        self.assertEqual(post[0][0], 'a')
        self.assertEqual(post[0][1].tolist(), [1]) #type: ignore

        self.assertEqual(post[-1][0], 't')
        self.assertEqual(post[-1][1].tolist(), list(range(1, 21))) #type: ignore



    def test_series_axis_window_items_d(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        post = tuple(s1._axis_window_items(as_array=True, size=5, start_shift=-5, window_sized=False))

        self.assertEqual(post[0][0], 'a')
        self.assertEqual(post[0][1].tolist(), [1]) #type: ignore

        self.assertEqual(post[1][0], 'b')
        self.assertEqual(post[1][1].tolist(), [1, 2]) #type: ignore

        self.assertEqual(post[-1][0], 't')
        self.assertEqual(post[-1][1].tolist(), [16, 17, 18, 19, 20]) #type: ignore



    def test_series_axis_window_items_e(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        # start shift needs to be 1 less than window to go to start of window
        post = tuple(s1._axis_window_items(as_array=True, size=5, label_shift=-4, window_sized=False))

        self.assertEqual(post[0][0], 'a')
        self.assertEqual(post[0][1].tolist(), [1, 2, 3, 4, 5]) #type: ignore

        self.assertEqual(post[1][0], 'b')
        self.assertEqual(post[1][1].tolist(), [2, 3, 4, 5, 6]) #type: ignore

        self.assertEqual(post[-1][0], 't')
        self.assertEqual(post[-1][1].tolist(), [20]) #type: ignore



    def test_series_axis_window_items_f(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        # start shift needs to be 1 less than window to go to start of window
        post = tuple(s1._axis_window_items(as_array=True, size=5, label_shift=-4, window_sized=True))

        self.assertEqual(post[0][0], 'a')
        self.assertEqual(post[0][1].tolist(), [1, 2, 3, 4, 5]) #type: ignore

        self.assertEqual(post[1][0], 'b')
        self.assertEqual(post[1][1].tolist(), [2, 3, 4, 5, 6]) #type: ignore

        self.assertEqual(post[-1][0], 'p')
        self.assertEqual(post[-1][1].tolist(), [16, 17, 18, 19, 20]) #type: ignore


    def test_series_axis_window_items_g(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        with self.assertRaises(RuntimeError):
            tuple(s1._axis_window_items(as_array=True, size=0))

        with self.assertRaises(RuntimeError):
            tuple(s1._axis_window_items(size=2, as_array=True, step=-1))


    def test_series_axis_window_items_h(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        post = tuple(s1._axis_window_items(as_array=True, size=1))
        self.assertEqual(post[0][0], 'a')
        self.assertEqual(post[0][1].tolist(), [1]) #type: ignore

        self.assertEqual(post[-1][0], 't')
        self.assertEqual(post[-1][1].tolist(), [20]) #type: ignore



    def test_series_axis_window_items_i(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))
        # step equal to window size produces adaject windows
        post = tuple(s1._axis_window_items(as_array=True, size=3, step=3))

        self.assertEqual(post[0][0], 'c')
        self.assertEqual(post[0][1].tolist(), [1, 2, 3]) #type: ignore

        self.assertEqual(post[1][0], 'f')
        self.assertEqual(post[1][1].tolist(), [4, 5, 6]) #type: ignore

        self.assertEqual(post[-1][0], 'r')
        self.assertEqual(post[-1][1].tolist(), [16, 17, 18]) #type: ignore


    def test_series_axis_window_items_j(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))
        # adjacent windows with label on first value, keeping incomplete windows
        post = tuple(s1._axis_window_items(as_array=True, size=3, step=3, label_shift=-2, window_sized=False))

        self.assertEqual(post[0][0], 'a')
        self.assertEqual(post[0][1].tolist(), [1, 2, 3]) #type: ignore

        self.assertEqual(post[1][0], 'd')
        self.assertEqual(post[1][1].tolist(), [4, 5, 6]) #type: ignore

        self.assertEqual(post[-1][0], 's')
        self.assertEqual(post[-1][1].tolist(), [19, 20]) #type: ignore



    def test_series_axis_window_items_k(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))
        # adjacent windows with label on first value, keeping incomplete windows
        post = tuple(s1._axis_window_items(as_array=True, size=3, window_valid=lambda w: np.sum(w) % 2 == 1))

        self.assertEqual(post[0][0], 'd')
        self.assertEqual(post[0][1].tolist(), [2, 3, 4]) #type: ignore

        self.assertEqual(post[1][0], 'f')
        self.assertEqual(post[1][1].tolist(), [4, 5, 6]) #type: ignore

        self.assertEqual(post[-1][0], 't')
        self.assertEqual(post[-1][1].tolist(), [18, 19, 20]) #type: ignore



    def test_series_axis_window_items_m(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))
        # adjacent windows with label on first value, keeping incomplete windows
        weight = np.array([.25, .5, .5, .25])
        post = tuple(s1._axis_window_items(as_array=True, size=4, window_func=lambda a: a * weight))

        self.assertEqual(post[0][0], 'd')
        self.assertEqual(post[0][1].tolist(), [0.25, 1, 1.5, 1]) #type: ignore

        self.assertEqual(post[-1][0], 't')
        self.assertEqual(post[-1][1].tolist(), [4.25, 9, 9.5, 5]) #type: ignore

    #---------------------------------------------------------------------------

    def test_series_iter_window_array_a(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        self.assertEqual(
                tuple(tuple(a) for a in s1.iter_window_array(size=2)),
                ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20))
                )

    def test_series_iter_window_array_b(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))
        s2 = s1.iter_window_array(size=2).apply(np.mean)
        self.assertEqual(s2.to_pairs(),
                (('b', 1.5), ('c', 2.5), ('d', 3.5), ('e', 4.5), ('f', 5.5), ('g', 6.5), ('h', 7.5), ('i', 8.5), ('j', 9.5), ('k', 10.5), ('l', 11.5), ('m', 12.5), ('n', 13.5), ('o', 14.5), ('p', 15.5), ('q', 16.5), ('r', 17.5), ('s', 18.5), ('t', 19.5))
        )


    #---------------------------------------------------------------------------
    def test_series_iter_window_a(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        self.assertEqual(
                tuple(s.index.values.tolist() for s in s1.iter_window(size=2)), # type: ignore
                (['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e'], ['e', 'f'], ['f', 'g'], ['g', 'h'], ['h', 'i'], ['i', 'j'], ['j', 'k'], ['k', 'l'], ['l', 'm'], ['m', 'n'], ['n', 'o'], ['o', 'p'], ['p', 'q'], ['q', 'r'], ['r', 's'], ['s', 't'])
                )

        self.assertEqual(
            s1.iter_window(size=5, label_shift=-4, step=6, window_sized=False
                    ).apply(lambda s: len(s.index)).to_pairs(),
            (('a', 5), ('g', 5), ('m', 5), ('s', 2))
        )


    def test_series_iter_window_b(self) -> None:

        s1 = Series(range(10), index=self.get_letters(10))

        with self.assertRaises(TypeError):
            s1.iter_window() #type: ignore

        with self.assertRaises(TypeError):
            s1.iter_window(3) #type: ignore

        with self.assertRaises(TypeError):
            s1.iter_window(foo=3) #type: ignore

        self.assertEqual(
                tuple(x.to_pairs() for x in s1.iter_window(size=2, step=2)), #type: ignore
                ((('a', 0), ('b', 1)), (('c', 2), ('d', 3)), (('e', 4), ('f', 5)), (('g', 6), ('h', 7)), (('i', 8), ('j', 9)))
                )

    def test_series_iter_window_c(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20))

        self.assertEqual(
                tuple(w.tolist() for w in s1.iter_window_array( #type: ignore
                        size=7,
                        step=7,
                        window_sized=False,
                        label_shift=-6,
                        )),
                ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [15, 16, 17, 18, 19, 20])
                )


    def test_series_iter_window_d(self) -> None:
        post1 = sf.Series(range(12)).iter_window_array(
                size=5,
                start_shift=-10,
                window_sized=True).apply(np.mean)

        self.assertEqual(post1.to_pairs(),
                ((4, 2.0), (5, 3.0), (6, 4.0), (7, 5.0), (8, 6.0), (9, 7.0), (10, 8.0), (11, 9.0)))

        post2 = sf.Series(range(12)).iter_window_array(
                size=5,
                start_shift=0,
                window_sized=True).apply(np.mean)

        self.assertEqual(post2.to_pairs(),
                ((4, 2.0), (5, 3.0), (6, 4.0), (7, 5.0), (8, 6.0), (9, 7.0), (10, 8.0), (11, 9.0)))


    #---------------------------------------------------------------------------
    def test_series_bool_a(self) -> None:
        s1 = Series(range(1, 21), index=self.get_letters(20))
        with self.assertRaises(ValueError):
            bool(s1)


    #---------------------------------------------------------------------------
    def test_series_round_a(self) -> None:
        s1 = Series(np.arange(8) + .001)
        s2 = round(s1) #type: ignore

        self.assertEqual(s2.to_pairs(),
                ((0, 0.0), (1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0), (7, 7.0)))


    #---------------------------------------------------------------------------
    def test_series_str_capitalize_a(self) -> None:
        s1 = Series(('foo', 'bar'), index=('x', 'y'))
        s2 = s1.via_str.capitalize()

        self.assertEqual(s2.to_pairs(),
            (('x', 'Foo'), ('y', 'Bar'))
            )

        s3 = Series((20, 30), index=('x', 'y'))
        s4 = s3.via_str.capitalize()

        self.assertEqual(s4.to_pairs(),
            (('x', '20'), ('y', '30'))
            )

    def test_series_str_center_a(self) -> None:
        s1 = Series(('foo', 'bar'), index=('x', 'y'))
        s2 = s1.via_str.center(9, '-')

        self.assertEqual(s2.to_pairs(),
            (('x', '---foo---'), ('y', '---bar---'))
            )

        s3 = Series((20, 30), index=('x', 'y'))
        s4 = s3.via_str.center(4)

        self.assertEqual(s4.to_pairs(),
            (('x', ' 20 '), ('y', ' 30 '))
            )

    def test_series_str_encode_a(self) -> None:
        s1 = Series(('foo', 'bar'), index=('x', 'y'))

        s2 = s1.via_str.encode('ascii')

        self.assertEqual(s2.to_pairs(),
            (('x', b'foo'), ('y', b'bar'))
            )

    def test_series_str_decode_a(self) -> None:
        s1 = Series((b'foo', b'bar'), index=('x', 'y'))

        s2 = s1.via_str.decode('utf-8')
        self.assertEqual(s2.to_pairs(),
            (('x', 'foo'), ('y', 'bar'))
            )

    def test_series_str_len_a(self) -> None:
        s1 = Series((100, 4), index=('x', 'y'))
        s2 = s1.via_str.len()
        self.assertEqual(s2.to_pairs(),
                (('x', 3), ('y', 1)))


    def test_series_str_ljust_a(self) -> None:
        s1 = Series(('foo', 'bar'), index=('x', 'y'))
        s2 = s1.via_str.ljust(9, '-')

        self.assertEqual(s2.to_pairs(),
            (('x', 'foo------'), ('y', 'bar------'))
            )

        s3 = Series((20, 30), index=('x', 'y'))
        s4 = s3.via_str.ljust(4)

        self.assertEqual(s4.to_pairs(),
            (('x', '20  '), ('y', '30  '))
            )


    def test_series_str_replace_a(self) -> None:
        s1 = Series(('*foo*', '*bar*'), index=('x', 'y'))
        s2 = s1.via_str.replace('*', '!')

        self.assertEqual(s2.to_pairs(),
                (('x', '!foo!'), ('y', '!bar!')))


    def test_series_str_rjust_a(self) -> None:
        s1 = Series(('foo', 'bar'), index=('x', 'y'))
        s2 = s1.via_str.rjust(9, '-')

        self.assertEqual(s2.to_pairs(),
            (('x', '------foo'), ('y', '------bar'))
            )

        s3 = Series((20, 30), index=('x', 'y'))
        s4 = s3.via_str.rjust(4)

        self.assertEqual(s4.to_pairs(),
            (('x', '  20'), ('y', '  30'))
            )


    def test_series_str_rsplit_a(self) -> None:
        s1 = Series(('f*oo', 'b*ar'), index=('x', 'y'))
        s2 = s1.via_str.rsplit('*')

        self.assertEqual(s2.to_pairs(),
                (('x', ('f', 'oo')), ('y', ('b', 'ar'))))


    def test_series_str_rstrip_a(self) -> None:
        s1 = Series((' foo  ', ' bar  '), index=('x', 'y'))
        s2 = s1.via_str.rstrip()
        self.assertEqual(s2.to_pairs(),
                (('x', ' foo'), ('y', ' bar')))


    def test_series_str_split_a(self) -> None:
        s1 = Series(('f*oo', 'b*ar'), index=('x', 'y'))
        s2 = s1.via_str.split('*')

        self.assertEqual(s2.to_pairs(),
                (('x', ('f', 'oo')), ('y', ('b', 'ar'))))

    def test_series_str_strip_a(self) -> None:
        s1 = Series(('*foo*', '*bar*'), index=('x', 'y'))
        s2 = s1.via_str.strip('*')
        self.assertEqual(s2.to_pairs(),
                (('x', 'foo'), ('y', 'bar')))


    def test_series_str_swapcase_a(self) -> None:
        s1 = Series(('fOO', 'bAR'), index=('x', 'y'))
        s2 = s1.via_str.swapcase()
        self.assertEqual(s2.to_pairs(),
                (('x', 'Foo'), ('y', 'Bar')))

    def test_series_str_title_a(self) -> None:
        s1 = Series(('fOO', 'bAR'), index=('x', 'y'))
        s2 = s1.via_str.title()
        self.assertEqual(s2.to_pairs(),
                (('x', 'Foo'), ('y', 'Bar')))

    def test_series_str_upper_a(self) -> None:
        s1 = Series(('fOO', 'bAR'), index=('x', 'y'))
        s2 = s1.via_str.upper()
        self.assertEqual(s2.to_pairs(),
                (('x', 'FOO'), ('y', 'BAR')))


    def test_series_str_zfill_a(self) -> None:
        s1 = Series(('3', '40'), index=('x', 'y'))
        s2 = s1.via_str.zfill(4)
        self.assertEqual(s2.to_pairs(),
                (('x', '0003'), ('y', '0040')))

    #---------------------------------------------------------------------------
    def test_series_str_count_a(self) -> None:
        s1 = Series(('foo', 'foo foo bar'), index=('x', 'y'))
        s2 = s1.via_str.count('foo')
        self.assertEqual(s2.to_pairs(),
                (('x', 1), ('y', 2)))

    def test_series_str_endswith_a(self) -> None:
        s1 = Series(('foo', 'foo foo bar'), index=('x', 'y'))
        s2 = s1.via_str.endswith('bar')
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', True)))

    def test_series_str_endswith_b(self) -> None:
        s1 = Series(('foo', 'fall', 'funk'), index=('x', 'y', 'z'))
        s2 = s1.via_str.endswith(('oo', 'nk'))
        self.assertEqual(s2.to_pairs(),
                (('x', True), ('y', False), ('z', True)))


    def test_series_str_startswith_a(self) -> None:
        s1 = Series(('foo', 'foo foo bar'), index=('x', 'y'))
        s2 = s1.via_str.startswith('foo')
        self.assertEqual(s2.to_pairs(),
                (('x', True), ('y', True)))

    def test_series_str_startswith_b(self) -> None:
        s1 = Series(('foo', 'fall', 'funk'), index=('x', 'y', 'z'))
        s2 = s1.via_str.startswith(('fa', 'fo'))
        self.assertEqual(s2.to_pairs(),
                (('x', True), ('y', True), ('z', False)))

    def test_series_str_find_a(self) -> None:
        s1 = Series(('foo', 'bar foo bar'), index=('x', 'y'))
        s2 = s1.via_str.find('oo')
        self.assertEqual(s2.to_pairs(),
                (('x', 1), ('y', 5)))


    def test_series_str_index_a(self) -> None:
        s1 = Series(('foo', 'bar foo bar'), index=('x', 'y'))
        with self.assertRaises(ValueError):
            _ = s1.via_str.index('aaa')
        s2 = s1.via_str.index('oo')
        self.assertEqual(s2.to_pairs(),
                (('x', 1), ('y', 5)))


    def test_series_str_isalnum_a(self) -> None:
        s1 = Series(('foo', '3234', '@#$'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isalnum()
        self.assertEqual(s2.to_pairs(),
                (('x', True), ('y', True), ('z', False)))

    def test_series_str_isalpha_a(self) -> None:
        s1 = Series(('foo', '3234', '@#$'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isalpha()
        self.assertEqual(s2.to_pairs(),
                (('x', True), ('y', False), ('z', False)))

    def test_series_str_isdecimal_a(self) -> None:
        s1 = Series(('foo', '3234', '@#$'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isdecimal()
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', True), ('z', False)))

    def test_series_str_isdigit_a(self) -> None:
        s1 = Series(('foo', '3234', '@#$'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isdigit()
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', True), ('z', False)))

    def test_series_str_islower_a(self) -> None:
        s1 = Series(('foo', '3234', 'AAA'), index=('x', 'y', 'z'))
        s2 = s1.via_str.islower()
        self.assertEqual(s2.to_pairs(),
                (('x', True), ('y', False), ('z', False)))

    def test_series_str_isnumeric_a(self) -> None:
        s1 = Series(('foo', '3234', 'AAA'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isnumeric()
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', True), ('z', False)))

    def test_series_str_isspace_a(self) -> None:
        s1 = Series(('foo', '   ', 'AAA'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isspace()
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', True), ('z', False)))

    def test_series_str_istitle_a(self) -> None:
        s1 = Series(('foo', '   ', 'Aaa'), index=('x', 'y', 'z'))
        s2 = s1.via_str.istitle()
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', False), ('z', True)))

    def test_series_str_isupper_a(self) -> None:
        s1 = Series(('foo', '   ', 'AAA'), index=('x', 'y', 'z'))
        s2 = s1.via_str.isupper()
        self.assertEqual(s2.to_pairs(),
                (('x', False), ('y', False), ('z', True)))


    def test_series_str_rfind_a(self) -> None:
        s1 = Series(('foo', 'bar foo bar'), index=('x', 'y'))
        s2 = s1.via_str.rfind('oo')
        self.assertEqual(s2.to_pairs(),
                (('x', 1), ('y', 5)))


    def test_series_str_rindex_a(self) -> None:
        s1 = Series(('foo', 'bar foo bar'), index=('x', 'y'))
        with self.assertRaises(ValueError):
            _ = s1.via_str.rindex('aaa')
        s2 = s1.via_str.rindex('oo')
        self.assertEqual(s2.to_pairs(),
                (('x', 1), ('y', 5)))

    def test_series_str_lower_a(self) -> None:
        s1 = Series(('foO', 'AAA'), index=('x', 'y'))
        s2 = s1.via_str.lower()
        self.assertEqual(s2.to_pairs(),
                (('x', 'foo'), ('y', 'aaa')))

    def test_series_str_lstrip_a(self) -> None:
        s1 = Series(('  foo', ' aaa'), index=('x', 'y'))
        s2 = s1.via_str.lstrip()
        self.assertEqual(s2.to_pairs(),
                (('x', 'foo'), ('y', 'aaa')))

    def test_series_str_partition_a(self) -> None:
        s1 = Series(('f*oo', 'b*ar'), index=('x', 'y'))
        s2 = s1.via_str.partition('*')

        self.assertEqual(s2.to_pairs(),
                (('x', ('f', '*', 'oo')), ('y', ('b', '*', 'ar'))))

    def test_series_str_rpartition_a(self) -> None:
        s1 = Series(('f*o*o', 'b*a*r'), index=('x', 'y'))
        s2 = s1.via_str.rpartition('*')

        self.assertEqual(s2.to_pairs(),
                (('x', ('f*o', '*', 'o')), ('y', ('b*a', '*', 'r'))))


    def test_series_str_getitem_a(self) -> None:
        s1 = Series(["ab_asldkj", "cd_LKSJ", "df_foooooo"])
        self.assertEqual(s1.via_str[:2].to_pairs(),
                ((0, 'ab'), (1, 'cd'), (2, 'df'))
                )
        self.assertEqual(s1.via_str[0].to_pairs(),
                ((0, 'a'), (1, 'c'), (2, 'd'))
                )


    #---------------------------------------------------------------------------
    def test_series_via_dt_year_a(self) -> None:
        dt64 = np.datetime64

        s1 = Series(('2014', '2013'), index=('x', 'y'))

        with self.assertRaises(RuntimeError):
            _ = s1.via_dt.year

        s2 = Series((dt64('2014-02'), dt64('2013-11')), index=('x', 'y')).via_dt.year

        self.assertEqual(
                s2.to_pairs(),
                (('x', 2014), ('y', 2013))
                )


    def test_series_via_dt_day_a(self) -> None:
        dt64 = np.datetime64

        s1 = Series(('2014', '2013'), index=('x', 'y'))

        with self.assertRaises(RuntimeError):
            _ = s1.via_dt.day

        s2 = Series((dt64('2014-02'), dt64('2013-11')), index=('x', 'y'))

        with self.assertRaises(RuntimeError):
            _ = s2.via_dt.day

        s3 = Series((dt64('2014-02-12'), dt64('2013-11-28')), index=('x', 'y'))

        post = s3.via_dt.day
        self.assertEqual(post.to_pairs(),
                (('x', 12), ('y', 28))
                )

        def todt(date_str: str) -> datetime.date:
            return datetime.date(*(int(x) for x in date_str.split('-')))

        s4 = Series((todt('2014-02-12'), todt('2013-11-28')), index=('x', 'y'))

        post = s4.via_dt.day
        self.assertEqual(post.to_pairs(),
                (('x', 12), ('y', 28))
                )


    def test_series_via_dt_isoformat_a(self) -> None:

        s1 = Series(('2014-01-02T05:02', '2013-02-05T16:55'),
                index=('x', 'y'),
                dtype=np.datetime64
                )
        post = s1.via_dt.isoformat('*')
        self.assertEqual(post.to_pairs(),
                (('x', '2014-01-02*05:02:00'), ('y', '2013-02-05*16:55:00'))
                )

    #---------------------------------------------------------------------------
    def test_series_via_dt_weekday_a(self) -> None:

        s1 = Series(('2014-01-02T05:02', '2013-02-05T16:55'),
                index=('x', 'y'),
                dtype='datetime64[ns]'
                )
        self.assertEqual(s1.via_dt.weekday().to_pairs(),
                (('x', 3), ('y', 1)))

        # we do not permit nanosecond to got microsecond
        with self.assertRaises(RuntimeError):
            s1.via_dt.isoformat()

    def test_series_via_dt_weekday_b(self) -> None:
        index = IndexDate.from_date_range('0001-01-01', '1000-01-01')
        s1 = Series(range(len(index)), index=index)
        wd1 = s1.index.via_dt.weekday()

        s2 = Series(s1.index.values.astype(object))
        wd2 = s2.via_dt.weekday()

        wd3 = np.array([dt.weekday() for dt in s2.values], dtype=DTYPE_INT_DEFAULT)

        self.assertTrue((wd1 == wd3).all())
        self.assertTrue((wd2 == wd3).all())


    #---------------------------------------------------------------------------
    def test_series_via_dt_quarter_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')

        s1 = Series(index.values, index=index).via_dt.quarter()
        self.assertEqual(s1['2021-01-01'], 1)
        self.assertEqual(s1['2021-03-31'], 1)
        self.assertEqual(s1['2021-04-01'], 2)
        self.assertEqual(s1['2021-06-30'], 2)
        self.assertEqual(s1['2021-07-01'], 3)
        self.assertEqual(s1['2021-09-30'], 3)
        self.assertEqual(s1['2021-10-01'], 4)
        self.assertEqual(s1['2021-12-31'], 4)

        self.assertEqual((s1 == 1).sum(), 2888)
        self.assertEqual((s1 == 2).sum(), 2912)
        self.assertEqual((s1 == 3).sum(), 2944)
        self.assertEqual((s1 == 4).sum(), 2944)

    #---------------------------------------------------------------------------
    def test_series_via_dt_is_month_end_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values, index=index).via_dt.is_month_end()
        self.assertEqual(s1.sum(), 384)
        self.assertEqual(s1['2021-12-31'], True)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], False)
        self.assertEqual(s1['2021-01-31'], True)

    def test_series_via_dt_is_month_end_b(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values.astype(object), index=index).via_dt.is_month_end()
        self.assertEqual(s1.sum(), 384)
        self.assertEqual(s1['2021-12-31'], True)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], False)
        self.assertEqual(s1['2021-01-31'], True)

    #---------------------------------------------------------------------------

    def test_series_via_dt_is_month_start_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values, index=index).via_dt.is_month_start()
        self.assertEqual(s1.sum(), 384)
        self.assertEqual(s1['2021-12-31'], False)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], True)
        self.assertEqual(s1['2021-01-31'], False)

    def test_series_via_dt_is_month_start_b(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values.astype(object), index=index).via_dt.is_month_start()
        self.assertEqual(s1.sum(), 384)
        self.assertEqual(s1['2021-12-31'], False)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], True)
        self.assertEqual(s1['2021-01-31'], False)


    #---------------------------------------------------------------------------
    def test_series_via_dt_is_year_end_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values, index=index).via_dt.is_year_end()
        self.assertEqual(s1.sum(), 32)
        self.assertEqual(s1['2021-12-31'], True)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], False)
        self.assertEqual(s1['2021-01-31'], False)

    def test_series_via_dt_is_year_end_b(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values.astype(object), index=index).via_dt.is_year_end()
        self.assertEqual(s1.sum(), 32)
        self.assertEqual(s1['2021-12-31'], True)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], False)
        self.assertEqual(s1['2021-01-31'], False)

    #---------------------------------------------------------------------------

    def test_series_via_dt_is_year_start_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values, index=index).via_dt.is_year_start()
        self.assertEqual(s1.sum(), 32)
        self.assertEqual(s1['2021-12-31'], False)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], True)
        self.assertEqual(s1['2021-01-31'], False)

    def test_series_via_dt_is_year_start_b(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values.astype(object), index=index).via_dt.is_year_start()
        self.assertEqual(s1.sum(), 32)
        self.assertEqual(s1['2021-12-31'], False)
        self.assertEqual(s1['2021-12-30'], False)
        self.assertEqual(s1['2021-01-01'], True)
        self.assertEqual(s1['2021-01-31'], False)


    #---------------------------------------------------------------------------
    def test_series_via_dt_is_quarter_end_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values, index=index).via_dt.is_quarter_end()
        self.assertEqual(s1.sum(), 128)
        self.assertEqual(s1['2021-03-31'], True)
        self.assertEqual(s1['2021-06-29'], False)
        self.assertEqual(s1['2021-06-30'], True)
        self.assertEqual(s1['2021-09-30'], True)
        self.assertEqual(s1['2021-09-29'], False)
        self.assertEqual(s1['2021-12-31'], True)

    def test_series_via_dt_is_quarter_end_b(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values.astype(object), index=index).via_dt.is_quarter_end()
        self.assertEqual(s1.sum(), 128)
        self.assertEqual(s1['2021-03-31'], True)
        self.assertEqual(s1['2021-06-29'], False)
        self.assertEqual(s1['2021-06-30'], True)
        self.assertEqual(s1['2021-09-30'], True)
        self.assertEqual(s1['2021-09-29'], False)
        self.assertEqual(s1['2021-12-31'], True)


    #---------------------------------------------------------------------------
    def test_series_via_dt_is_quarter_start_a(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values, index=index).via_dt.is_quarter_start()
        self.assertEqual(s1.sum(), 128)
        self.assertEqual(s1['2021-01-01'], True)
        self.assertEqual(s1['2021-03-31'], False)
        self.assertEqual(s1['2021-04-01'], True)
        self.assertEqual(s1['2021-06-29'], False)
        self.assertEqual(s1['2021-06-30'], False)
        self.assertEqual(s1['2021-07-01'], True)
        self.assertEqual(s1['2021-09-30'], False)
        self.assertEqual(s1['2021-09-29'], False)
        self.assertEqual(s1['2021-10-01'], True)
        self.assertEqual(s1['2021-12-31'], False)

    def test_series_via_dt_is_quarter_start_b(self) -> None:
        index = IndexDate.from_date_range('1990-01-01', '2021-12-31')
        s1 = Series(index.values.astype(object), index=index).via_dt.is_quarter_start()
        self.assertEqual(s1.sum(), 128)
        self.assertEqual(s1['2021-01-01'], True)
        self.assertEqual(s1['2021-03-31'], False)
        self.assertEqual(s1['2021-04-01'], True)
        self.assertEqual(s1['2021-06-29'], False)
        self.assertEqual(s1['2021-06-30'], False)
        self.assertEqual(s1['2021-07-01'], True)
        self.assertEqual(s1['2021-09-30'], False)
        self.assertEqual(s1['2021-09-29'], False)
        self.assertEqual(s1['2021-10-01'], True)
        self.assertEqual(s1['2021-12-31'], False)


    #---------------------------------------------------------------------------
    def test_series_via_dt_hour_a(self) -> None:

        s1 = Series(('2014-01-02T05:02', '2013-02-05T16:55', '2020-11-30T23:55'),
                index=('x', 'y', 'z'),
                dtype='datetime64[ns]'
                )
        self.assertEqual(s1.via_dt.hour.to_pairs(),
                (('x', 5), ('y', 16), ('z', 23)))

        s2 = Series(('2014-01-02T05:02', '2013-02-05T16:55', '2020-11-30T23:55'),
                index=('x', 'y', 'z'),
                dtype='datetime64[s]'
                ).astype(object)
        self.assertEqual(s2.via_dt.hour.to_pairs(),
                (('x', 5), ('y', 16), ('z', 23)))

    def test_series_via_dt_minute_a(self) -> None:

        s1 = Series(('2014-01-02T05:02', '2013-02-05T16:55', '2020-11-30T23:51'),
                index=('x', 'y', 'z'),
                dtype='datetime64[ns]'
                )
        self.assertEqual(s1.via_dt.minute.to_pairs(),
                (('x', 2), ('y', 55), ('z', 51)))

        s2 = Series(('2014-01-02T05:02', '2013-02-05T16:55', '2020-11-30T23:51'),
                index=('x', 'y', 'z'),
                dtype='datetime64[s]'
                ).astype(object)
        self.assertEqual(s2.via_dt.minute.to_pairs(),
                (('x', 2), ('y', 55), ('z', 51)))


    def test_series_via_dt_second_a(self) -> None:

        s1 = Series(('2014-01-02T05:02:00', '2013-02-05T16:55:30', '2020-11-30T23:51:13'),
                index=('x', 'y', 'z'),
                dtype='datetime64[ns]'
                )
        self.assertEqual(s1.via_dt.second.to_pairs(),
                (('x', 0), ('y', 30), ('z', 13)))

        s2 = Series(('2014-01-02T05:02:00', '2013-02-05T16:55:30', '2020-11-30T23:51:13'),
                index=('x', 'y', 'z'),
                dtype='datetime64[s]'
                ).astype(object)
        self.assertEqual(s2.via_dt.second.to_pairs(),
                (('x', 0), ('y', 30), ('z', 13)))



    #---------------------------------------------------------------------------
    def test_series_via_dt_fromisoformat_a(self) -> None:
        s1 = Series(('2014-02-12', '2013-11-28'), index=('x', 'y'))
        post = s1.via_dt.fromisoformat()

        self.assertEqual(post.values.tolist(),
                [datetime.date(2014, 2, 12), datetime.date(2013, 11, 28)])

        with self.assertRaises(RuntimeError):
            _ = Series(('2014-02', '2013-11'), index=('x', 'y')).via_dt.fromisoformat()

        with self.assertRaises(RuntimeError):
            _ = Series(('2014', '2013'), index=('x', 'y')).via_dt.fromisoformat()

    def test_series_via_dt_fromisoformat_b(self) -> None:
        s1 = Series(('2014-02-12T05:03:20', '2013-11-28T23:45:34'), index=('x', 'y'))
        post = s1.via_dt.fromisoformat()
        self.assertEqual(post.values.tolist(),
                [datetime.datetime(2014, 2, 12, 5, 3, 20),
                datetime.datetime(2013, 11, 28, 23, 45, 34)])


    def test_series_via_dt_fromisoformat_c(self) -> None:
        s1 = Series(('2014-02-12', '2013-11-28'), index=('x', 'y'), dtype=object)
        post = s1.via_dt.fromisoformat()

        self.assertEqual(post.values.tolist(),
                [datetime.date(2014, 2, 12), datetime.date(2013, 11, 28)])

    def test_series_via_dt_strptime_a(self) -> None:
        s1 = Series(('12/2/2014', '11/28/2013'), index=('x', 'y'), dtype=object)
        post = s1.via_dt.strptime('%m/%d/%Y')


        self.assertEqual(post.values.tolist(),
                [datetime.datetime(2014, 12, 2, 0, 0),
                datetime.datetime(2013, 11, 28, 0, 0)])

    def test_series_via_dt_strpdate_a(self) -> None:
        s1 = Series(('12/2/2014', '11/28/2013'), index=('x', 'y'), dtype=object)
        post = s1.via_dt.strpdate('%m/%d/%Y')

        self.assertEqual(post.values.tolist(),
                [datetime.date(2014, 12, 2),
                datetime.date(2013, 11, 28)])


        # import ipdb; ipdb.set_trace()

    #---------------------------------------------------------------------------

    def test_series_equals_a(self) -> None:

        s1 = Series(range(1, 21), index=self.get_letters(20), dtype=np.int64)
        s2 = Series(range(1, 21), index=self.get_letters(20), dtype=np.int64)
        s3 = Series(range(1, 21), index=self.get_letters(20), dtype=np.int64, name='foo')
        s4 = Series(range(1, 21), index=self.get_letters(20), dtype=np.int32)
        s5 = Series(range(0, 20), index=self.get_letters(20), dtype=np.int64)
        s6 = Series(range(1, 21), dtype=np.int64)

        self.assertTrue(s1.equals(s1))
        self.assertTrue(s1.equals(s2, compare_class=True))
        self.assertTrue(s1.equals(s2, compare_class=False))

        self.assertFalse(s1.equals(s3, compare_name=True))
        self.assertTrue(s1.equals(s3, compare_name=False))

        self.assertFalse(s1.equals(s4, compare_dtype=True))
        self.assertTrue(s1.equals(s4, compare_dtype=False))

        self.assertFalse(s1.equals(s5))
        self.assertFalse(s1.equals(s6))

    def test_series_equals_b(self) -> None:

        ih1 = IndexHierarchy.from_product(('a', 'b'), range(10))
        ih2 = IndexHierarchy.from_product(('a', 'b'), range(10))
        ih3 = IndexHierarchy.from_product(('a', 'c'), range(10))

        s1 = Series(range(1, 21), index=ih1)
        s2 = Series(range(1, 21), index=ih2)
        s3 = Series(range(1, 21), index=ih3)

        self.assertTrue(s1.equals(s2))
        self.assertFalse(s1.equals(s3))


    def test_series_equals_c(self) -> None:

        s1 = Series((1, 2, 5, np.nan), index=self.get_letters(4))
        s2 = Series((1, 2, 5, np.nan), index=self.get_letters(4))

        self.assertTrue(s1.equals(s2))
        self.assertFalse(s1.equals(s2, skipna=False))

    def test_series_equals_d(self) -> None:

        s1 = Series((1, 2, 5), index=('a', 'b', np.nan))
        s2 = Series((1, 2, 5), index=('a', 'b', np.nan))

        self.assertTrue(s1.equals(s2))
        self.assertFalse(s1.equals(s2, skipna=False))


    def test_series_equals_e(self) -> None:

        s1 = Series((1, 2, 5), index=('a', 'b', 'c'))
        s2 = Series(('1', '2', '5'), index=('a', 'b', 'c'))

        self.assertFalse(s1.equals(s2, compare_dtype=False))

    def test_series_equals_f(self) -> None:

        s1 = Series((1, None, 5), index=('a', 'b', 'c'))
        s2 = Series((1, np.nan, 5), index=('a', 'b', 'c'))

        self.assertFalse(s1.equals(s2, compare_dtype=False))

    def test_series_equals_g(self) -> None:

        s1 = Series((1, 2, 5), index=('a', 'b', 'c'))
        s2 = Series((2, 5), index=('b', 'c'))

        a1 = s1.values

        self.assertFalse(s1.equals(a1, compare_class=True))
        self.assertFalse(s1.equals(a1, compare_class=False))
        self.assertFalse(s1.equals(s2))

    #---------------------------------------------------------------------------
    def test_series_enum_a(self) -> None:

        class Bar(str, Enum):
            a = 'a'
            b = 'b'
            c = 'c'

        s1 = sf.Series([Bar.a, Bar.b, Bar.c])

        self.assertEqual(s1.values.tolist(), [Bar.a, Bar.b, Bar.c])
        self.assertEqual(s1[1], Bar.b)
        self.assertEqual(s1[1:].values.tolist(), [Bar.b, Bar.c])


    def test_series_enum_b(self) -> None:

        class Bar(str, Enum):
            a = 'a'
            b = 'b'
            c = 'c'

        s1 = sf.Series(Bar, index=Bar)

        self.assertEqual(
                s1[Bar.b:].to_pairs(), #type: ignore
                ((Bar.b, Bar.b), (Bar.c, Bar.c))
                )


    def test_series_enum_c(self) -> None:
        # see: https://github.com/InvestmentSystems/static-frame/issues/239

        class Bar(str, Enum):
            a = 'a'
            b = 'b'
            c = 'c'

        s1 = sf.Series(Bar)

        # for str Enum, must compare to .value
        self.assertEqual((s1 == Bar.c).values.tolist(),
                [False, False, False])
        self.assertEqual((s1 == Bar.c.value).values.tolist(),
                [False, False, True])

        # comparisons to normal Enum's work as expected
        class Foo(Enum):
            a = 'a'
            b = 'b'
            c = 'c'

        s2 = sf.Series(Foo)
        self.assertEqual((s2 == Foo.c).values.tolist(),
                [False, False, True])
        self.assertEqual((s2 == Foo.c.value).values.tolist(),
                [False, False, False])



    #---------------------------------------------------------------------------

    def test_series_insert_a(self) -> None:

        s1 = Series((1, None, 5), index=('a', 'b', 'c'))
        s2 = Series((1, 3.4, 5), index=('d', 'e', 'f'))

        with self.assertRaises(NotImplementedError):
            _ = s1._insert(1, (3, 4), after=True) #type: ignore

        s3 = s1._insert(1, s2, after=False)

        self.assertEqual(s3.to_pairs(),
                (('a', 1), ('d', 1.0), ('e', 3.4), ('f', 5.0), ('b', None), ('c', 5))
                )

        s4 = s1._insert(2, s2, after=True)

        self.assertEqual(s4.to_pairs(),
                (('a', 1), ('b', None), ('c', 5), ('d', 1.0), ('e', 3.4), ('f', 5.0))
                )

    def test_series_insert_b(self) -> None:

        s1 = Series((1, None, 5), index=('a', 'b', 'c'))
        s2 = Series((1, 3.4, 5), index=('d', 'e', 'f'))

        s3 = s1.insert_before('c', s2)
        self.assertEqual(s3.to_pairs(),
                (('a', 1), ('b', None), ('d', 1.0), ('e', 3.4), ('f', 5.0), ('c', 5))
                )

        s4 = s1.insert_after('c', s2)
        self.assertEqual(s4.to_pairs(),
                (('a', 1), ('b', None), ('c', 5), ('d', 1.0), ('e', 3.4), ('f', 5.0))
                )


    def test_series_insert_c(self) -> None:

        s1 = Series((1, None, 5), index=('a', 'b', 'c'))
        s2 = Series((), index=())
        s3 = s1.insert_before('a', s2)
        self.assertEqual(id(s1), id(s3))

        with self.assertRaises(RuntimeError):
            s1.insert_before(slice('a', 'c'), s2)
        with self.assertRaises(RuntimeError):
            s1.insert_after(slice('a', 'c'), s2)


    def test_series_insert_d(self) -> None:

        s1 = Series((1, 2, 5), index=('a', 'b', 'c'))
        s2 = Series((1, 3), index=('d', 'e'))
        s3 = s1.insert_after(ILoc[-1], s2)
        self.assertEqual(s3.to_pairs(),
                (('a', 1), ('b', 2), ('c', 5), ('d', 1), ('e', 3))
                )

        s4 = s1.insert_before(ILoc[-1], s2)
        self.assertEqual(s4.to_pairs(),
                (('a', 1), ('b', 2), ('d', 1), ('e', 3), ('c', 5))
                )

    def test_series_insert_e(self) -> None:

        s1 = Series((1, 2, 5), index=('a', 'b', 'c'))
        s2 = Series((1, 3), index=('d', 'e'))
        with self.assertRaises(IndexError):
            _ = s1.insert_after(ILoc[-4], s2)
        with self.assertRaises(IndexError):
            _ = s1.insert_before(ILoc[-4], s2)
        with self.assertRaises(IndexError):
            _ = s1.insert_after(ILoc[3], s2)
        with self.assertRaises(IndexError):
            _ = s1.insert_before(ILoc[3], s2)

    #---------------------------------------------------------------------------

    def test_series_drop_a(self) -> None:
        s1 = Series(['a', 'b', 'c'],
            index=IndexHierarchy.from_labels([('X', 1), ('X', 2), ('Y', 1)]))

        s2 = s1.drop[np.array((True, False, True))]
        self.assertEqual(s2.to_pairs(),
                ((('X', 2), 'b'),))

    def test_series_drop_b(self) -> None:
        s1 = Series(['a', 'b', 'c'])

        s2 = s1.drop[np.array((True, False, True))]
        self.assertEqual(s2.to_pairs(),
                ((1, 'b'),))

    #---------------------------------------------------------------------------
    def test_series_from_overlay_a(self) -> None:
        s1 = Series((1, None, 5), index=('a', 'b', 'c'))
        s2 = Series((10, 30, -3), index=('a', 'b', 'c'))

        # NOTE: even though the result is all-integer, the dtype is int; this is the same with Pandas combine-first
        s3 = Series.from_overlay((s1, s2))
        self.assertEqual(s3.to_pairs(),
                (('a', 1), ('b', 30), ('c', 5)))
        self.assertEqual(s3.dtype.kind, 'O')


    def test_series_from_overlay_b(self) -> None:
        s1 = Series((1, np.nan, 5), index=('a', 'b', 'c'))
        s2 = Series((10, 30, -3, 3.1), index=('a', 'b', 'c', 'd'))
        s3 = Series((199, 230), index=('c', 'b'))

        s4 = Series.from_overlay((s1, s2, s3))
        self.assertEqual(s4.to_pairs(),
                (('a', 1.0), ('b', 30.0), ('c', 5.0), ('d', 3.1))
                )
        self.assertEqual(s4.dtype.kind, 'f')

        s5 = Series.from_overlay((s3, s1, s2))
        self.assertEqual(s5.to_pairs(),
                (('a', 1.0), ('b', 230.0), ('c', 199.0), ('d', 3.1))
                )
        self.assertEqual(s5.dtype.kind, 'f')


    def test_series_from_overlay_c(self) -> None:
        s1 = Series(('er', np.nan, 'pq'), index=('a', 'b', 'c'))
        s2 = Series(('io', 'tw', 'wf', None), index=('a', 'b', 'c', 'd'))
        s3 = Series(('mn', 'dd'), index=('e', 'd'))

        s4 = Series.from_overlay((s1, s2, s3), name='foo')

        self.assertEqual(s4.to_pairs(),
                (('a', 'er'), ('b', 'tw'), ('c', 'pq'), ('d', 'dd'), ('e', 'mn'))
                )
        self.assertEqual(s4.dtype.kind, 'O')
        self.assertEqual(s4.name, 'foo')

    def test_series_from_overlay_d(self) -> None:
        s1 = Series(('er', 'xx', 'pq'), index=('a', 'b', 'c'))
        s2 = Series(('io', 'tw', 'wf', 'ge'), index=('a', 'b', 'c', 'd'))
        s3 = Series(('mn', 'dd'), index=('e', 'd'))

        s4 = Series.from_overlay((s3, s1, s3))

        self.assertEqual(s4.to_pairs(),
                (('a', 'er'), ('b', 'xx'), ('c', 'pq'), ('d', 'dd'), ('e', 'mn'))
                )
        self.assertEqual(s4.dtype.kind, 'O')

    def test_series_from_overlay_e(self) -> None:
        s1 = Series((1, np.nan, 5), index=('a', 'b', 'c'))
        s2 = Series((10, 30, -3, 3.1), index=('a', 'b', 'c', 'd'))
        s3 = Series((199, 230), index=('c', 'b'))

        s4 = Series.from_overlay((s1, s2, s3), name='foo')
        self.assertEqual(s4.to_pairs(),
                (('a', 1.0), ('b', 30.0), ('c', 5.0), ('d', 3.1))
                )
        self.assertEqual(s4.dtype.kind, 'f')
        self.assertEqual(s4.name, 'foo')

        s5 = Series.from_overlay((s3, s1, s2))
        self.assertEqual(s5.to_pairs(),
                (('a', 1.0), ('b', 230.0), ('c', 199.0), ('d', 3.1))
                )
        self.assertEqual(s5.dtype.kind, 'f')


    def test_series_from_overlay_f(self) -> None:
        s1 = Series((1, np.nan, 5), index=('a', 'b', 'c'))
        s2 = Series((10, 30, -3, 3.1), index=('a', 'b', 'c', 'd'))
        s3 = Series((199, 230), index=('c', 'b'))

        s4 = Series.from_overlay((s1, s2, s3), union=False)
        self.assertEqual(s4.to_pairs(),
                (('b', 30.0), ('c', 5.0))
                )
        self.assertEqual(s4.dtype.kind, 'f')

    def test_series_from_overlay_g(self) -> None:
        s1 = Series((1, np.nan, np.nan), index=('a', 'b', 'c'))
        s2 = Series((10, 30, np.nan, 3.1), index=('a', 'b', 'c', 'd'))
        s3 = Series((199, np.nan), index=('c', 'b'))

        s4 = Series.from_overlay((s1, s2, s3), union=False)
        self.assertEqual(s4.to_pairs(),
                (('b', 30.0), ('c', 199.0))
                )
        self.assertEqual(s4.dtype.kind, 'f')


    def test_series_from_overlay_h(self) -> None:
        s1 = Series((1, np.nan, np.nan), index=('a', 'b', 'c'))
        s2 = Series((10, 30, 1.1, 3.1), index=('a', 'b', 'c', 'd'))
        s3 = Series((None, 'foo'), index=('c', 'b'))

        # last series does not force a type coercion
        s4 = Series.from_overlay((s1, s2, s3))

        self.assertEqual(s4.to_pairs(),
                (('a', 1.0), ('b', 30.0), ('c', 1.1), ('d', 3.1)))
        self.assertEqual(s4.dtype.kind, 'f')


    def test_series_from_overlay_i(self) -> None:
        s1 = Series(('2020', None, None, '1999'), index=('a', 'd', 'c', 'b'), dtype=np.datetime64)
        s2 = Series(('2020-05-03', None, '1983-09-21', '1830-05-02'), index=('a', 'b', 'c', 'd'), dtype=np.datetime64)

        s3 = Series(('1233-05-03', '1444-01-04', '1322-09-21', '2834-05-02'), index=('a', 'b', 'c', 'd'), dtype=np.datetime64)

        # year gets coerced to date going from s1 to s2
        s4 = Series.from_overlay((s1, s2, s3))
        self.assertEqual(s4.to_pairs(),
                (('a', np.datetime64('2020-01-01')), ('b', np.datetime64('1999-01-01')), ('c', np.datetime64('1983-09-21')), ('d', np.datetime64('1830-05-02')))
                )


    def test_series_from_overlay_j(self) -> None:

        s1 = Series((1, np.nan, np.nan),
                index=Index(('a', 'b', 'c'), name='foo'))
        s2 = Series((10, 30, 1.1, 3.1),
                index=Index(('a', 'b', 'c', 'd'), name='foo'))

        # last series does not force a type coercion
        s4 = Series.from_overlay((s1, s2))
        self.assertEqual(s4.index.name, 'foo')
        self.assertEqual(s4.to_pairs(),
                (('a', 1.0), ('b', 30.0), ('c', 1.1), ('d', 3.1)))
        self.assertEqual(s4.dtype.kind, 'f')

    def test_series_from_overlay_k(self) -> None:
        s1 = Series((1, np.nan, np.nan), index=('a', 'b', 'c'))
        s2 = Series((10, 30, np.nan, 3.1), index=('a', 'b', 'c', 'd'))
        s3 = Series((199, np.nan), index=('c', 'b'))

        s4 = Series.from_overlay((s1, s2, s3), index=('b', 'd'))
        self.assertEqual(s4.to_pairs(),
                (('b', 30.0), ('d', 3.1))
                )
        self.assertEqual(s4.dtype.kind, 'f')

    def test_series_from_overlay_l(self) -> None:
        s1 = Series((1, np.nan, 5), index=('a', 'b', 'c'))
        s2 = Series((10, 30, -3, 3.1), index=('a', 'b', 'c', 'd'), name=1)
        s3 = Series((199, 230), index=('c', 'b'))

        s4 = Series.from_overlay(s for s in (s1, s2, s3) if s.name != 1)
        self.assertEqual(s4.to_pairs(),
                (('a', 1.0), ('b', 230.0), ('c', 5.0))
                )


    #---------------------------------------------------------------------------
    def test_series_sample_a(self) -> None:
        s1 = Series(('io', 'tw', 'wf', 'ge'), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.sample(2, seed=8).to_pairs(),
                (('b', 'tw'), ('c', 'wf')))

    def test_series_sample_b(self) -> None:
        s1 = Series(range(4), index=IndexHierarchy.from_product(('a', 'b'), ('x', 'y')))
        self.assertEqual(s1.sample(3, seed=19).to_pairs(),
                ((('a', 'x'), 0), (('b', 'x'), 2), (('b', 'y'), 3))
                )

    #---------------------------------------------------------------------------
    def test_series_deepcopy_a(self) -> None:

        s1 = Series(['a', 'b', 'c'],
                index=IndexHierarchy.from_labels([('X', 1), ('X', 2), ('Y', 1)])
                )
        s2 = copy.deepcopy(s1)
        self.assertTrue(id(s1.values) != id(s2.values))
        self.assertTrue(id(s1.index.values_at_depth(1)) != id(s2.index.values_at_depth(1)))


    #---------------------------------------------------------------------------
    def test_series_std_a(self) -> None:

        s1 = Series(range(1, 6))
        self.assertEqual(round(s1.std(), 2), 1.41)
        self.assertEqual(round(s1.std(ddof=1), 2), 1.58)

    #---------------------------------------------------------------------------
    def test_series_var_a(self) -> None:

        s1 = Series(range(1, 6))
        self.assertEqual(round(s1.var(), 2), 2.0)
        self.assertEqual(round(s1.var(ddof=1), 2), 2.5)

    #---------------------------------------------------------------------------
    def test_series_via_fill_value_a(self) -> None:

        s1 = Series(range(3), index=tuple('abc'))
        s2 = Series(range(5), index=tuple('abcde'))

        self.assertEqual(
                (s1.via_fill_value(0) + s2).to_pairs(),
                (('a', 0), ('b', 2), ('c', 4), ('d', 3), ('e', 4))
                )

        self.assertEqual(
                (s1.via_fill_value(0) - s2).to_pairs(),
                (('a', 0), ('b', 0), ('c', 0), ('d', -3), ('e', -4))
                )

        self.assertEqual(
                (s1.via_fill_value(0) * s2).to_pairs(),
                (('a', 0), ('b', 1), ('c', 4), ('d', 0), ('e', 0))
                )

        self.assertEqual(
                round(s1.via_fill_value(1) / (s2 + 1), 1).to_pairs(),
                (('a', 0.0), ('b', 0.5), ('c', 0.7), ('d', 0.2), ('e', 0.2))
                )

        self.assertEqual(
                ((s1 * 2).via_fill_value(0) // s2).to_pairs(),
                (('a', 0), ('b', 2), ('c', 2), ('d', 0), ('e', 0))
                )

        self.assertEqual(
                (s1.via_fill_value(10) % s2).to_pairs(),
                (('a', 0), ('b', 0), ('c', 0), ('d', 1), ('e', 2))
                )

        self.assertEqual(
                (s1.via_fill_value(2) ** s2).to_pairs(),
                (('a', 1), ('b', 1), ('c', 4), ('d', 8), ('e', 16))
                )


        self.assertEqual(
                (s1.via_fill_value(-1) < s2).to_pairs(),
                (('a', False), ('b', False), ('c', False), ('d', True), ('e', True))
                )
        self.assertEqual(
                (s1.via_fill_value(0) <= s2).to_pairs(),
                (('a', True), ('b', True), ('c', True), ('d', True), ('e', True))
                )
        self.assertEqual(
                (s1.via_fill_value(4) == s2).to_pairs(),
                (('a', True), ('b', True), ('c', True), ('d', False), ('e', True))
                )
        self.assertEqual(
                (s1.via_fill_value(4) != s2).to_pairs(),
                (('a', False), ('b', False), ('c', False), ('d', True), ('e', False))
                )
        self.assertEqual(
                (s1.via_fill_value(10) > s2).to_pairs(),
                (('a', False), ('b', False), ('c', False), ('d', True), ('e', True))
                )
        self.assertEqual(
                (s1.via_fill_value(3) >= s2).to_pairs(),
                (('a', True), ('b', True), ('c', True), ('d', True), ('e', False))
                )


    def test_series_via_fill_value_b(self) -> None:

        s1 = Series((False, True, True), index=tuple('abc'))
        s2 = Series((True, True, False, True, False), index=tuple('abcde'))

        self.assertEqual(
                (s1.via_fill_value(False) >> s2).to_pairs(),
                (('a', 0), ('b', 0), ('c', 1), ('d', 0), ('e', 0))
                )

        self.assertEqual(
                (s1.via_fill_value(False) << s2).to_pairs(),
                (('a', 0), ('b', 2), ('c', 1), ('d', 0), ('e', 0))
                )


        self.assertEqual(
                (s1.via_fill_value(False) & s2).to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', False), ('e', False))
                )
        self.assertEqual(
                (s1.via_fill_value(False) | s2).to_pairs(),
               (('a', True), ('b', True), ('c', True), ('d', True), ('e', False))
                )
        self.assertEqual(
                (s1.via_fill_value(False) ^ s2).to_pairs(),
               (('a', True), ('b', False), ('c', True), ('d', True), ('e', False))
                )

    def test_series_via_fill_value_c(self) -> None:

        s1 = Series(range(3), index=tuple('abc'))

        self.assertEqual(
                (3 + s1.via_fill_value(0)).to_pairs(),
                (('a', 3), ('b', 4), ('c', 5))
                )

        self.assertEqual(
                (3 - s1.via_fill_value(0)).to_pairs(),
                (('a', 3), ('b', 2), ('c', 1))
                )

        self.assertEqual(
                (2 * s1.via_fill_value(0)).to_pairs(),
                (('a', 0), ('b', 2), ('c', 4))
                )

        s2 = s1 + 1

        self.assertEqual(
                round(10 / s2.via_fill_value(1), 1).to_pairs(),
                (('a', 10.0), ('b', 5.0), ('c', 3.3))
                )

        self.assertEqual(
                (10 // s2.via_fill_value(1)).to_pairs(),
                (('a', 10), ('b', 5), ('c', 3))
                )


    def test_series_via_fill_value_d(self) -> None:

        s1 = sf.Series(range(2), index=tuple('ab'))
        s2 = sf.Series(np.arange(1,4)*4, index=tuple('bcd'))
        s3 = sf.Series(np.arange(2,5)*100, index=tuple('cde'))


        s4 = (s1.via_fill_value(1) * s2).via_fill_value(0) + s3

        self.assertEqual(s4.to_pairs(),
                (('a', 0), ('b', 4), ('c', 208), ('d', 312), ('e', 400)))


    def test_series_via_fill_value_e(self) -> None:

        s1 = sf.Series(range(2), index=tuple('ab'))
        s2 = sf.Series(np.arange(1,4)*4, index=tuple('bcd'))

        with self.assertRaises(RuntimeError):
            s1.via_fill_value(0) + s2.via_fill_value(1)

    #---------------------------------------------------------------------------
    def test_series_via_re_search_a(self) -> None:

        s1 = sf.Series(('aaa', 'aab', 'cab'))

        s2 = s1.via_re('ab').search()
        self.assertEqual(s2.to_pairs(),
                ((0, False), (1, True), (2, True)))


        s3 = s1.via_re('AB', re.I).search()
        self.assertEqual(s3.to_pairs(),
                ((0, False), (1, True), (2, True)))


        s4 = s1.via_re('AB', re.I).search(0, 2)
        self.assertEqual(s4.to_pairs(),
                ((0, False), (1, False), (2, False)))

    def test_series_via_re_findall_a(self) -> None:
        s1 = sf.Series(('aaaaa', 'aabab', 'cabbaaaab'))

        s2 = s1.via_re('AB', re.I).findall()
        self.assertEqual(s2.to_pairs(),
                ((0, ()), (1, ('ab', 'ab')), (2, ('ab', 'ab')))
                )

    def test_series_via_re_split_a(self) -> None:
        s1 = sf.Series(('a.,aa.,aa', 'aa.,bab', 'cab.,baaa.,ab'))

        s2 = s1.via_re('.,').split()
        self.assertEqual(s2.to_pairs(),
                ((0, ('a', 'aa', 'aa')), (1, ('aa', 'bab')), (2, ('cab', 'baaa', 'ab')))
                )

    def test_series_via_re_sub_a(self) -> None:
        s1 = sf.Series(('a.,aa.,aa', 'aa.,bab', 'cab.,baaa.,ab'))
        s2 = s1.via_re('.,').sub('===')

        self.assertEqual(s2.to_pairs(),
                ((0, 'a===aa===aa'), (1, 'aa===bab'), (2, 'cab===baaa===ab'))
                )
        self.assertEqual(s2.dtype, np.dtype('<U15'))

    def test_series_via_re_subn_a(self) -> None:
        s1 = sf.Series(('a.,aa.,aa', 'aa.,bab', 'cab.,baaa.,ab'))
        s2 = s1.via_re('.,').subn('===')

        self.assertEqual(s2.to_pairs(),
                ((0, ('a===aa===aa', 2)), (1, ('aa===bab', 1)), (2, ('cab===baaa===ab', 2)))
                )

    def test_series_via_re_match_a(self) -> None:
        s1 = sf.Series(('aaaaaa', 'aabab', 'cabbaaaab'))
        s2 = s1.via_re('aa').match()
        self.assertEqual(s2.to_pairs(),
                ((0, True), (1, True), (2, False))
                )

        s3 = s1.via_re('aa').match(pos=4)
        self.assertEqual(s3.to_pairs(),
                ((0, True), (1, False), (2, True))
                )

        s4 = s1.via_re('aa').match(pos=4, endpos=1)
        self.assertEqual(s4.to_pairs(),
                ((0, False), (1, False), (2, False))
                )

    def test_series_via_re_fullmatch_a(self) -> None:
        s1 = sf.Series(('aaaaaa', 'aabab', 'cabbaaaab'))
        s2 = s1.via_re('aa').fullmatch()
        self.assertEqual(s2.to_pairs(),
                ((0, False), (1, False), (2, False))
                )

        s3 = s1.via_re('aa').fullmatch(pos=4)
        self.assertEqual(s3.to_pairs(),
                ((0, True), (1, False), (2, False))
                )

        s4 = s1.via_re('aa').fullmatch(pos=4, endpos=6)
        self.assertEqual(s4.to_pairs(),
                ((0, True), (1, False), (2, True))
                )



    #---------------------------------------------------------------------------
    def test_series_rank_ordinal_a(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_ordinal()
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 5, 2, 0, 8, 1, 9, 3, 6, 7])

    def test_series_rank_ordinal_b(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_ordinal(fill_value=-1)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 5, 2, 0, 7, 1, 8, 3, -1, 6])

    def test_series_rank_ordinal_c(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_ordinal(skipna=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 5, 2, 0, 7, 1, 8, 3, 9, 6])

    def test_series_rank_ordinal_d(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_ordinal(ascending=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [5, 4, 7, 9, 1, 8, 0, 6, 3, 2])




    def test_series_rank_dense_a(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_dense()
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [3, 4, 2, 0, 5, 1, 5, 2, 4, 4])

    def test_series_rank_dense_b(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_dense(fill_value=-1)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [3, 4, 2, 0, 5, 1, 5, 2, -1, 4])

    def test_series_rank_dense_c(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_dense(skipna=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [3, 4, 2, 0, 5, 1, 5, 2, 6, 4])

    def test_series_rank_dense_d(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_dense(ascending=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [2, 1, 3, 5, 0, 4, 0, 3, 1, 1])




    def test_series_rank_min_a(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_min()
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 5, 2, 0, 8, 1, 8, 2, 5, 5])

    def test_series_rank_min_b(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_min(fill_value=-1)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 5, 2, 0, 7, 1, 7, 2, -1, 5])

    def test_series_rank_min_c(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_min(skipna=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 5, 2, 0, 7, 1, 7, 2, 9, 5])

    def test_series_rank_min_d(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_min(ascending=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [5, 2, 6, 9, 0, 8, 0, 6, 2, 2])




    def test_series_rank_max_a(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_max()
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 7, 3, 0, 9, 1, 9, 3, 7, 7])

    def test_series_rank_max_b(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_max(fill_value=-1)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 6, 3, 0, 8, 1, 8, 3, -1, 6])

    def test_series_rank_max_c(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_max(skipna=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 6, 3, 0, 8, 1, 8, 3, 9, 6])

    def test_series_rank_max_d(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_max(ascending=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4, 3, 6, 8, 0, 7, 0, 6, 3, 3])



    def test_series_rank_mean_a(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_mean()
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4.0, 6.0, 2.5, 0.0, 8.5, 1.0, 8.5, 2.5, 6.0, 6.0])

    def test_series_rank_mean_b(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_mean(fill_value=-1)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4.0, 5.5, 2.5, 0.0, 7.5, 1.0, 7.5, 2.5, -1.0, 5.5])

    def test_series_rank_mean_c(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, np.nan, 15], name='foo')
        s2 = s1.rank_mean(skipna=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4.0, 5.5, 2.5, 0.0, 7.5, 1.0, 7.5, 2.5, 9.0, 5.5])

    def test_series_rank_mean_d(self) -> None:

        s1 = sf.Series([8, 15, 7, 2, 20, 4, 20, 7, 15, 15], name='foo')
        s2 = s1.rank_mean(ascending=False)
        self.assertEqual(s2.name, 'foo')
        self.assertEqual(s2.values.tolist(), [4.5, 2.5, 6.0, 8.5, 0.0, 7.5, 0.0, 6.0, 2.5, 2.5])


    #---------------------------------------------------------------------------
    def test_series_isfalsy_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.isfalsy().to_pairs(),
            (('a', False), ('b', False), ('c', False), ('d', True))
            )

        s2 = Series(('a', 'b', '', 'c'), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s2.isfalsy().to_pairs(),
            (('a', False), ('b', False), ('c', True), ('d', False))
            )

        s3 = Series((0, -2, 0, 2), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s3.isfalsy().to_pairs(),
            (('a', True), ('b', False), ('c', True), ('d', False))
            )

        s4 = Series(('', False, 0, np.nan), dtype=object, index=('a', 'b', 'c', 'd'))
        self.assertEqual(s4.isfalsy().to_pairs(),
            (('a', True), ('b', True), ('c', True), ('d', True))
            )

    def test_series_notfalsy_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.notfalsy().to_pairs(),
            (('a', True), ('b', True), ('c', True), ('d', False))
            )

        s2 = Series(('a', 'b', '', 'c'), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s2.notfalsy().to_pairs(),
            (('a', True), ('b', True), ('c', False), ('d', True))
            )

        s3 = Series((0, -2, 0, 2), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s3.notfalsy().to_pairs(),
            (('a', False), ('b', True), ('c', False), ('d', True))
            )

        s4 = Series(('', False, 0, np.nan), dtype=object, index=('a', 'b', 'c', 'd'))
        self.assertEqual(s4.notfalsy().to_pairs(),
            (('a', False), ('b', False), ('c', False), ('d', False))
            )

    #---------------------------------------------------------------------------
    def test_series_dropfalsy_a(self) -> None:

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.dropfalsy().to_pairs(),
            (('a', 234.3), ('b', 3.2), ('c', 6.4))
            )

        s2 = Series(('a', 'b', '', 'c'), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s2.dropfalsy().to_pairs(),
            (('a', 'a'), ('b', 'b'), ('d', 'c'))
            )

        s3 = Series((0, -2, 0, 2), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s3.dropfalsy().to_pairs(),
            (('b', -2), ('d', 2))
            )

        s4 = Series(('', False, 0, np.nan), dtype=object, index=('a', 'b', 'c', 'd'))
        self.assertEqual(s4.dropfalsy().to_pairs(), ())

    def test_series_dropfalsy_b(self) -> None:

        s1 = Series((4, 2, 5), index=('a', 'b', 'c'))
        self.assertEqual(s1.dropfalsy().to_pairs(),
            (('a', 4), ('b', 2), ('c', 5))
            )


    #---------------------------------------------------------------------------
    def test_series_fillfalsy_a(self) -> None:

        s1 = Series(('a', 'b', ''), index=('a', 'b', 'c'))
        self.assertEqual(s1.fillfalsy('x').to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'x')),
                )

        s2 = Series(('a', 'b', 0), index=('a', 'b', 'c'), dtype=object)
        self.assertEqual(s2.fillfalsy('x').to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'x')),
                )

        s3 = Series(('a', None, ''), index=('a', 'b', 'c'), dtype=object)
        self.assertEqual(s3.fillfalsy('x').to_pairs(),
                (('a', 'a'), ('b', 'x'), ('c', 'x')),
                )

    def test_series_fillfalsy_b(self) -> None:

        s1 = Series(('a', None, ''), index=('a', 'b', 'c'), dtype=object)
        self.assertEqual(s1.fillfalsy(Series.from_dict({'c':20, 'b':30})).to_pairs(),
                (('a', 'a'), ('b', 30), ('c', 20)),
                )

if __name__ == '__main__':
    unittest.main()
