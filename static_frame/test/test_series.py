
from itertools import zip_longest
from itertools import combinations
import unittest
from collections import OrderedDict
from io import StringIO
import string
import hashlib

import numpy as np

from static_frame.test.test_case import TestCase

import static_frame as sf
# assuming located in the same directory
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import Frame
from static_frame import FrameGO
from static_frame import TypeBlocks
from static_frame import Display
from static_frame import mloc
from static_frame import DisplayConfig
from static_frame import IndexHierarchy
from static_frame import IndexHierarchyGO

from static_frame.core.util import _isna
from static_frame.core.util import _resolve_dtype
from static_frame.core.util import _resolve_dtype_iter
from static_frame.core.util import _array_to_duplicated
from static_frame.core.util import _array_set_ufunc_many


from static_frame.core.operator_delegate import _all
from static_frame.core.operator_delegate import _any
from static_frame.core.operator_delegate import _nanall
from static_frame.core.operator_delegate import _nanany

nan = np.nan

LONG_SAMPLE_STR = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

# TODO:
# test Series.clip

class TestUnit(TestCase):

    #---------------------------------------------------------------------------
    # test series

    def test_series_init_a(self):
        s1 = Series(np.nan, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s1.dtype == float)
        self.assertTrue(len(s1) == 4)

        s2 = Series(False, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s2.dtype == bool)
        self.assertTrue(len(s2) == 4)

        s3 = Series(None, index=('a', 'b', 'c', 'd'))

        self.assertTrue(s3.dtype == object)
        self.assertTrue(len(s3) == 4)


    def test_series_init_b(self):
        s1 = Series(['a', 'b', 'c', 'd'], index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')))

        # testing direct specification of string type
        s2 = Series(['a', 'b', 'c', 'd'], index=('a', 'b', 'c', 'd'), dtype=str)
        self.assertEqual(s2.to_pairs(),
                (('a', 'a'), ('b', 'b'), ('c', 'c'), ('d', 'd')))

    def test_series_init_c(self):
        # these test get different results in Pyhthon 3.6
        # s1 = Series(dict(a=1, b=4), dtype=int)
        # self.assertEqual(s1.to_pairs(),
        #         (('a', 1), ('b', 4)))

        # s1 = Series(dict(b=4, a=1), dtype=int)
        # self.assertEqual(s1.to_pairs(),
        #         (('a', 1), ('b', 4)))

        s1 = Series(OrderedDict([('b', 4), ('a', 1)]), dtype=int)
        self.assertEqual(s1.to_pairs(),
                (('b', 4), ('a', 1)))

    def test_series_init_d(self):
        # single element, when the element is a string
        s1 = Series('abc', index=range(4))
        self.assertEqual(s1.to_pairs(),
                ((0, 'abc'), (1, 'abc'), (2, 'abc'), (3, 'abc')))

        # this is an array with shape == (), or a single element
        s2 = Series(np.array('abc'), index=range(4))
        self.assertEqual(s2.to_pairs(),
                ((0, 'abc'), (1, 'abc'), (2, 'abc'), (3, 'abc')))

        # single element, generator index
        s3 = Series(None, index=(x * 10 for x in (1,2,3)))
        self.assertEqual(s3.to_pairs(),
                ((10, None), (20, None), (30, None))
                )

    def test_series_init_e(self):
        s1 = Series(dict(a=1, b=2, c=np.nan, d=None), dtype=object)
        self.assertEqual(s1.to_pairs(),
                (('a', 1), ('b', 2), ('c', nan), ('d', None))
                )
        with self.assertRaises(ValueError):
            s1.values[1] = 23


    def test_series_slice_a(self):
        # create a series from a single value
        # s0 = Series(3, index=('a',))

        # generator based construction of values and index
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        # self.assertEqual(s1['b'], 1)
        # self.assertEqual(s1['d'], 3)

        s2 = s1['a':'c'] # with Pandas this is inclusive
        self.assertEqual(s2.values.tolist(), [0, 1, 2])
        self.assertTrue(s2['b'] == s1['b'])

        s3 = s1['c':]
        self.assertEqual(s3.values.tolist(), [2, 3])
        self.assertTrue(s3['d'] == s1['d'])

        self.assertEqual(s1['b':'d'].values.tolist(), [1, 2, 3])

        self.assertEqual(s1[['a', 'c']].values.tolist(), [0, 2])


    def test_series_keys_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1.keys()), ['a', 'b', 'c', 'd'])

    def test_series_iter_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1), ['a', 'b', 'c', 'd'])

    def test_series_items_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(list(s1.items()), [('a', 0), ('b', 1), ('c', 2), ('d', 3)])


    def test_series_intersection_a(self):
        # create a series from a single value
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s3 = s1['c':]
        self.assertEqual(s1.index.intersection(s3.index).values.tolist(),
            ['c', 'd'])


    def test_series_intersection_b(self):
        # create a series from a single value
        idxa = IndexGO(('a', 'b', 'c'))
        idxb = IndexGO(('b', 'c', 'd'))

        self.assertEqual(idxa.intersection(idxb).values.tolist(),
            ['b', 'c'])

        self.assertEqual(idxa.union(idxb).values.tolist(),
            ['a', 'b', 'c', 'd'])




    def test_series_binary_operator_a(self):
        '''Test binary operators where one operand is a numeric.
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list((s1 * 3).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual(list((s1 / .5).items()),
                [('a', 0.0), ('b', 2.0), ('c', 4.0), ('d', 6.0)])

        self.assertEqual(list((s1 ** 3).items()),
                [('a', 0), ('b', 1), ('c', 8), ('d', 27)])


    def test_series_binary_operator_b(self):
        '''Test binary operators with Series of same index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list((s1 + s2).items()),
                [('a', 0), ('b', 3), ('c', 6), ('d', 9)])

        self.assertEqual(list((s1 * s2).items()),
                [('a', 0), ('b', 2), ('c', 8), ('d', 18)])


    def test_series_binary_operator_c(self):
        '''Test binary operators with Series of different index
        '''
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = Series((x * 2 for x in range(4)), index=('c', 'd', 'e', 'f'))

        self.assertAlmostEqualItems(list((s1 * s2).items()),
                [('a', nan), ('b', nan), ('c', 0), ('d', 6), ('e', nan), ('f', nan)]
                )


    def test_series_binary_operator_d(self):
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


    def test_series_reindex_a(self):
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


    def test_series_reindex_b(self):
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



    def test_series_reindex_c(self):
        s1 = Series(('a', 'b', 'c', 'd'), index=((0, x) for x in range(4)))
        self.assertEqual(s1.loc[(0, 2)], 'c')

        self.assertEqual(
                s1.reindex(((0, 1), (0, 3), (4,5)), fill_value=None).to_pairs(),
                (((0, 1), 'b'), ((0, 3), 'd'), ((4, 5), None)))


        # s2 = s1.reindex(('c', 'd', 'a'))
        # self.assertEqual(list(s2.items()), [('c', 2), ('d', 3), ('a', 0)])

        # s3 = s1.reindex(['a','b'])
        # self.assertEqual(list(s3.items()), [('a', 0), ('b', 1)])




    def test_series_isnull_a(self):

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



    def test_series_isnull_b(self):

        # NOTE: this is a problematic case as it as a string with numerics and None
        s1 = Series((234.3, 'a', None, 6.4, np.nan), index=('a', 'b', 'c', 'd', 'e'))

        self.assertEqual(list(s1.isna().items()),
                [('a', False), ('b', False), ('c', True), ('d', False), ('e', True)]
                )

    def test_series_notnull(self):

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


    def test_series_dropna(self):

        s1 = Series((234.3, 3.2, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s2 = Series((234.3, None, 6.4, np.nan), index=('a', 'b', 'c', 'd'))
        s3 = Series((234.3, 5, 6.4, -234.3), index=('a', 'b', 'c', 'd'))
        s4 = Series((234.3, None, None, None), index=('a', 'b', 'c', 'd'))
        s5 = Series(('p', 'q', 'e', 'g'), index=('a', 'b', 'c', 'd'))
        s6 = Series((False, True, False, True), index=('a', 'b', 'c', 'd'))


        self.assertEqual(list(s2.dropna().items()),
                [('a', 234.3), ('c', 6.4)])


    def test_series_fillna_a(self):

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


    def test_series_from_pairs_a(self):

        def gen():
            r1 = range(10)
            r2 = iter(range(10, 20))
            for x in r1:
                yield x, next(r2)

        s1 = Series.from_items(gen())
        self.assertEqual(s1.loc[7:9].values.tolist(), [17, 18, 19])

        # NOTE: ordere here is unstable until python 3.6
        s2 = Series.from_items(dict(a=30, b=40, c=50).items())
        self.assertEqual(s2['c'], 50)
        self.assertEqual(s2['b'], 40)
        self.assertEqual(s2['a'], 30)


    def test_series_contains_a(self):

        s1 = Series.from_items(zip(('a', 'b', 'c'), (10, 20, 30)))
        self.assertTrue('b' in s1)
        self.assertTrue('c' in s1)
        self.assertTrue('a' in s1)

        self.assertFalse('d' in s1)
        self.assertFalse('' in s1)


    def test_series_sum_a(self):

        s1 = Series.from_items(zip(('a', 'b', 'c'), (10, 20, 30)))
        self.assertEqual(s1.sum(), 60)

        s1 = Series.from_items(zip(('a', 'b', 'c', 'd'), (10, 20, 30, np.nan)))
        self.assertEqual(s1.sum(), 60)

        s1 = Series.from_items(zip(('a', 'b', 'c', 'd'), (10, 20, 30, None)))
        self.assertEqual(s1.sum(), 60)


    def test_series_sum_b(self):
        s1 = Series(list('abc'), dtype=object)
        self.assertEqual(s1.sum(), 'abc')
        # get the same result from character arrays
        s2 = sf.Series(list('abc'))
        self.assertEqual(s2.sum(), 'abc')



    def test_series_mask_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(
                s1.mask.loc[['b', 'd']].values.tolist(),
                [False, True, False, True])
        self.assertEqual(s1.mask.iloc[1:].values.tolist(),
                [False, True, True, True])

        self.assertEqual(s1.masked_array.loc[['b', 'd']].sum(), 2)
        self.assertEqual(s1.masked_array.loc[['a', 'b']].sum(), 5)



    def test_series_assign_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))


        self.assertEqual(
                s1.assign.loc[['b', 'd']](3000).values.tolist(),
                [0, 3000, 2, 3000])

        self.assertEqual(
                s1.assign['b':](300).values.tolist(),
                [0, 300, 300, 300])


    def test_series_assign_b(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(list(s1.isin([2]).items()),
                [('a', False), ('b', False), ('c', True), ('d', False)])

        self.assertEqual(list(s1.isin({2, 3}).items()),
                [('a', False), ('b', False), ('c', True), ('d', True)])

        self.assertEqual(list(s1.isin(range(2, 4)).items()),
                [('a', False), ('b', False), ('c', True), ('d', True)])


    def test_series_assign_c(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.assign.loc['c':](0).to_pairs(),
                (('a', 0), ('b', 1), ('c', 0), ('d', 0))
                )
        self.assertEqual(s1.assign.loc['c':]((20, 30)).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 30)))

        self.assertEqual(s1.assign['c':](s1['c':] * 10).to_pairs(),
                (('a', 0), ('b', 1), ('c', 20), ('d', 30)))

        self.assertEqual(s1.assign['c':](Series({'d':40, 'c':60})).to_pairs(),
                (('a', 0), ('b', 1), ('c', 60), ('d', 40)))


    def test_series_assign_d(self):
        s1 = Series(tuple('pqrs'), index=('a', 'b', 'c', 'd'))
        s2 = s1.assign['b'](None)
        self.assertEqual(s2.to_pairs(),
                (('a', 'p'), ('b', None), ('c', 'r'), ('d', 's')))
        self.assertEqual(s1.assign['b':](None).to_pairs(),
                (('a', 'p'), ('b', None), ('c', None), ('d', None)))


    def test_series_loc_extract_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        # TODO: raaise exectin when doing a loc that Pandas reindexes



    def test_series_group_a(self):

        s1 = Series((0, 1, 0, 1), index=('a', 'b', 'c', 'd'))

        groups = tuple(s1.iter_group_items())

        self.assertEqual([g[0] for g in groups], [0, 1])

        self.assertEqual([g[1].to_pairs() for g in groups],
                [(('a', 0), ('c', 0)), (('b', 1), ('d', 1))])

    def test_series_group_b(self):

        s1 = Series(('foo', 'bar', 'foo', 20, 20),
                index=('a', 'b', 'c', 'd', 'e'),
                dtype=object)

        groups = tuple(s1.iter_group_items())


        self.assertEqual([g[0] for g in groups],
                [20, 'bar', 'foo'])
        self.assertEqual([g[1].to_pairs() for g in groups],
                [(('d', 20), ('e', 20)), (('b', 'bar'),), (('a', 'foo'), ('c', 'foo'))])


    def test_series_group_c(self):

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



    def test_series_iter_element_a(self):

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



    def test_series_sort_index_a(self):

        s1 = Series((10, 3, 28, 21, 15),
                index=('a', 'c', 'b', 'e', 'd'),
                dtype=object)

        self.assertEqual(s1.sort_index().to_pairs(),
                (('a', 10), ('b', 28), ('c', 3), ('d', 15), ('e', 21)))

        self.assertEqual(s1.sort_values().to_pairs(),
                (('c', 3), ('a', 10), ('d', 15), ('e', 21), ('b', 28)))


    def test_series_relabel_a(self):

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        s2 = s1.relabel({'b': 'bbb'})
        self.assertEqual(s2.to_pairs(),
                (('a', 0), ('bbb', 1), ('c', 2), ('d', 3)))

        self.assertEqual(mloc(s2.values), mloc(s1.values))


    def test_series_relabel_b(self):

        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        s2 = s1.relabel({'a':'x', 'b':'y', 'c':'z', 'd':'q'})

        self.assertEqual(list(s2.items()),
            [('x', 0), ('y', 1), ('z', 2), ('q', 3)])

    def test_series_get_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))
        self.assertEqual(s1.get('q'), None)
        self.assertEqual(s1.get('a'), 0)
        self.assertEqual(s1.get('f', -1), -1)


    def test_series_all_a(self):
        s1 = Series(range(4), index=('a', 'b', 'c', 'd'))

        self.assertEqual(s1.all(), False)
        self.assertEqual(s1.any(), True)


    def test_series_all_b(self):
        s1 = Series([True, True, np.nan, True], index=('a', 'b', 'c', 'd'), dtype=object)

        self.assertEqual(s1.all(skipna=False), True)
        self.assertEqual(s1.all(skipna=True), False)
        self.assertEqual(s1.any(), True)


    def test_series_unique_a(self):
        s1 = Series([10, 10, 2, 2], index=('a', 'b', 'c', 'd'), dtype=int)

        self.assertEqual(s1.unique().tolist(), [2, 10])

        s2 = Series(['b', 'b', 'c', 'c'], index=('a', 'b', 'c', 'd'), dtype=object)
        self.assertEqual(s2.unique().tolist(), ['b', 'c'])


    def test_series_unique_a(self):
        s1 = Series([10, 10, 2, 2], index=('a', 'b', 'c', 'd'), dtype=int)

        self.assertEqual(s1.unique().tolist(), [2, 10])

        s2 = Series(['b', 'b', 'c', 'c'], index=('a', 'b', 'c', 'd'), dtype=object)
        self.assertEqual(s2.unique().tolist(), ['b', 'c'])



    def test_series_duplicated_a(self):
        s1 = Series([1, 10, 10, 5, 2, 2],
                index=('a', 'b', 'c', 'd', 'e', 'f'), dtype=int)

        # this is showing all duplicates, not just the first-found
        self.assertEqual(s1.duplicated().to_pairs(),
                (('a', False), ('b', True), ('c', True), ('d', False), ('e', True), ('f', True)))

        self.assertEqual(s1.duplicated(exclude_first=True).to_pairs(),
                (('a', False), ('b', False), ('c', True), ('d', False), ('e', False), ('f', True)))

        self.assertEqual(s1.duplicated(exclude_last=True).to_pairs(),
                (('a', False), ('b', True), ('c', False), ('d', False), ('e', True), ('f', False)))


    def test_series_duplicated_b(self):
        s1 = Series([5, 3, 3, 3, 7, 2, 2, 2, 1],
                index=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), dtype=int)

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


    def test_series_drop_duplicated_a(self):
        s1 = Series([5, 3, 3, 3, 7, 2, 2, 2, 1],
                index=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), dtype=int)

        self.assertEqual(s1.drop_duplicated().to_pairs(),
                (('a', 5), ('e', 7), ('i', 1)))

        self.assertEqual(s1.drop_duplicated(exclude_first=True).to_pairs(),
                (('a', 5), ('b', 3), ('e', 7), ('f', 2), ('i', 1))
                )


    def test_series_reindex_add_level(self):
        s1 = Series(['a', 'b', 'c'])

        s2 = s1.reindex_add_level('I')
        self.assertEqual(s2.index.depth, 2)
        self.assertEqual(s2.to_pairs(),
                ((('I', 0), 'a'), (('I', 1), 'b'), (('I', 2), 'c')))

        s3 = s2.reindex_flat()
        self.assertEqual(s3.index.depth, 1)
        self.assertEqual(s3.to_pairs(),
                ((('I', 0), 'a'), (('I', 1), 'b'), (('I', 2), 'c')))


    def test_series_drop_level_a(self):
        s1 = Series(['a', 'b', 'c'],
                index=IndexHierarchy.from_labels([('A', 1), ('B', 1), ('C', 1)]))
        s2 = s1.reindex_drop_level()
        self.assertEqual(s2.to_pairs(),
                (('A', 'a'), ('B', 'b'), ('C', 'c'))
                )


    @unittest.skip('non required dependency')
    def test_series_from_pandas_a(self):
        import pandas as pd

        pds = pd.Series([3,4,5], index=list('abc'))
        sfs = Series.from_pandas(pds)
        self.assertEqual(list(pds.items()), list(sfs.items()))

        # mutate Pandas
        pds['c'] = 50
        self.assertNotEqual(pds['c'], sfs['c'])

        # owning data
        pds = pd.Series([3,4,5], index=list('abc'))
        sfs = Series.from_pandas(pds, own_data=True, own_index=True)
        self.assertEqual(list(pds.items()), list(sfs.items()))

        # NOTE: some operations in Pandas can refresh the values attribute


    def test_series_astype_a(self):

        s1 = Series(['a', 'b', 'c'])

        s2 = s1.astype(object)
        self.assertEqual(s2.to_pairs(),
                ((0, 'a'), (1, 'b'), (2, 'c')))
        self.assertTrue(s2.dtype == object)

        # we cannot convert to float
        with self.assertRaises(ValueError):
            s1.astype(float)

    def test_series_astype_b(self):

        s1 = Series([1, 3, 4, 0])

        s2 = s1.astype(bool)
        self.assertEqual(
                s2.to_pairs(),
                ((0, True), (1, True), (2, True), (3, False)))
        self.assertTrue(s2.dtype == bool)


    def test_series_min_max_a(self):

        s1 = Series([1, 3, 4, 0])
        self.assertEqual(s1.min(), 0)
        self.assertEqual(s1.max(), 4)


        s2 = sf.Series([-1, 4, None, np.nan])
        self.assertEqual(s2.min(), -1)
        self.assertTrue(np.isnan(s2.min(skipna=False)))

        self.assertEqual(s2.max(), 4)
        self.assertTrue(np.isnan(s2.max(skipna=False)))


    def test_series_min_max_b(self):
        # string objects work as expected; when fixed length strings, however, the do not

        s1 = Series(list('abc'), dtype=object)
        self.assertEqual(s1.min(), 'a')
        self.assertEqual(s1.max(), 'c')

        # get the same result from character arrays
        s2 = sf.Series(list('abc'))
        self.assertEqual(s2.min(), 'a')
        self.assertEqual(s2.max(), 'c')




if __name__ == '__main__':
    unittest.main()
