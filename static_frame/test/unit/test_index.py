
import unittest
import numpy as np
import pickle
import datetime
from io import StringIO

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexDate
from static_frame import IndexHierarchy
from static_frame import Series
from static_frame import Frame
from static_frame import IndexYearMonth
from static_frame import IndexYear
from static_frame import IndexSecond
from static_frame import IndexMillisecond

from static_frame import HLoc
from static_frame import ILoc


from static_frame.test.test_case import TestCase
from static_frame.core.index import _requires_reindex
from static_frame.core.index import _is_index_initializer


class TestUnit(TestCase):

    def test_index_init_a(self):
        idx1 = Index(('a', 'b', 'c', 'd'), name='foo')
        idx2 = Index(idx1)

        self.assertEqual(idx1.name, 'foo')
        self.assertEqual(idx2.name, 'foo')


    def test_index_init_b(self):

        idx1 = IndexHierarchy.from_product(['A', 'B'], [1, 2])

        idx2 = Index(idx1)

        self.assertEqual(idx2.values.tolist(),
            [('A', 1), ('A', 2), ('B', 1), ('B', 2)])


    def test_index_init_c(self):


        s1 = Series(('a', 'b', 'c'))
        idx2 = Index(s1)
        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c']
                )

    def test_index_loc_to_iloc_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(
                idx.loc_to_iloc(np.array([True, False, True, False])).tolist(),
                [0, 2])

        self.assertEqual(idx.loc_to_iloc(slice('c',)), slice(None, 3, None))
        self.assertEqual(idx.loc_to_iloc(slice('b','d')), slice(1, 4, None))
        self.assertEqual(idx.loc_to_iloc('d'), 3)


    def test_index_mloc_a(self):
        idx = Index(('a', 'b', 'c', 'd'))
        self.assertTrue(idx.mloc == idx[:2].mloc)

    def test_index_mloc_b(self):
        idx = IndexGO(('a', 'b', 'c', 'd'))
        idx.append('e')
        self.assertTrue(idx.mloc == idx[:2].mloc)


    def test_index_dtype_a(self):
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(str(idx.dtype), '<U1')
        idx.append('eee')
        self.assertEqual(str(idx.dtype), '<U3')


    def test_index_shape_a(self):
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.shape, (4,))
        idx.append('e')
        self.assertEqual(idx.shape, (5,))


    def test_index_ndim_a(self):
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.ndim, 1)
        idx.append('e')
        self.assertEqual(idx.ndim, 1)


    def test_index_size_a(self):
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.size, 4)
        idx.append('e')
        self.assertEqual(idx.size, 5)

    def test_index_nbytes_a(self):
        idx = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idx.nbytes, 16)
        idx.append('e')
        self.assertEqual(idx.nbytes, 20)


    def test_index_rename_a(self):
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        idx1.append('e')
        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')

    def test_index_positions_a(self):
        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.positions.tolist(), list(range(4)))

        idx1.append('e')
        self.assertEqual(idx1.positions.tolist(), list(range(5)))


    def test_index_unique(self):

        with self.assertRaises(KeyError):
            idx = Index(('a', 'b', 'c', 'a'))
        with self.assertRaises(KeyError):
            idx = IndexGO(('a', 'b', 'c', 'a'))

        with self.assertRaises(KeyError):
            idx = Index(['a', 'a'])
        with self.assertRaises(KeyError):
            idx = IndexGO(['a', 'a'])

        with self.assertRaises(KeyError):
            idx = Index(np.array([True, False, True], dtype=bool))
        with self.assertRaises(KeyError):
            idx = IndexGO(np.array([True, False, True], dtype=bool))

        # acceptable but not advisiable
        idx = Index([0, '0'])


    def test_index_creation_a(self):
        idx = Index(('a', 'b', 'c', 'd'))

        #idx2 = idx['b':'d']

        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])

        self.assertEqual(idx[2:].values.tolist(), ['c', 'd'])

        self.assertEqual(idx.loc['b':].values.tolist(), ['b', 'c', 'd'])

        self.assertEqual(idx.loc['b':'d'].values.tolist(), ['b', 'c', 'd'])

        self.assertEqual(idx.loc_to_iloc(['b', 'b', 'c']), [1, 1, 2])

        self.assertEqual(idx.loc['c'], 'c')

        idxgo = IndexGO(('a', 'b', 'c', 'd'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd'])

        idxgo.append('e')
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e'])

        idxgo.extend(('f', 'g'))
        self.assertEqual(idxgo.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'g'])


    def test_index_creation_b(self):
        idx = Index((x for x in ('a', 'b', 'c', 'd') if x in {'b', 'd'}))
        self.assertEqual(idx.loc_to_iloc('b'), 0)
        self.assertEqual(idx.loc_to_iloc('d'), 1)


    def test_index_unary_operators_a(self):
        idx = Index((20, 30, 40, 50))

        invert_idx = -idx
        self.assertEqual(invert_idx.tolist(),
                [-20, -30, -40, -50],)

        # this is strange but consistent with NP
        not_idx = ~idx
        self.assertEqual(not_idx.tolist(),
                [-21, -31, -41, -51],)

    def test_index_binary_operators_a(self):
        idx = Index((20, 30, 40, 50))

        self.assertEqual((idx + 2).tolist(),
                [22, 32, 42, 52])
        self.assertEqual((2 + idx).tolist(),
                [22, 32, 42, 52])
        self.assertEqual((idx * 2).tolist(),
                [40, 60, 80, 100])
        self.assertEqual((2 * idx).tolist(),
                [40, 60, 80, 100])
        self.assertEqual((idx - 2).tolist(),
                [18, 28, 38, 48])
        self.assertEqual(
                (2 - idx).tolist(),
                [-18, -28, -38, -48])


    def test_index_binary_operators_b(self):
        '''Both operands are Index instances
        '''
        idx1 = Index((20, 30, 40, 50))
        idx2 = Index((20, 3, 4, 5))

        self.assertEqual((idx1 == idx2).tolist(), [True, False, False, False])



    def test_index_ufunc_axis_a(self):

        idx = Index((30, 40, 50))

        self.assertEqual(idx.min(), 30)
        self.assertEqual(idx.max(), 50)
        self.assertEqual(idx.sum(), 120)

    def test_index_isin_a(self):

        idx = Index((30, 40, 50))

        self.assertEqual(idx.isin([40, 50]).tolist(), [False, True, True])
        self.assertEqual(idx.isin({40, 50}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(frozenset((40, 50))).tolist(), [False, True, True])

        self.assertEqual(idx.isin({40: 'a', 50: 'b'}).tolist(), [False, True, True])

        self.assertEqual(idx.isin(range(35, 45)).tolist(), [False, True, False])

        self.assertEqual(idx.isin((x * 10 for x in (3, 4, 5, 6, 6))).tolist(), [True, True, True])



    def test_index_contains_a(self):

        index = Index(('a', 'b', 'c'))
        self.assertTrue('a' in index)
        self.assertTrue('d' not in index)


    def test_index_grow_only_a(self):

        index = IndexGO(('a', 'b', 'c'))
        index.append('d')
        self.assertEqual(index.loc_to_iloc('d'), 3)

        index.extend(('e', 'f'))
        self.assertEqual(index.loc_to_iloc('e'), 4)
        self.assertEqual(index.loc_to_iloc('f'), 5)

        # creating an index form an Index go takes the np arrays, but not the mutable bits
        index2 = Index(index)
        index.append('h')

        self.assertEqual(len(index2), 6)
        self.assertEqual(len(index), 7)

        index3 = index[2:]
        index3.append('i')

        self.assertEqual(index3.values.tolist(), ['c', 'd', 'e', 'f', 'h', 'i'])
        self.assertEqual(index.values.tolist(), ['a', 'b', 'c', 'd', 'e', 'f', 'h'])



    def test_index_sort(self):

        index = Index(('a', 'c', 'd', 'e', 'b'))
        self.assertEqual(
                [index.sort().loc_to_iloc(x) for x in sorted(index.values)],
                [0, 1, 2, 3, 4])
        self.assertEqual(
                [index.sort(ascending=False).loc_to_iloc(x) for x in sorted(index.values)],
                [4, 3, 2, 1, 0])


    def test_index_relable(self):

        index = Index(('a', 'c', 'd', 'e', 'b'))

        self.assertEqual(
                index.relabel(lambda x: x.upper()).values.tolist(),
                ['A', 'C', 'D', 'E', 'B'])

        # letter to number
        s1 = Series(range(5), index=index.values)

        self.assertEqual(
                index.relabel(s1).values.tolist(),
                [0, 1, 2, 3, 4]
                )

        self.assertEqual(index.relabel({'e': 'E'}).values.tolist(),
                ['a', 'c', 'd', 'E', 'b'])



    def test_index_date_a(self):

        index = IndexDate.from_date_range('2018-01-01', '2018-03-01')
        self.assertEqual(index.values[0], np.datetime64('2018-01-01'))
        self.assertEqual(index.values[-1], np.datetime64('2018-03-01'))
        self.assertEqual(index.loc['2018-02-22'],
                np.datetime64('2018-02-22'))


    def test_index_date_b(self):

        with self.assertRaises(Exception):
            IndexDate([3,4,5], dtype=np.int64)

        idx1 = IndexDate(['2017', '2018'])
        self.assertTrue(idx1[0].__class__ == np.datetime64)
        self.assertEqual(idx1.loc_to_iloc('2018-01-01'), 1)

        idx2 = IndexDate(['2017-01', '2018-07'])
        self.assertTrue(idx2[0].__class__ == np.datetime64)
        self.assertEqual(idx2.loc['2017-01-01'],
                np.datetime64('2017-01-01'))

    def test_index_date_c(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-03-15', 2)

        self.assertEqual((index == '2017').sum(), 9)
        self.assertEqual((index == '2018-02').sum(), 14)
        self.assertEqual((index == '2018').sum(), 37)

    def test_index_date_d(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-03-15', 2)
        # selct by year and year month
        self.assertAlmostEqualValues(index.loc['2017'].values,
                np.array(['2017-12-15', '2017-12-17', '2017-12-19', '2017-12-21',
               '2017-12-23', '2017-12-25', '2017-12-27', '2017-12-29',
               '2017-12-31'], dtype='datetime64[D]'))

        self.assertAlmostEqualValues(index.loc['2018-02'].values,
                np.array(['2018-02-01', '2018-02-03', '2018-02-05', '2018-02-07',
               '2018-02-09', '2018-02-11', '2018-02-13', '2018-02-15',
               '2018-02-17', '2018-02-19', '2018-02-21', '2018-02-23',
               '2018-02-25', '2018-02-27'], dtype='datetime64[D]'))

        self.assertEqual(index.loc['2018-02-19'],
                np.datetime64('2018-02-19'))

    def test_index_date_e(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-03-15', 2)

        post = index + np.timedelta64(2, 'D')

        self.assertEqual(post[0], np.datetime64('2017-12-17'))


    def test_index_date_f(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-01-15')

        post = index + datetime.timedelta(days=10)

        self.assertEqual(post[0], np.datetime64('2017-12-25'))
        self.assertEqual(post[-1], np.datetime64('2018-01-25'))


    def test_index_date_g(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-02-15')

        post = index.loc['2018':'2018-01']
        self.assertEqual(len(post), 31)
        self.assertEqual(post[0], np.datetime64('2018-01-01'))
        self.assertEqual(post[-1], np.datetime64('2018-01-31'))


    def test_index_date_h(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-02-15')

        post = index.loc['2018':'2018-01-15']
        self.assertEqual(len(post), 15)
        self.assertEqual(post[0], np.datetime64('2018-01-01'))
        self.assertEqual(post[-1], np.datetime64('2018-01-15'))


    def test_index_date_i(self):
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')

        post = index.loc['2017-12': '2018-01']
        self.assertEqual(len(post), 62)
        self.assertEqual(post[0], np.datetime64('2017-12-01'))
        self.assertEqual(post[-1], np.datetime64('2018-01-31'))


    def test_index_date_j(self):
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')

        post = index.loc['2017-12': '2018']
        self.assertEqual(len(post), 77)
        self.assertEqual(post[0], np.datetime64('2017-12-01'))
        self.assertEqual(post[-1], np.datetime64('2018-02-15'))


    def test_index_date_k(self):
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')
        post = index.loc[['2017-12-10', '2018-02-06']]
        self.assertEqual(len(post), 2)
        self.assertEqual(post[0], np.datetime64('2017-12-10'))
        self.assertEqual(post[-1], np.datetime64('2018-02-06'))


    def test_index_date_m(self):
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')
        # NOTE: this type of selection should possibly not be permitted
        post = index.loc[['2017', '2018']]
        self.assertEqual(len(post), 93)
        self.assertEqual(post[0], np.datetime64('2017-11-15'))
        self.assertEqual(post[-1], np.datetime64('2018-02-15'))

    def test_index_date_n(self):
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')
        # NOTE: this type of selection should possibly not be permitted
        post = index.loc[['2017-12', '2018-02']]
        self.assertEqual(len(post), 46)
        self.assertEqual(post[0], np.datetime64('2017-12-01'))
        self.assertEqual(post[-1], np.datetime64('2018-02-15'))
        self.assertEqual(
            set(post.values.astype('datetime64[M]')),
            {np.datetime64('2018-02'), np.datetime64('2017-12')}
            )


    def test_index_date_from_year_month_range_a(self):
        index = IndexDate.from_year_month_range('2017-12', '2018-03')

        self.assertEqual((index == '2017').sum(), 31)
        self.assertEqual((index == '2018').sum(), 90)

        self.assertEqual(
            [str(d) for d in np.unique(index.values.astype('datetime64[M]'))],
            ['2017-12', '2018-01', '2018-02', '2018-03'])


    def test_index_date_from_year_range_a(self):
        index = IndexDate.from_year_range('2016', '2018')
        self.assertEqual(len(index), 1096)
        self.assertEqual(
                [str(d) for d in np.unique(index.values.astype('datetime64[Y]'))],
                ['2016', '2017', '2018'])

        index = IndexDate.from_year_range('2016', '2018', 2)
        self.assertEqual(len(index), 548)
        self.assertEqual(
                [str(d) for d in np.unique(index.values.astype('datetime64[Y]'))],
                ['2016', '2017', '2018'])


    def test_index_date_series_a(self):

        s = Series(range(62),
                index=IndexDate.from_year_month_range('2017-12', '2018-01'))

        self.assertEqual(s.sum(), 1891)
        self.assertEqual(s.loc[s.index == '2018-01'].sum(), 1426)
        self.assertEqual(s.loc[s.index == '2017-12'].sum(), 465)

        self.assertEqual(s['2018-01-24'], 54)

        self.assertEqual(
                s['2018-01-28':].to_pairs(),
                ((np.datetime64('2018-01-28'), 58), (np.datetime64('2018-01-29'), 59), (np.datetime64('2018-01-30'), 60), (np.datetime64('2018-01-31'), 61))
                )

        # import ipdb; ipdb.set_trace()

    def test_index_year_month_a(self):
        idx1 = IndexYearMonth(('2018-01', '2018-06'))

        self.assertEqual(idx1.values.tolist(),
            [datetime.date(2018, 1, 1), datetime.date(2018, 6, 1)])


    def test_index_year_month_from_date_range_a(self):
        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15')
        self.assertEqual(len(index), 4)

        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15', 2)
        self.assertEqual(len(index), 2)

    def test_index_year_month_from_year_month_range_a(self):

        index = IndexYearMonth.from_year_month_range(
                '2017-12-15', '2018-03-15')
        self.assertAlmostEqualValues(index.values,
                np.array(['2017-12', '2018-01', '2018-02', '2018-03'],
                dtype='datetime64[M]'))

        index = IndexYearMonth.from_year_month_range('2017-12', '2018-03')
        self.assertEqual(len(index), 4)

        self.assertEqual([str(d) for d in index.values],
                ['2017-12', '2018-01', '2018-02', '2018-03'])

        index = IndexYearMonth.from_year_month_range('2017-12', '2018-03', step=2)
        self.assertEqual([str(d) for d in index], ['2017-12', '2018-02'])


    def test_index_year_month_from_year_range_a(self):

        index = IndexYearMonth.from_year_range('2010', '2018')

        self.assertEqual(len(index), 108)
        self.assertEqual(str(index.min()), '2010-01')
        self.assertEqual(str(index.max()), '2018-12')

        index = IndexYearMonth.from_year_range('2010', '2018', 6)

        self.assertEqual(
                [str(d) for d in IndexYearMonth.from_year_range('2010', '2018', 6)],
                ['2010-01', '2010-07', '2011-01', '2011-07', '2012-01', '2012-07', '2013-01', '2013-07', '2014-01', '2014-07', '2015-01', '2015-07', '2016-01', '2016-07', '2017-01', '2017-07', '2018-01', '2018-07'])


    def test_index_year_from_date_range_a(self):

        index = IndexYear.from_date_range('2014-12-15', '2018-03-15')
        self.assertEqual(len(index), 5)

        index = IndexYear.from_date_range('2014-12-15', '2018-03-15', step=2)
        self.assertEqual([str(d) for d in index.values],
                ['2014', '2016', '2018'])


    def test_index_year_from_year_month_range_a(self):

        index = IndexYear.from_year_month_range('2014-12', '2018-03')
        self.assertEqual(len(index), 5)


    def test_index_year_from_year_range_a(self):

        index = IndexYear.from_year_range('2010', '2018')
        self.assertEqual(len(index), 9)


    def test_index_date_loc_to_iloc_a(self):

        index = IndexDate.from_date_range('2018-01-01', '2018-03-01')

        self.assertEqual(
                index.loc_to_iloc(np.datetime64('2018-02-11')),
                41)

        self.assertEqual(index.loc_to_iloc('2018-02-11'), 41)

        self.assertEqual(
                index.loc_to_iloc(slice('2018-02-11', '2018-02-24')),
                slice(41, 55, None))


    def test_index_tuples_a(self):

        index = Index([('a','b'), ('b','c'), ('c','d')])
        s1 = Series(range(3), index=index)

        self.assertEqual(s1[('b', 'c'):].values.tolist(), [1, 2])

        self.assertEqual(s1[[('b', 'c'), ('a', 'b')]].values.tolist(), [1, 0])

        self.assertEqual(s1[('b', 'c')], 1)
        self.assertEqual(s1[('c', 'd')], 2)

        s2 = Series(range(10), index=((1, x) for x in range(10)))
        self.assertEqual(s2[(1, 5):].values.tolist(),
                [5, 6, 7, 8, 9])

        self.assertEqual(s2[[(1, 7), (1, 5), (1, 0)]].values.tolist(),
                [7, 5, 0])



    def test_requires_reindex(self):
        a = Index([1, 2, 3])
        b = Index([1, 2, 3])
        c = Index([1, 3, 2])
        d = Index([1, 2, 3, 4])
        e = Index(['a', 2, 3])

        self.assertFalse(_requires_reindex(a, b))
        self.assertTrue(_requires_reindex(a, c))
        self.assertTrue(_requires_reindex(a, c))
        self.assertTrue(_requires_reindex(a, d))
        self.assertTrue(_requires_reindex(a, e))

    def test_index_pickle_a(self):
        a = Index([('a','b'), ('b','c'), ('c','d')])
        b = Index([1, 2, 3, 4])
        c = IndexYear.from_date_range('2014-12-15', '2018-03-15')

        for index in (a, b, c):
            pbytes = pickle.dumps(index)
            index_new = pickle.loads(pbytes)
            for v in index: # iter labels
                # import ipdb; ipdb.set_trace()
                # this compares Index objects
                self.assertFalse(index_new._labels.flags.writeable)
                self.assertEqual(index_new.loc[v], index.loc[v])

    def test_index_drop_a(self):

        index = Index(list('abcdefg'))

        self.assertEqual(index._drop_loc('d').values.tolist(),
                ['a', 'b', 'c', 'e', 'f', 'g'])

        self.assertEqual(index._drop_loc(['a', 'g']).values.tolist(),
                ['b', 'c', 'd', 'e', 'f'])

        self.assertEqual(index._drop_loc(slice('b', None)).values.tolist(),
                ['a'])


    def test_index_iloc_loc_to_iloc(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.loc_to_iloc(ILoc[1]), 1)
        self.assertEqual(idx.loc_to_iloc(ILoc[[0, 2]]), [0, 2])


    def test_index_loc_to_iloc_boolen_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        # unlike Pandas, both of these presently fail
        with self.assertRaises(KeyError):
            idx.loc_to_iloc([False, True])

        with self.assertRaises(KeyError):
            idx.loc_to_iloc([False, True, False, True])

        # but a Boolean array works
        post = idx.loc_to_iloc(np.array([False, True, False, True]))
        self.assertEqual(post.tolist(), [1, 3])


    def test_index_loc_to_iloc_boolen_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        # returns nothing as index does not match anything
        post = idx.loc_to_iloc(Series([False, True, False, True]))
        self.assertTrue(len(post) == 0)

        post = idx.loc_to_iloc(Series([False, True, False, True],
                index=('b', 'c', 'd', 'a')))
        self.assertEqual(post.tolist(), [0, 2])

        post = idx.loc_to_iloc(Series([False, True, False, True],
                index=list('abcd')))
        self.assertEqual(post.tolist(), [1,3])


    def test_index_drop_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.drop.iloc[2].values.tolist(), ['a', 'b', 'd'])
        self.assertEqual(idx.drop.iloc[2:].values.tolist(), ['a', 'b'])
        self.assertEqual(
                idx.drop.iloc[np.array([True, False, False, True])].values.tolist(),
                ['b', 'c'])

    def test_index_drop_b(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.drop.loc['c'].values.tolist(), ['a', 'b', 'd'])
        self.assertEqual(idx.drop.loc['b':'c'].values.tolist(), ['a', 'd'])

        self.assertEqual(
                idx.drop.loc[np.array([True, False, False, True])].values.tolist(),
                ['b', 'c']
                )

    def test_index_roll_a(self):

        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.roll(-2).values.tolist(),
                ['c', 'd', 'a', 'b'])

        self.assertEqual(idx.roll(1).values.tolist(),
                ['d', 'a', 'b', 'c'])


    def test_index_attributes_a(self):
        idx = Index(('a', 'b', 'c', 'd'))

        self.assertEqual(idx.shape, (4,))
        self.assertEqual(idx.dtype.kind, 'U')
        self.assertEqual(idx.ndim, 1)
        self.assertEqual(idx.nbytes, 16)


    def test_index_name_a(self):

        idx1 = Index(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.name, 'foo')

        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')

    def test_name_b(self):

        with self.assertRaises(TypeError):
            Index(('a', 'b', 'c', 'd'), name=['x', 'y'])

        with self.assertRaises(TypeError):
            Index(('a', 'b', 'c', 'd'), name={'x', 'y'})


    def test_index_name_c(self):

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(idx1.name, 'foo')

        idx2 = idx1.rename('bar')
        self.assertEqual(idx2.name, 'bar')

        idx1.append('e')
        idx2.append('x')

        self.assertEqual(idx1.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])

        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c', 'd', 'x'])


    def test_index_to_pandas_a(self):

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, idx1.name)
        self.assertTrue((pdidx.values == idx1.values).all())


    def test_index_to_pandas_b(self):
        import pandas
        idx1 = IndexDate(('2018-01-01', '2018-06-01'), name='foo')
        pdidx = idx1.to_pandas()
        self.assertEqual(pdidx.name, idx1.name)
        self.assertTrue((pdidx.values == idx1.values).all())
        self.assertTrue(pdidx[1].__class__ == pandas.Timestamp)


    def test_index_from_pandas_a(self):
        import pandas

        pdidx = pandas.Index(list('abcd'))
        idx = Index.from_pandas(pdidx)
        self.assertEqual(idx.values.tolist(), ['a', 'b', 'c', 'd'])


    def test_index_from_pandas_a(self):
        import pandas

        pdidx = pandas.DatetimeIndex(('2018-01-01', '2018-06-01'), name='foo')
        idx = IndexDate.from_pandas(pdidx)
        self.assertEqual(idx.values.tolist(),
                [datetime.date(2018, 1, 1), datetime.date(2018, 6, 1)])


    def test_index_iter_label_a(self):

        idx1 = IndexGO(('a', 'b', 'c', 'd'), name='foo')
        self.assertEqual(list(idx1.iter_label(0)), ['a', 'b', 'c', 'd'])

        post = idx1.iter_label(0).apply(lambda x: x.upper())
        self.assertEqual(post.to_pairs(),
                ((0, 'A'), (1, 'B'), (2, 'C'), (3, 'D')))


    def test_index_intersection_a(self):

        idx1 = Index(('a', 'b', 'c', 'd', 'e'))

        a1 = np.array(['c', 'dd', 'b', 'a'])

        idx2 = idx1.intersection(a1)

        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c'])


    def test_index_union_a(self):

        idx1 = IndexGO(('a', 'b', 'c', 'd', 'e'))
        idx1.append('f')
        a1 = np.array(['c', 'dd', 'b', 'a'])

        idx2 = idx1.union(a1)

        self.assertEqual(idx2.values.tolist(),
                ['a', 'b', 'c', 'd', 'dd', 'e', 'f'])


    def test_index_to_html_a(self):

        idx1 = IndexGO(('a', 'b', 'c'))

        self.assertEqual(idx1.to_html(),
                '<table border="1"><thead><tr><th><span style="color: #777777">&lt;IndexGO&gt;</span></th></tr></thead><tbody><tr><td>a</td></tr><tr><td>b</td></tr><tr><td>c</td></tr></tbody></table>')

    def test_index_to_html_datatables_a(self):

        idx1 = IndexGO(('a', 'b', 'c'))

        sio = StringIO()

        post = idx1.to_html_datatables(sio, show=False)

        self.assertEqual(post, None)

        self.assertTrue(len(sio.read()) > 1300)


    def test_index_millisecond_a(self):

        msg = '''2016-04-28 04:22:12.226
2016-04-28 16:29:21.32
2016-04-28 17:36:13.733
2016-04-30 20:21:07.848
2016-05-01 00:00:33.483
2016-05-01 03:02:03.584
2016-05-01 09:26:43.185
2016-05-01 13:45:22.576
2016-05-01 15:25:46.15'''

        idx = IndexMillisecond(msg.split('\n'))
        self.assertEqual(str(idx.dtype), 'datetime64[ms]')

        self.assertEqual(idx.loc['2016-04-30T20:21:07.848'],
                np.datetime64('2016-04-30T20:21:07.848'))

        self.assertAlmostEqualValues(
                idx.loc['2016-05-01T09:26:43.185':].values,
                np.array(['2016-05-01T09:26:43.185', '2016-05-01T13:45:22.576',
       '2016-05-01T15:25:46.150'], dtype='datetime64[ms]'))

        self.assertAlmostEqualValues(idx.loc['2016-05'].values,
                np.array(['2016-05-01T00:00:33.483', '2016-05-01T03:02:03.584',
               '2016-05-01T09:26:43.185', '2016-05-01T13:45:22.576',
               '2016-05-01T15:25:46.150'], dtype='datetime64[ms]')
                )

        self.assertEqual(idx.loc['2016-05-01T00'].values,
                np.array(['2016-05-01T00:00:33.483'], dtype='datetime64[ms]'))




    def test_index_millisecond_b(self):
        # integer arguments are interpreted as milliseconds from the epoch
        idx = IndexMillisecond(range(10))
        self.assertAlmostEqualValues(idx.loc['1970-01-01T00:00:00.007':].values,
                np.array(['1970-01-01T00:00:00.007', '1970-01-01T00:00:00.008',
               '1970-01-01T00:00:00.009'], dtype='datetime64[ms]'))


    def test_index_second_a(self):
        # integer arguments are interpreted as seconds from the epoch
        idx = IndexSecond(range(10))
        self.assertAlmostEqualValues(idx.loc['1970-01-01T00:00:07':].values,
                np.array(['1970-01-01T00:00:07', '1970-01-01T00:00:08',
               '1970-01-01T00:00:09'], dtype='datetime64[s]')
                )


    def test_index_millisecond_series_a(self):

        msg = '''2016-04-28 04:22:12.226
2016-04-28 16:29:21.32
2016-04-28 17:36:13.733
2016-04-30 20:21:07.848
2016-05-01 00:00:33.483
2016-05-01 03:02:03.584
2016-05-01 09:26:43.185
2016-05-01 13:45:22.576
2016-05-01 15:25:46.15'''

        idx = IndexMillisecond(msg.split('\n'))
        s = Series(range(9), index=idx)

        self.assertEqual(s['2016-05-01T00:00:33.483'], 4)

        self.assertEqual(s['2016-05-01T00:00:33.483':].values.tolist(),
                [4, 5, 6, 7, 8])

        self.assertEqual(s['2016-05'].to_pairs(),
                ((np.datetime64('2016-05-01T00:00:33.483'), 4), (np.datetime64('2016-05-01T03:02:03.584'), 5), (np.datetime64('2016-05-01T09:26:43.185'), 6), (np.datetime64('2016-05-01T13:45:22.576'), 7), (np.datetime64('2016-05-01T15:25:46.150'), 8)))

        self.assertEqual(s['2016-05-01T09'].to_pairs(),
                ((np.datetime64('2016-05-01T09:26:43.185'), 6),))


    def test_index_millisecond_frame_a(self):

        msg = '''2016-04-28 04:22:12.226
2016-04-28 16:29:21.32
2016-04-28 17:36:13.733
2016-04-30 20:21:07.848
2016-05-01 00:00:33.483
2016-05-01 03:02:03.584
2016-05-01 09:26:43.185
2016-05-01 13:45:22.576
2016-05-01 15:25:46.15'''

        f = Frame.from_records((x, y) for x, y in enumerate(msg.split('\n')))

        idx = IndexMillisecond(f[1])
        self.assertAlmostEqualValues(idx.values,
                np.array(['2016-04-28T04:22:12.226', '2016-04-28T16:29:21.320',
               '2016-04-28T17:36:13.733', '2016-04-30T20:21:07.848',
               '2016-05-01T00:00:33.483', '2016-05-01T03:02:03.584',
               '2016-05-01T09:26:43.185', '2016-05-01T13:45:22.576',
               '2016-05-01T15:25:46.150'], dtype='datetime64[ms]'))


        idx = IndexSecond(f[1])

        self.assertAlmostEqualValues(idx.values,
            np.array(['2016-04-28T04:22:12', '2016-04-28T16:29:21',
           '2016-04-28T17:36:13', '2016-04-30T20:21:07',
           '2016-05-01T00:00:33', '2016-05-01T03:02:03',
           '2016-05-01T09:26:43', '2016-05-01T13:45:22',
           '2016-05-01T15:25:46'], dtype='datetime64[s]'))


        f2 = f.set_index(1, index_constructor=IndexMillisecond)
        self.assertEqual(f2.loc['2016-05', 0].values.tolist(),
                [4, 5, 6, 7, 8])




if __name__ == '__main__':
    unittest.main()

