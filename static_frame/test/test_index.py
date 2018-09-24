
import unittest
import numpy as np

from static_frame import Index
from static_frame import IndexGO
from static_frame import IndexDate
from static_frame import Series
from static_frame import IndexYearMonth
from static_frame import IndexYear
from static_frame import HLoc

from static_frame.test.test_case import TestCase


class TestUnit(TestCase):


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

        self.assertEqual(idx.loc['c'].values.tolist(), ['c'])



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


    def test_index_date_b(self):

        with self.assertRaises(Exception):
            index = IndexDate([3,4,5])

        with self.assertRaises(Exception):
            index = IndexDate(['2017', '2018'])

        with self.assertRaises(Exception):
            index = IndexDate(['2017-01', '2018-07'])


    def test_index_date_c(self):
        index = IndexDate.from_date_range('2017-12-15', '2018-03-15', 2)

        self.assertEqual((index == '2017').sum(), 9)
        self.assertEqual((index == '2018-02').sum(), 14)
        self.assertEqual((index == '2018').sum(), 37)


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


    def test_index_year_month_from_date_range_a(self):
        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15')
        self.assertEqual(len(index), 4)

        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15', 2)
        self.assertEqual(len(index), 2)

    def test_index_year_month_from_year_month_range_a(self):

        with self.assertRaises(Exception):
            index = IndexYearMonth.from_year_month_range(
                    '2017-12-15', '2018-03-15')

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


if __name__ == '__main__':
    unittest.main()


