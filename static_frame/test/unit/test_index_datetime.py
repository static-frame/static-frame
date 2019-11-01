
import unittest
import numpy as np
# import pickle
import datetime
# import typing as tp
# from io import StringIO

# from static_frame import Index
# from static_frame import IndexGO
from static_frame import IndexDate
# from static_frame import IndexHierarchy
from static_frame import Series
from static_frame import Frame
from static_frame import IndexYearMonth
from static_frame import IndexYear
from static_frame import IndexSecond
from static_frame import IndexMillisecond

# from static_frame import HLoc
# from static_frame import ILoc


from static_frame.test.test_case import TestCase
# from static_frame.core.index import _requires_reindex
# from static_frame.core.exception import ErrorInitIndex


class TestUnit(TestCase):



    def test_index_date_a(self) -> None:

        index = IndexDate.from_date_range('2018-01-01', '2018-03-01')
        self.assertEqual(index.values[0], np.datetime64('2018-01-01'))
        self.assertEqual(index.values[-1], np.datetime64('2018-03-01'))
        self.assertEqual(index.loc['2018-02-22'],
                np.datetime64('2018-02-22'))


    def test_index_date_b(self) -> None:

        with self.assertRaises(Exception):
            IndexDate([3,4,5], dtype=np.int64)  # type: ignore #pylint: disable=E1123

        idx1 = IndexDate(['2017', '2018'])
        self.assertTrue(idx1[0].__class__ == np.datetime64)
        self.assertEqual(idx1.loc_to_iloc('2018-01-01'), 1)

        idx2 = IndexDate(['2017-01', '2018-07'])
        self.assertTrue(idx2[0].__class__ == np.datetime64)
        self.assertEqual(idx2.loc['2017-01-01'],
                np.datetime64('2017-01-01'))

    def test_index_date_c(self) -> None:
        index = IndexDate.from_date_range('2017-12-15', '2018-03-15', 2)

        self.assertEqual((index == '2017').sum(), 9)
        self.assertEqual((index == '2018-02').sum(), 14)
        self.assertEqual((index == '2018').sum(), 37)

    def test_index_date_d(self) -> None:
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

    def test_index_date_e(self) -> None:
        index = IndexDate.from_date_range('2017-12-15', '2018-03-15', 2)

        post = index + np.timedelta64(2, 'D')

        self.assertEqual(post[0], np.datetime64('2017-12-17'))


    def test_index_date_f(self) -> None:
        index = IndexDate.from_date_range('2017-12-15', '2018-01-15')

        post = index + datetime.timedelta(days=10)

        self.assertEqual(post[0], np.datetime64('2017-12-25'))
        self.assertEqual(post[-1], np.datetime64('2018-01-25'))


    def test_index_date_g(self) -> None:
        index = IndexDate.from_date_range('2017-12-15', '2018-02-15')

        post = index.loc['2018':'2018-01']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(len(post), 31)
        self.assertEqual(post[0], np.datetime64('2018-01-01'))
        self.assertEqual(post[-1], np.datetime64('2018-01-31'))


    def test_index_date_h(self) -> None:
        index = IndexDate.from_date_range('2017-12-15', '2018-02-15')

        post = index.loc['2018':'2018-01-15']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(len(post), 15)
        self.assertEqual(post[0], np.datetime64('2018-01-01'))
        self.assertEqual(post[-1], np.datetime64('2018-01-15'))


    def test_index_date_i(self) -> None:
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')

        post = index.loc['2017-12': '2018-01']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(len(post), 62)
        self.assertEqual(post[0], np.datetime64('2017-12-01'))
        self.assertEqual(post[-1], np.datetime64('2018-01-31'))


    def test_index_date_j(self) -> None:
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')

        post = index.loc['2017-12': '2018']  # type: ignore  # https://github.com/python/typeshed/pull/3024
        self.assertEqual(len(post), 77)
        self.assertEqual(post[0], np.datetime64('2017-12-01'))
        self.assertEqual(post[-1], np.datetime64('2018-02-15'))


    def test_index_date_k(self) -> None:
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')
        post = index.loc[['2017-12-10', '2018-02-06']]
        self.assertEqual(len(post), 2)
        self.assertEqual(post[0], np.datetime64('2017-12-10'))
        self.assertEqual(post[-1], np.datetime64('2018-02-06'))


    def test_index_date_m(self) -> None:
        index = IndexDate.from_date_range('2017-11-15', '2018-02-15')
        # NOTE: this type of selection should possibly not be permitted
        post = index.loc[['2017', '2018']]
        self.assertEqual(len(post), 93)
        self.assertEqual(post[0], np.datetime64('2017-11-15'))
        self.assertEqual(post[-1], np.datetime64('2018-02-15'))

    def test_index_date_n(self) -> None:
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


    def test_index_date_from_year_month_range_a(self) -> None:
        index = IndexDate.from_year_month_range('2017-12', '2018-03')

        self.assertEqual((index == '2017').sum(), 31)
        self.assertEqual((index == '2018').sum(), 90)

        self.assertEqual(
            [str(d) for d in np.unique(index.values.astype('datetime64[M]'))],
            ['2017-12', '2018-01', '2018-02', '2018-03'])


    def test_index_date_from_year_range_a(self) -> None:
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


    def test_index_date_series_a(self) -> None:

        s = Series(range(62),
                index=IndexDate.from_year_month_range('2017-12', '2018-01'))

        self.assertEqual(s.sum(), 1891)
        self.assertEqual(s.loc[s.index == '2018-01'].sum(), 1426)
        self.assertEqual(s.loc[s.index == '2017-12'].sum(), 465)

        self.assertEqual(s['2018-01-24'], 54)

        self.assertEqual(
                s['2018-01-28':].to_pairs(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                ((np.datetime64('2018-01-28'), 58), (np.datetime64('2018-01-29'), 59), (np.datetime64('2018-01-30'), 60), (np.datetime64('2018-01-31'), 61))
                )


    def test_index_year_month_a(self) -> None:
        idx1 = IndexYearMonth(('2018-01', '2018-06'))

        self.assertEqual(idx1.values.tolist(),
            [datetime.date(2018, 1, 1), datetime.date(2018, 6, 1)])


    def test_index_year_month_b(self) -> None:
        idx1 = IndexYearMonth(('2017-12', '2018-01', '2018-02', '2018-03', '2018-04'))

        post1 = idx1.loc[np.datetime64('2018-02'):]
        self.assertEqual(
                post1.values.tolist(),
                [datetime.date(2018, 2, 1), datetime.date(2018, 3, 1), datetime.date(2018, 4, 1)]
                )

        # a year datetime64
        post2 = idx1.loc[np.datetime64('2018'):]
        self.assertEqual(
                post2.values.tolist(),
                [datetime.date(2018, 1, 1), datetime.date(2018, 2, 1), datetime.date(2018, 3, 1), datetime.date(2018, 4, 1)]
                )


    def test_index_year_month_from_date_range_a(self) -> None:
        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15')
        self.assertEqual(len(index), 4)

        index = IndexYearMonth.from_date_range('2017-12-15', '2018-03-15', 2)
        self.assertEqual(len(index), 2)

    def test_index_year_month_from_year_month_range_a(self) -> None:

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


    def test_index_year_month_from_year_range_a(self) -> None:

        index = IndexYearMonth.from_year_range('2010', '2018')

        self.assertEqual(len(index), 108)
        self.assertEqual(str(index.min()), '2010-01')
        self.assertEqual(str(index.max()), '2018-12')

        index = IndexYearMonth.from_year_range('2010', '2018', 6)

        self.assertEqual(
                [str(d) for d in IndexYearMonth.from_year_range('2010', '2018', 6)],
                ['2010-01', '2010-07', '2011-01', '2011-07', '2012-01', '2012-07', '2013-01', '2013-07', '2014-01', '2014-07', '2015-01', '2015-07', '2016-01', '2016-07', '2017-01', '2017-07', '2018-01', '2018-07'])


    def test_index_year_from_date_range_a(self) -> None:

        index = IndexYear.from_date_range('2014-12-15', '2018-03-15')
        self.assertEqual(len(index), 5)

        index = IndexYear.from_date_range('2014-12-15', '2018-03-15', step=2)
        self.assertEqual([str(d) for d in index.values],
                ['2014', '2016', '2018'])


    def test_index_year_from_year_month_range_a(self) -> None:

        index = IndexYear.from_year_month_range('2014-12', '2018-03')
        self.assertEqual(len(index), 5)


    def test_index_year_from_year_range_a(self) -> None:

        index = IndexYear.from_year_range('2010', '2018')
        self.assertEqual(len(index), 9)


    def test_index_date_loc_to_iloc_a(self) -> None:

        index = IndexDate.from_date_range('2018-01-01', '2018-03-01')

        self.assertEqual(
                index.loc_to_iloc(np.datetime64('2018-02-11')),
                41)

        self.assertEqual(index.loc_to_iloc('2018-02-11'), 41)

        self.assertEqual(
                index.loc_to_iloc(slice('2018-02-11', '2018-02-24')),  # type: ignore
                slice(41, 55, None))





    def test_index_millisecond_a(self) -> None:

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
                idx.loc['2016-05-01T09:26:43.185':].values,  # type: ignore  # https://github.com/python/typeshed/pull/3024
                np.array(['2016-05-01T09:26:43.185', '2016-05-01T13:45:22.576',
       '2016-05-01T15:25:46.150'], dtype='datetime64[ms]'))

        self.assertAlmostEqualValues(idx.loc['2016-05'].values,
                np.array(['2016-05-01T00:00:33.483', '2016-05-01T03:02:03.584',
               '2016-05-01T09:26:43.185', '2016-05-01T13:45:22.576',
               '2016-05-01T15:25:46.150'], dtype='datetime64[ms]')
                )

        self.assertEqual(idx.loc['2016-05-01T00'].values,
                np.array(['2016-05-01T00:00:33.483'], dtype='datetime64[ms]'))




    def test_index_millisecond_b(self) -> None:
        # integer arguments are interpreted as milliseconds from the epoch
        idx = IndexMillisecond(range(10))
        self.assertAlmostEqualValues(idx.loc['1970-01-01T00:00:00.007':].values,  # type: ignore  # https://github.com/python/typeshed/pull/3024
                np.array(['1970-01-01T00:00:00.007', '1970-01-01T00:00:00.008',
               '1970-01-01T00:00:00.009'], dtype='datetime64[ms]'))


    def test_index_second_a(self) -> None:
        # integer arguments are interpreted as seconds from the epoch
        idx = IndexSecond(range(10))
        self.assertAlmostEqualValues(idx.loc['1970-01-01T00:00:07':].values,  # type: ignore  # https://github.com/python/typeshed/pull/3024
                np.array(['1970-01-01T00:00:07', '1970-01-01T00:00:08',
               '1970-01-01T00:00:09'], dtype='datetime64[s]')
                )


    def test_index_millisecond_series_a(self) -> None:

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

        self.assertEqual(s['2016-05-01T00:00:33.483':].values.tolist(),  # type: ignore  # https://github.com/python/typeshed/pull/3024
                [4, 5, 6, 7, 8])

        self.assertEqual(s['2016-05'].to_pairs(),
                ((np.datetime64('2016-05-01T00:00:33.483'), 4), (np.datetime64('2016-05-01T03:02:03.584'), 5), (np.datetime64('2016-05-01T09:26:43.185'), 6), (np.datetime64('2016-05-01T13:45:22.576'), 7), (np.datetime64('2016-05-01T15:25:46.150'), 8)))

        self.assertEqual(s['2016-05-01T09'].to_pairs(),
                ((np.datetime64('2016-05-01T09:26:43.185'), 6),))


    def test_index_millisecond_frame_a(self) -> None:

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

        idx1 = IndexMillisecond(f[1])
        self.assertAlmostEqualValues(idx1.values,
                np.array(['2016-04-28T04:22:12.226', '2016-04-28T16:29:21.320',
               '2016-04-28T17:36:13.733', '2016-04-30T20:21:07.848',
               '2016-05-01T00:00:33.483', '2016-05-01T03:02:03.584',
               '2016-05-01T09:26:43.185', '2016-05-01T13:45:22.576',
               '2016-05-01T15:25:46.150'], dtype='datetime64[ms]'))


        idx2 = IndexSecond(f[1])

        self.assertAlmostEqualValues(idx2.values,
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



