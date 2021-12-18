

import unittest

import numpy as np

from static_frame.core.loc_map import LocMap
from static_frame.core.index import Index
from static_frame.core.index_datetime import IndexDate

from static_frame.core.util import NULL_SLICE

from static_frame.test.test_case import TestCase


class TestUnit(TestCase):


    def test_loc_map_a(self) -> None:
        idx = Index(['a', 'b', 'c'])
        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key='b',
                offset=0,
                partial_selection=False,
                )
        self.assertEqual(post1, 1)

        post2 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=NULL_SLICE,
                offset=None,
                partial_selection=False,
                )
        self.assertEqual(post2, NULL_SLICE)

    def test_loc_map_b(self) -> None:
        idx = Index(['a', 'b', 'c', 'd', 'e'])
        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=['b', 'd'],
                offset=None,
                partial_selection=False,
                )
        self.assertEqual(post1, [1, 3])

        post2 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=['b', 'd'],
                offset=10,
                partial_selection=False,
                )
        self.assertEqual(post2, [11, 13])




    def test_loc_map_slice_a(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-01', '1985-01-08')

        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01-01'), dt64('1985-01-04')),
                offset=None,
                partial_selection=False,
                )
        self.assertEqual(post1, slice(0, 4, None))

        post2 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01-01'), dt64('1985-01-04'), 2),
                offset=None,
                partial_selection=False,
                )
        self.assertEqual(post2, slice(0, 4, 2))


    def test_loc_map_slice_b(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-01', '1985-01-08')

        with self.assertRaises(RuntimeError):
            post1 = LocMap.loc_to_iloc(
                    label_to_pos=idx._map,
                    labels=idx._labels,
                    positions=idx._positions,
                    key=slice(dt64('1985-01-01'), dt64('1985-01-04'), dt64('1985-01-04')),
                    offset=None,
                    partial_selection=False,
                    )


    def test_loc_map_slice_c(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-01', '1985-01-08')

        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01-01'), dt64('1985-01-04')),
                offset=2,
                partial_selection=False,
                )
        self.assertEqual(post1, slice(2, 6, None))



    def test_loc_map_slice_d(self) -> None:
        dt64 = np.datetime64
        idx = IndexDate.from_date_range('1985-01-06', '1985-04-08')

        post1 = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key=slice(dt64('1985-01'), dt64('1985-03')),
                offset=None,
                partial_selection=False,
                )
        self.assertEqual(post1, slice(0, 85, None))

if __name__ == '__main__':
    unittest.main()

