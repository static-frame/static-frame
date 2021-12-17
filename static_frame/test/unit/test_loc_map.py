

import unittest

from static_frame.core.loc_map import LocMap
from static_frame.core.index import Index
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):


    def test_loc_map_a(self) -> None:

        idx = Index(['a', 'b', 'c'])
        post = LocMap.loc_to_iloc(
                label_to_pos=idx._map,
                labels=idx._labels,
                positions=idx._positions,
                key='b',
                offset=0,
                partial_selection=False,
                )
        self.assertEqual(post, 1)

if __name__ == '__main__':
    unittest.main()

