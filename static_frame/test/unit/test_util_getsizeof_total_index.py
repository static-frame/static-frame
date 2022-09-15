import unittest
from sys import getsizeof

import numpy as np

from static_frame import Index
from static_frame import IndexGO
from static_frame.core.index_datetime import IndexDateGO
from static_frame.core.util import getsizeof_total
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_simple_index(self) -> None:
        idx = Index(('a', 'b', 'c'))
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            idx
        )))

    def test_object_index(self) -> None:
        idx = Index((1, 'b', (2, 3)))
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            # _positions is an object dtype numpy array
            1, 'b',
            2, 3,
            (2, 3),
            idx._positions,
            idx._recache,
            idx._name,
            idx
        )))

    def test_loc_is_iloc(self) -> None:
        idx = Index((0, 1, 2), loc_is_iloc=True)
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            # idx._name, # both _map and _name are None
            idx
        )))

    def test_empty_index(self) -> None:
        idx = Index(())
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            idx
        )))

    def test_name_adds_size(self) -> None:
        idx1 = Index(('a', 'b', 'c'))
        idx2 = idx1.rename('with_name')
        self.assertTrue(getsizeof_total(idx1) < getsizeof_total(idx2))

    def test_more_values_adds_size(self) -> None:
        idx1 = Index(('a', 'b', 'c'))
        idx2 = Index(('a', 'b', 'c', 'd'))
        self.assertTrue(getsizeof_total(idx1) < getsizeof_total(idx2))

    def test_more_nested_values_adds_size(self) -> None:
        idx1 = Index((1, 'b', (2, 3)))
        idx2 = Index((1, 'b', (2, 3, 4, 5)))
        self.assertTrue(getsizeof_total(idx1) < getsizeof_total(idx2))

    def test_more_doubly_nested_values_adds_size(self) -> None:
        idx1 = Index((1, 'b', ('c', (8, 9), 'd')))
        idx2 = Index((1, 'b', ('c', (8, 9, 10), 'd')))
        self.assertTrue(getsizeof_total(idx1) < getsizeof_total(idx2))

    def test_loc_is_iloc_reduces_size(self) -> None:
        # idx1 will be smaller since the _positions and _labels variables point to the same array
        idx1 = Index((0, 1, 2), loc_is_iloc=True)
        idx2 = Index((0, 1, 2))
        self.assertTrue(getsizeof_total(idx1) < getsizeof_total(idx2))

    def test_index_go(self) -> None:
        idx = IndexGO(('a', 'b', 'c'))
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            'a', 'b', 'c',
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            idx
        )))

    def test_index_go_after_append(self) -> None:
        idx = IndexGO(('a', 'b', 'c'))
        idx.append('d')
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            'a', 'b', 'c', 'd',
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            idx
        )))

    def test_index_datetime_go(self) -> None:
        idx = IndexDateGO.from_date_range('1994-01-01', '1995-01-01')
        self.assertEqual(getsizeof_total(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            *idx._labels_mutable, # Note: _labels_mutable is not nested
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            idx
        )))

if __name__ == '__main__':
    unittest.main()