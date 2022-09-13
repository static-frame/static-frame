import unittest
from sys import getsizeof

from static_frame import Index
from static_frame import IndexGO
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_simple_index(self):
        idx = Index(('a', 'b', 'c'))
        self.assertEqual(getsizeof(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            None # for Index instance garbage collector overhead
        )))

    def test_object_index(self):
        idx = Index((1, 'b', (2, 3)))
        self.assertEqual(getsizeof(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            # _positions is an object dtype numpy array
            1, 'b',
            2, 3,
            (2, 3),
            idx._positions,
            idx._recache,
            idx._name,
            None # for Index instance garbage collector overhead
        )))

    def test_loc_is_iloc(self):
        idx = Index((0, 1, 2), loc_is_iloc=True)
        self.assertEqual(getsizeof(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            # idx._name, # both _map and _name are None
            None # for Index instance garbage collector overhead
        )))

    def test_empty_index(self):
        idx = Index(())
        self.assertEqual(getsizeof(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            None # for Index instance garbage collector overhead
        )))

    def test_name_adds_size(self):
        idx1 = Index(('a', 'b', 'c'))
        idx2 = idx1.rename('with_name')
        self.assertTrue(getsizeof(idx1) < getsizeof(idx2))

    def test_more_values_adds_size(self):
        idx1 = Index(('a', 'b', 'c'))
        idx2 = Index(('a', 'b', 'c', 'd'))
        self.assertTrue(getsizeof(idx1) < getsizeof(idx2))

    def test_more_nested_values_adds_size(self):
        idx1 = Index((1, 'b', (2, 3)))
        idx2 = Index((1, 'b', (2, 3, 4, 5)))
        self.assertTrue(getsizeof(idx1) < getsizeof(idx2))

    def test_more_doubly_nested_values_adds_size(self):
        idx1 = Index((1, 'b', ('c', (8, 9), 'd')))
        idx2 = Index((1, 'b', ('c', (8, 9, 10), 'd')))
        self.assertTrue(getsizeof(idx1) < getsizeof(idx2))

    def test_loc_is_iloc_reduces_size(self):
        # idx1 will be smaller since the _positions and _labels variables point to the same array
        idx1 = Index((0, 1, 2), loc_is_iloc=True)
        idx2 = Index((0, 1, 2))
        self.assertTrue(getsizeof(idx1) < getsizeof(idx2))

    def test_index_go(self):
        idx = IndexGO(('a', 'b', 'c'))
        self.assertEqual(getsizeof(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            'a', 'b', 'c',
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            None # for Index instance garbage collector overhead
        )))

    def test_index_go_after_append(self):
        idx = IndexGO(('a', 'b', 'c'))
        idx.append('d')
        self.assertEqual(getsizeof(idx), sum(getsizeof(e) for e in (
            idx._map,
            idx._labels,
            idx._positions,
            idx._recache,
            idx._name,
            'a', 'b', 'c', 'd',
            idx._labels_mutable,
            idx._labels_mutable_dtype,
            idx._positions_mutable_count,
            None # for Index instance garbage collector overhead
        )))

if __name__ == '__main__':
    unittest.main()