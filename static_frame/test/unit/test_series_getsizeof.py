import unittest
from sys import getsizeof

from static_frame import Series
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_simple(self):
        s = Series(('a', 'b', 'c'))
        self.assertEqual(getsizeof(s), sum(getsizeof(e) for e in (
            s.values,
            s._index,
            s._name,
            None # for Series instance garbage collector overhead
        )))

    def test_with_index(self):
        # This is notably larger than simple since the index is not loc_is_iloc optimised
        s = Series(('a', 'b', 'c'), index=(0, 1, 2))
        self.assertEqual(getsizeof(s), sum(getsizeof(e) for e in (
            s.values,
            s._index,
            s._name,
            None # for Series instance garbage collector overhead
        )))

    def test_object_values(self):
        s = Series(('a', (2, (3, 4), 5), 'c'))
        self.assertEqual(getsizeof(s), sum(getsizeof(e) for e in (
            'a',
            2,
            3, 4,
            (3, 4),
            5,
            (2, (3, 4), 5),
            'c',
            s.values,
            s._index,
            s._name,
            None # for Series instance garbage collector overhead
        )))

    def test_with_name_is_larger(self):
        s1 = Series(('a', 'b', 'c'))
        s2 = s1.rename('named_series')
        self.assertTrue(getsizeof(s1) < getsizeof(s2))

    def test_larger_series_is_larger_a(self):
        s1 = Series(('a', 'b', 'c'))
        s2 = Series(('a', 'b', 'c', 'd'))
        self.assertTrue(getsizeof(s1) < getsizeof(s2))

    def test_larger_series_is_larger_b(self):
        s1 = Series(('a', 'b', 'c'))
        s2 = Series(('abc', 'def', 'ghi'))
        self.assertTrue(getsizeof(s1) < getsizeof(s2))

    def test_larger_nested_series_is_larger(self):
        s1 = Series(('a', (2, (4, 5), 8), 'c'))
        s2 = Series(('a', (2, (4, 5, 6), 8), 'c'))
        self.assertTrue(getsizeof(s1) < getsizeof(s2))

    def test_larger_index_is_larger(self):
        s1 = Series(('a', 'b', 'c'), index=(0, 1, 2))
        s2 = Series(('a', 'b', 'c'), index=('abc', 'def', 'ghi'))
        self.assertTrue(getsizeof(s1) < getsizeof(s2))

if __name__ == '__main__':
    unittest.main()