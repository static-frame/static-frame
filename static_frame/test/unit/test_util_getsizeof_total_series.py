import unittest
from sys import getsizeof

from static_frame import Series
from static_frame.core.util import getsizeof_total
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_simple(self):
        s = Series(('a', 'b', 'c'))
        seen = set()
        self.assertEqual(getsizeof_total(s), sum(getsizeof_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
        )) + getsizeof(s))

    def test_with_index(self):
        s = Series(('a', 'b', 'c'), index=(0, 1, 2))
        seen = set()
        self.assertEqual(getsizeof_total(s), sum(getsizeof_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
        )) + getsizeof(s))

    def test_object_values(self):
        s = Series(('a', (2, (3, 4), 5), 'c'))
        seen = set()
        self.assertEqual(getsizeof_total(s), sum(getsizeof_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
        )) + getsizeof(s))

    def test_with_name_is_larger(self):
        s1 = Series(('a', 'b', 'c'))
        s2 = s1.rename('named_series')
        self.assertTrue(getsizeof_total(s1) < getsizeof_total(s2))

    def test_larger_series_is_larger_a(self):
        s1 = Series(('a', 'b', 'c'))
        s2 = Series(('a', 'b', 'c', 'd'))
        self.assertTrue(getsizeof_total(s1) < getsizeof_total(s2))

    def test_larger_series_is_larger_b(self):
        s1 = Series(('a', 'b', 'c'))
        s2 = Series(('abc', 'def', 'ghi'))
        self.assertTrue(getsizeof_total(s1) < getsizeof_total(s2))

    def test_larger_nested_series_is_larger(self):
        s1 = Series(('a', (2, (4, 5), 8), 'c'))
        s2 = Series(('a', (2, (4, 5, 6), 8), 'c'))
        self.assertTrue(getsizeof_total(s1) < getsizeof_total(s2))

    def test_larger_index_is_larger(self):
        s1 = Series(('a', 'b', 'c'), index=(0, 1, 2))
        s2 = Series(('a', 'b', 'c'), index=('abc', 'def', 'ghi'))
        self.assertTrue(getsizeof_total(s1) < getsizeof_total(s2))

    def test_series_he_before_hash(self):
        s = Series(('a', 'b', 'c')).to_series_he()
        seen = set()
        self.assertEqual(getsizeof_total(s), sum(getsizeof_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
            # s._hash, # not initialized yet
        )) + getsizeof(s))

    def test_series_he_after_hash(self):
        s = Series(('a', 'b', 'c')).to_series_he()
        hash(s) # to initialize _hash
        seen = set()
        self.assertEqual(getsizeof_total(s), sum(getsizeof_total(e, seen=seen) for e in (
            s.values,
            s._index,
            s._name,
            s._hash,
        )) + getsizeof(s))

if __name__ == '__main__':
    unittest.main()