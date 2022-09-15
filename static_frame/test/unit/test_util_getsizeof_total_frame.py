import typing as tp
import unittest
from sys import getsizeof

import frame_fixtures as ff

from static_frame.core.util import getsizeof_total
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):
    def test_simple(self) -> None:
        f = ff.parse('s(3,4)')
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_string_index(self) -> None:
        f = ff.parse('s(3,4)|i(I,str)|c(I,str)')
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_object_index(self) -> None:
        f = ff.parse('s(8,12)|i(I,object)|c(I,str)')
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_multiple_value_types(self) -> None:
        f = ff.parse('s(8,4)|i(I,object)|c(I,str)|v(object,int,bool,str)')
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            getsizeof(f)
        )))

    def test_frame_he_before_hash(self) -> None:
        f = ff.parse('s(3,4)').to_frame_he()
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            # getsizeof_total(f._hash, seen=seen), # not initialized yet
            getsizeof(f)
        )))

    def test_frame_he_after_hash(self) -> None:
        f = ff.parse('s(3,4)').to_frame_he()
        hash(f) # to initialize _hash
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            getsizeof_total(f._hash, seen=seen),
            getsizeof(f)
        )))

if __name__ == '__main__':
    unittest.main()
