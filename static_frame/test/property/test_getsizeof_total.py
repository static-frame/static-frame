import typing as tp
from sys import getsizeof

from hypothesis import given

from static_frame import Frame
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import TypeBlocks
from static_frame.core.memory_measure import memory_total
from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    @given(sfst.get_index())
    def test_getsizeof_total_index(self, i: Index) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(i), sum((
            memory_total(i._map, seen=seen),
            memory_total(i._labels, seen=seen),
            memory_total(i._positions, seen=seen),
            memory_total(i._recache, seen=seen),
            memory_total(i._name, seen=seen),
            getsizeof(i) if id(i) not in seen else 0
        )))

    @given(sfst.get_index_go())
    def test_getsizeof_total_index_go(self, i: IndexGO) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(i), sum((
            memory_total(i._map, seen=seen),
            memory_total(i._labels, seen=seen),
            memory_total(i._positions, seen=seen),
            memory_total(i._recache, seen=seen),
            memory_total(i._name, seen=seen),
            memory_total(i._labels_mutable, seen=seen),
            memory_total(i._labels_mutable_dtype, seen=seen),
            memory_total(i._positions_mutable_count, seen=seen),
            getsizeof(i) if id(i) not in seen else 0
        )))

    @given(sfst.get_series())
    def test_getsizeof_total_series(self, s: Series) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(s), sum((
            memory_total(s.values, seen=seen),
            memory_total(s._index, seen=seen),
            memory_total(s._name, seen=seen),
            getsizeof(s) if id(s) not in seen else 0
        )))

    @given(sfst.get_type_blocks())
    def test_getsizeof_total_type_blocks(self, tb: TypeBlocks) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(tb), sum((
            memory_total(tb._blocks, seen=seen),
            memory_total(tb._index, seen=seen),
            getsizeof(tb) if id(tb) not in seen else 0,
        )))

    @given(sfst.get_frame_or_frame_go())
    def test_getsizeof_total_frame(self, f: Frame) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(memory_total(f), sum((
            memory_total(f._blocks, seen=seen),
            memory_total(f._columns, seen=seen),
            memory_total(f._index, seen=seen),
            memory_total(f._name, seen=seen),
            getsizeof(f) if id(f) not in seen else 0
        )))

if __name__ == '__main__':
    import unittest
    unittest.main()
