import typing as tp
from sys import getsizeof

from hypothesis import given

from static_frame import Frame
from static_frame import Index
from static_frame import IndexGO
from static_frame import Series
from static_frame import TypeBlocks
from static_frame.core.util import getsizeof_total
from static_frame.test.property import strategies as sfst
from static_frame.test.test_case import TestCase


class TestUnit(TestCase):

    @given(sfst.get_index())
    def test_getsizeof_total_index(self, i: Index) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(i), sum((
            getsizeof_total(i._map, seen=seen),
            getsizeof_total(i._labels, seen=seen),
            getsizeof_total(i._positions, seen=seen),
            getsizeof_total(i._recache, seen=seen),
            getsizeof_total(i._name, seen=seen),
            getsizeof(i) if id(i) not in seen else 0
        )))

    @given(sfst.get_index_go())
    def test_getsizeof_total_index_go(self, i: IndexGO) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(i), sum((
            getsizeof_total(i._map, seen=seen),
            getsizeof_total(i._labels, seen=seen),
            getsizeof_total(i._positions, seen=seen),
            getsizeof_total(i._recache, seen=seen),
            getsizeof_total(i._name, seen=seen),
            getsizeof_total(i._labels_mutable, seen=seen),
            getsizeof_total(i._labels_mutable_dtype, seen=seen),
            getsizeof_total(i._positions_mutable_count, seen=seen),
            getsizeof(i) if id(i) not in seen else 0
        )))

    @given(sfst.get_series())
    def test_getsizeof_total_series(self, s: Series) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(s), sum((
            getsizeof_total(s.values, seen=seen),
            getsizeof_total(s._index, seen=seen),
            getsizeof_total(s._name, seen=seen),
            getsizeof(s) if id(s) not in seen else 0
        )))

    @given(sfst.get_type_blocks())
    def test_getsizeof_total_type_blocks(self, tb: TypeBlocks) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(tb), sum((
            getsizeof_total(tb._blocks, seen=seen),
            getsizeof_total(tb._index, seen=seen),
            getsizeof_total(tb._shape, seen=seen),
            getsizeof_total(tb._row_dtype, seen=seen),
            getsizeof_total(tb._dtypes, seen=seen),
            getsizeof(tb) if id(tb) not in seen else 0,
        )))

    @given(sfst.get_frame_or_frame_go())
    def test_getsizeof_total_frame(self, f: Frame) -> None:
        seen: tp.Set[int] = set()
        self.assertEqual(getsizeof_total(f), sum((
            getsizeof_total(f._blocks, seen=seen),
            getsizeof_total(f._columns, seen=seen),
            getsizeof_total(f._index, seen=seen),
            getsizeof_total(f._name, seen=seen),
            getsizeof(f) if id(f) not in seen else 0
        )))

if __name__ == '__main__':
    import unittest
    unittest.main()
